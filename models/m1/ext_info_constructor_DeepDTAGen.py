import os
from pathlib import Path

import pandas as pd
import torch
from mbapy.file import get_paths_with_extension
from tqdm import tqdm

try:
    from molvs import standardize_smiles
except:
    print('molvs not found')

from models.m1.ext_info_constructor import (encode_protein_via_esm,
                                            load_ESM_model, load_SMILES_model,
                                            smiles_to_selfies)

try:
    from models.m1.ext_info_constructor import fn_MaskMol
except:
    pass


@torch.no_grad()
def generate_protein_dataset(data_root: str, result_path: Path,
                             model_name: str = 'esm3-open', device: str = 'cuda',
                             chain_process: str = 'mean_each_mean', max_len: int = 1024):
    paths = get_paths_with_extension(data_root, ['.csv'])
    dfs = pd.concat([pd.read_csv(x) for x in paths], axis=0)
    seqs = dfs['target_sequence'].unique().tolist()
    print(f'generate_protein_dataset ({len(seqs)}):  {model_name} {chain_process} {max_len} {result_path}')
    model = load_ESM_model(model_name, device)
    data = {}
    if os.path.exists(result_path):
        data = torch.load(result_path, weights_only=False)
    CAT_TOKEN = ';' if 'esm3' in model_name else ''
    _check_chain_valid = lambda x, min_len: (isinstance(x, str) and len(x) > min_len) or (isinstance(x, dict) and 'aa_seq' in x and len(x['aa_seq']) > min_len)
    for seq in tqdm(seqs):
        if seq in data:
            continue
        if len(seq) > 2048:
            seq = seq[:2048]
        chains = [seq]
        if chain_process == 'mean_each_mean':
            # encode chains separately and apply avg to each emebddings
            feats = torch.cat([encode_protein_via_esm(model_name, model, chain).mean(dim=0, keepdim=True) for chain in chains if _check_chain_valid(chain, 1)], dim=0)
            data[seq] = feats.mean(dim=0).cpu()
        elif chain_process == 'cat_each_mean':
            # encode chains separately and apply avg to each emebddings
            feats = torch.cat([encode_protein_via_esm(model_name, model, chain).mean(dim=0, keepdim=True) for chain in chains if _check_chain_valid(chain, 4)], dim=0)
            data[seq] = feats.cpu()
        elif chain_process == 'cat_all':
            # encode chains in one 'protein', and apply avg to the seq
            feats = encode_protein_via_esm(model_name, model, CAT_TOKEN.join(chains)).squeeze(0)
            data[seq] = feats.mean(dim=0).cpu()
        elif chain_process == 'cat_all_truncate':
            # encode chains in one 'protein', but turncate if too long
            feats = encode_protein_via_esm(model_name, model, CAT_TOKEN.join(chains)[:max_len]).squeeze(0)
            data[seq] = feats.cpu() # [L, D]
        else:
            raise ValueError(f'Unknown chain_process: {chain_process}')
        torch.save(data, '/tmp/data.pt', pickle_protocol=5)
    
    torch.save(data, result_path, pickle_protocol=5)
    

@torch.no_grad()
def generate_SMILES_dataset(data_root: str, result_path: Path, model_name: str = 'PepDoRA', resolution: int = 224):
    paths = get_paths_with_extension(data_root, ['.csv'])
    dfs = pd.concat([pd.read_csv(x) for x in paths], axis=0)
    total_smiles = dfs['target_smiles'].unique().tolist()
    print(f'generate_SMILES_dataset ({len(total_smiles)}):  {model_name} {result_path}')
    model, tokenizer = load_SMILES_model(model_name, resolution=resolution)
    data = {}
    if model_name in {'GeminiMol', 'ImageMol', 'MaskMol_224', 'rdkit_vit_224',
                      'ChemBERTa_77M_MLM', 'ChemBERTa_77M_MTR',
                      'PepDoRA', 'MoLFormer', 'MolFormer', 'SELFormer'}:
        model = model.to('cuda')
        if model_name == 'GeminiMol':
            model.Encoder.readout.cuda()
    # process SMILES
    for smiles in tqdm(total_smiles):
        cid = smiles
        try:
            smiles = standardize_smiles(smiles)
        except:
            continue
        if model_name == 'SELFormer':
            smiles = smiles_to_selfies(smiles, None, None)
            if smiles is None:
                continue
        if model_name == 'MolAI':
            inputs, _ = tokenizer([smiles], max_smi_len=512)
            data[cid] = model.predict(inputs)
        elif model_name == 'GeminiMol':
            try:
                input_tensor = model.sents2tensor([smiles]).to(device='cuda')
                data[cid] = model.Encoder(input_tensor).cpu()
            except Exception as e:
                print(f'Error in encoding {cid} with GeminiMol: {e}.')
        elif model_name == 'PepDoRA-token':
            inputs = tokenizer(
                [smiles],
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=256
            )      
            data[cid] = inputs['input_ids']
        elif model_name == 'token':
            inputs = tokenizer([smiles])
            data[cid] = inputs['input_ids']
        elif model_name in {'MaskMol', 'ImageMol', 'MaskMol_224'}:
            data[cid] = fn_MaskMol([smiles], model, tokenizer, resolution=resolution).cpu()
        elif model_name == 'rdkit_vit_224':
            data[cid] = fn_MaskMol([smiles], model, tokenizer, resolution=resolution).cpu()
        else: # ChemBERTa_10M, ChemBERTa_100M_MLM, ChemBERTa_77M_MTR, ChemBERTa_77M_MLM, SELFormer, MoLFormer, PepDoRA
            inputs = tokenizer(
                [smiles],
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=256
            ).to('cuda')
            outputs = model(**inputs, output_hidden_states=True)
        
            data[cid] = dict(
                hidden_states=outputs.hidden_states[-1].cpu(),
                attention_mask=inputs['attention_mask'].cpu(),
                )
    torch.save(data, result_path, pickle_protocol=5)


if __name__ == '__main__':
    generate_protein_dataset(f'data/DeepDTAGen',
                             result_path=Path(f'../EHIGN_dataset/DeepDTAGen/protein_esm2_mean_each_mean.pt'),
                             model_name='esm2', device='cuda', chain_process='mean_each_mean', max_len=1024)
    generate_SMILES_dataset(f'data/DeepDTAGen',
                            result_path=Path(f'../EHIGN_dataset/DeepDTAGen/SMILES_ChemBERTa_77M_MLM.pt'),
                            model_name='ChemBERTa_77M_MLM')
    os._exit(0)