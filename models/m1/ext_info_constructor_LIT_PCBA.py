import os
import time
from io import StringIO
from pathlib import Path
from queue import Queue

import pandas as pd
import torch
from mbapy.file import get_paths_with_extension, opts_file
from pymol import cmd
from tqdm import tqdm

try:
    from molvs import standardize_smiles
except:
    print('molvs not found')

from models.m1.ext_info_constructor import (FOLDSEEK_PATH, SeqIO,
                                            encode_protein_via_esm,
                                            extract_data_from_unimol_repr,
                                            load_ESM_model, load_SMILES_model,
                                            smiles_to_selfies)

try:
    from models.m1.ext_info_constructor import fn_MaskMol
except:
    pass


@torch.no_grad()
def generate_protein_dataset(data_root: str, result_path: Path,
                             model_name: str = 'esm3-open', device: str = 'cuda',
                             chain_process: str = 'mean_each_mean', max_len: int = 1024,
                             all_pdbid: bool = False):
    paths = get_paths_with_extension(data_root, ['.mol2'], name_substr='protein.mol2')
    if not all_pdbid:
        pdbid_df = pd.read_excel('data/LIT_PCBA.xlsx', 'Sheet2')
        paths = list(filter(lambda x: os.path.basename(x)[:4].upper() in pdbid_df['pdbid'].tolist(), paths))
    print(f'generate_protein_dataset: {model_name} {chain_process} {max_len} {result_path}')
    model = load_ESM_model(model_name, device)
    lens = []
    data = {}
    CAT_TOKEN = ';' if 'esm3' in model_name else ''
    for rec_idx, rec_path in tqdm(enumerate(paths), total=len(paths)):
        rec_id = f'{Path(rec_path).parent.name}_{rec_idx}'
        if model_name in {'esm2', 'esm2-3B', 'esm2-3B-33', 'esm2-3B-30', 'esm2-8M', 'esm3-open'}:
            cmd.reinitialize()
            cmd.load(rec_path, 'rec')
            fasta = cmd.get_fastastr('rec')
            chains = [str(record.seq) for record in SeqIO.parse(StringIO(fasta), "fasta")]
            lens.extend([len(chain) for chain in chains])
        elif model_name in 'SaProt_650M_AF2':
            from models._utils.foldseek_util import get_struc_seq

            # seq, foldseek_seq, combined_seq = parsed_seqs, we only need combined_seq
            if not os.path.exists(rec_path.replace(".mol2", ".pdb")):
                os.system(f'obabel -imol2 {rec_path} -opdb -O {rec_path.replace(".mol2", ".pdb")}')
            parsed_seqs = get_struc_seq(FOLDSEEK_PATH, rec_path.replace(".mol2", ".pdb"), None, plddt_mask=False)
            chains = [seq[-1] for seq in parsed_seqs.values()]
            lens.extend([len(chain) for chain in chains])
        elif model_name == 'ProSST-2048':
            cmd.reinitialize()
            cmd.load(rec_path, 'rec')
            cmd.remove('resn HOH'); cmd.remove('inorganic')
            chains = []
            for chain_code in cmd.get_chains('rec'):
                pdb_io = StringIO(cmd.get_pdbstr(f'rec and chain {chain_code}'))
                try:
                    inputs = model[-1].predict_from_pdb([pdb_io])[0]
                    chains.append(inputs)
                except Exception as e:
                    print(rec_id, chain_code, e)
            if not chains:
                print(rec_id, 'no valid chain')
                continue
            lens.extend([len(chain['aa_seq']) for chain in chains])
        else:
            raise ValueError(f'Unknown model_name: {model_name}')
        
        _check_chain_valid = lambda x, min_len: (isinstance(x, str) and len(x) > min_len) or (isinstance(x, dict) and 'aa_seq' in x and len(x['aa_seq']) > min_len)

        if chain_process == 'mean_each_mean':
            # encode chains separately and apply avg to each emebddings
            feats = torch.cat([encode_protein_via_esm(model_name, model, chain).mean(dim=0, keepdim=True) for chain in chains if _check_chain_valid(chain, 16)], dim=0)
            data[rec_id] = feats.mean(dim=0).cpu()
        elif chain_process == 'cat_each_mean':
            # encode chains separately and apply avg to each emebddings
            feats = torch.cat([encode_protein_via_esm(model_name, model, chain).mean(dim=0, keepdim=True) for chain in chains if _check_chain_valid(chain, 4)], dim=0)
            data[rec_id] = feats.cpu()
        elif chain_process == 'cat_all':
            # encode chains in one 'protein', and apply avg to the seq
            feats = encode_protein_via_esm(model_name, model, CAT_TOKEN.join(chains)).squeeze(0)
            data[rec_id] = feats.mean(dim=0).cpu()
        elif chain_process == 'cat_all_truncate':
            # encode chains in one 'protein', but turncate if too long
            feats = encode_protein_via_esm(model_name, model, CAT_TOKEN.join(chains)[:max_len]).squeeze(0)
            data[rec_id] = feats.cpu() # [L, D]
        else:
            raise ValueError(f'Unknown chain_process: {chain_process}')
    
    torch.save(data, result_path, pickle_protocol=5)
    torch.save(lens, 'lens.pt')
    

@torch.no_grad()
def generate_SMILES_dataset(data_root_queue: Queue,
                            result_que: Queue, model_name: str = 'PepDoRA',
                            subname: str = '.smi', resolution: int = 224,
                            n_iter: int = 1, batchsize: int = 256):
    # random.shuffle(total_smiles)
    model, tokenizer = load_SMILES_model(model_name, resolution=resolution)
    smiles_batch = []
    if model_name in {'GeminiMol', 'ImageMol', 'MaskMol_224', 'rdkit_vit_224',
                      'ChemBERTa_10M', 'ChemBERTa_100M_MLM', 'ChemBERTa_77M_MLM', 'ChemBERTa_77M_MTR',
                      'PepDoRA', 'MoLFormer', 'MolFormer', 'SELFormer'}:
        model = model.to('cuda')
        if model_name == 'GeminiMol':
            model.Encoder.readout.cuda()
    for _ in range(n_iter):
        # load SMILES
        data_root = data_root_queue.get()
        paths = get_paths_with_extension(data_root, [subname])
        total_smiles = []
        for path in paths:
            rec_id = Path(path).parent.name
            smiles_lst = list(map(lambda x:x.strip().split(' ')[0], opts_file(path, way='lines')))
            total_smiles.extend([(f'{rec_id}_{Path(path).stem}_{i}', smiles) for i, smiles in enumerate(smiles_lst)])
        # process SMILES
        pbar = tqdm(total=len(total_smiles))
        for cid, smiles in total_smiles:
            pbar.update(1)
            pbar.set_description(f'Processing {model_name} {rec_id}')
            data_label = not 'inactive' in cid
            if model_name == 'SELFormer':
                encoded_smiles = smiles_to_selfies(smiles, None, None)
                if encoded_smiles is None:
                    continue
                smiles = encoded_smiles
            smiles_batch.append((cid, smiles, data_label))
            if len(smiles_batch) < batchsize:
                continue
            cid_lst, smiles_lst, data_labels_lst = zip(*smiles_batch)
            smiles_lst = list(smiles_lst)
            if model_name in {'UniMol-SMILES'}:
                unimol_repr = model.get_repr(smiles_lst, return_atomic_reprs=True)
                result_que.put((cid_lst, extract_data_from_unimol_repr(unimol_repr), data_labels_lst))
            elif model_name == 'MolAI':
                inputs, _ = tokenizer(smiles_lst, max_smi_len=512)
                result_que.put((cid_lst, model.predict(inputs), data_labels_lst))
            elif model_name == 'GeminiMol':
                try:
                    input_tensor = model.sents2tensor(smiles_lst).to(device='cuda')
                    result_que.put((cid_lst, model.Encoder(input_tensor).cpu(), data_labels_lst))
                except Exception as e:
                    print(f'Error in encoding {cid_lst} with GeminiMol: {e}.')
            elif model_name == 'PepDoRA-token':
                inputs = tokenizer(
                    smiles_lst,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=256
                )      
                result_que.put((cid_lst, inputs['input_ids'], data_labels_lst))
            elif model_name == 'token':
                inputs = list(map(tokenizer, smiles_lst))
                result_que.put((cid_lst, inputs, data_labels_lst))
            elif model_name in {'MaskMol', 'ImageMol', 'MaskMol_224'}:
                result_que.put((cid_lst, fn_MaskMol(smiles_lst, model, tokenizer, resolution=resolution).cpu(), data_labels_lst))
            elif model_name == 'rdkit_vit_224':
                result_que.put((cid_lst, fn_MaskMol(smiles_lst, model, tokenizer, resolution=resolution).cpu(), data_labels_lst))
            else: # ChemBERTa_10M, ChemBERTa_100M_MLM, ChemBERTa_77M_MTR, ChemBERTa_77M_MLM, SELFormer, MoLFormer, PepDoRA
                inputs = tokenizer(
                    smiles_lst,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=256
                ).to('cuda')
                outputs = model(**inputs, output_hidden_states=True)
            
                result_que.put((cid_lst, dict(
                    hidden_states=outputs.hidden_states[-1].cpu(),
                    attention_mask=inputs['attention_mask'].cpu(),
                    ), data_labels_lst))
            smiles_batch.clear()
            # avoid take too much memory, wait for the queue to be small enough
            while result_que.qsize() > 32:
                time.sleep(0.1)
        result_que.put(None)


def standardize_smiles_parallel(data_root: str, subname: str = '.smi'):
    paths = get_paths_with_extension(data_root, [subname])
    total_smiles = sum(len(opts_file(path, way='lines')) for path in paths)
    pbar = tqdm(total=total_smiles)
    for smi_path in paths:
        rec_id = Path(smi_path).parent.name
        smiles_lst = list(map(lambda x:x.strip().split(' ')[0], opts_file(smi_path, way='lines')))
        std_smiles_lst = []
        pbar.set_description(f'Standardize {rec_id} {Path(smi_path).stem}')
        for smiles_idx, smiles in enumerate(smiles_lst):
            try:
                std_smiles_lst.append(standardize_smiles(smiles))
            except:
                std_smiles_lst.append(smiles)
            pbar.update(1)
            pbar.set_postfix({'sub-process': f'{smiles_idx}/{len(smiles_lst)}'})
        opts_file(f'{smi_path}_std', mode='w', way='str', data='\n'.join(std_smiles_lst))
        print(f'standardize {rec_id} {Path(smi_path).stem} done, save to {smi_path}_std')


if __name__ == '__main__':
    # generate_protein_dataset(f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased',
    #                          result_path=Path(f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased/protein_esm3-open_split.pt'),
    #                          model_name='esm3-open', device='cpu', chain_process='mean_each_mean', max_len=1024)
    # generate_protein_dataset(f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased',
    #                          result_path=Path(f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased/protein_SaProt_650M_AF2_mean_each_mean.pt'),
    #                          model_name='SaProt_650M_AF2', device='cpu', chain_process='mean_each_mean', max_len=1024)
    # generate_protein_dataset(f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased',
    #                          result_path=Path(f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased/protein_ProSST-2048_mean_each_mean.pt'),
    #                          model_name='ProSST-2048', device='cpu', chain_process='mean_each_mean', max_len=1024)
    # generate_protein_dataset(f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased',
    #                          result_path=Path(f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased/protein_esm2-mean_each_mean.pt'),
    #                          model_name='esm2', device='cpu', chain_process='mean_each_mean', max_len=1024)
    
    # standardize_smiles_paraller(f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased')
    data_que, root_que = Queue(), Queue()
    root_que.put(f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased')
    generate_SMILES_dataset(data_root_queue=root_que,
                            result_que=data_que, model_name='ChemBERTa_77M_MLM', subname='.smi_std')
    pass