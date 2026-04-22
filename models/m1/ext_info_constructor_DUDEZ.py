import os
import time
from io import StringIO
from pathlib import Path
from queue import Queue

import pandas as pd
import torch
from mbapy import TaskPool
from mbapy.file import get_paths_with_extension, opts_file
from pymol import cmd
from tqdm import tqdm

try:
    from rdkit import Chem
    from molvs import standardize_smiles
except:
    print('rdkit or molvs not found')

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
                             chain_process: str = 'mean_each_mean', max_len: int = 1024):
    paths = get_paths_with_extension(data_root, ['.pdb'], name_substr='rec.crg.pdb')
    print(f'generate_protein_dataset: {model_name} {chain_process} {max_len} {result_path}')
    model = load_ESM_model(model_name, device)
    lens = []
    data = {}
    CAT_TOKEN = ';' if 'esm3' in model_name else ''
    for rec_path in tqdm(paths, total=len(paths)):
        rec_id = f'{Path(rec_path).parent.name}'
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
    torch.cuda.empty_cache()
    torch.save(data, result_path, pickle_protocol=5)
    torch.save(lens, 'lens.pt')
    

@torch.no_grad()
def generate_SMILES_dataset(data_root: str,
                            result_que: Queue, model_name: str = 'PepDoRA',
                            subname: str = '.smi', resolution: int = 224):
    paths = get_paths_with_extension(data_root, [subname])
    # if Extrema or Godiocks provided, use DUDE_Z as positive
    if len(paths) == 1:
        # if Extrema or Godiocks provided decoy, use ligands in DUDE_Z
        # if Extrema or Godiocks provided ligand, use decoys in DUDE_Z
        name_substr = '_ligand_poses_std' if 'decoy' in paths[0] else '_decoy_poses_std'
        paths.extend(get_paths_with_extension(Path(data_root).parent / 'DUDE_Z', [subname], name_substr=name_substr))
        print(f'generate_SMILES_dataset: found 1 path, extend paths: {paths}')
    total_smiles = []
    for path in paths:
        rec_id = Path(path).parent.parent.name + '_' + Path(path).parent.name
        smiles_lst = list(map(lambda x:x.strip().split(' ')[0], opts_file(path, way='lines')))
        cid = f'{rec_id}_{Path(path).stem}_X'
        data_label = not 'decoy' in cid
        total_smiles.extend(list(map(lambda x: [cid, x, data_label], smiles_lst)))
    # random.shuffle(total_smiles)
    model, tokenizer = load_SMILES_model(model_name, resolution=resolution)
    smiles_batch = []
    if model_name in {'GeminiMol', 'ImageMol', 'MaskMol_224', 'rdkit_vit_224',
                      'ChemBERTa_10M', 'ChemBERTa_100M_MLM', 'ChemBERTa_77M_MLM', 'ChemBERTa_77M_MTR',
                      'PepDoRA', 'MoLFormer', 'MolFormer', 'SELFormer'}:
        model = model.to('cuda')
        if model_name == 'GeminiMol':
            model.Encoder.readout.cuda()       
    pbar = tqdm(total=len(total_smiles))
    for idx, (cid, smiles, data_label) in enumerate(total_smiles):
        pbar.update(1)
        pbar.set_description(f'Processing {model_name} {cid}')
        if model_name == 'SELFormer':
            encoded_smiles = smiles_to_selfies(smiles, None, None)
            if encoded_smiles is None:
                continue
            smiles = encoded_smiles
        smiles_batch.append((cid, smiles, data_label))
        if len(smiles_batch) < 256 and idx != len(total_smiles) - 1:
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


def convert_mol2_to_smiles(mol2_file, output_file: Path):
    """
    将多分子mol2文件转换为SMILES列表
    适用于 DUDE-Z 的 dudez_*_ligand_poses.mol2 和 decoy_poses.mol2
    """
    # 按 @<TRIPOS>MOLECULE 标记分割（每个分子以此开头）
    blocks = opts_file(mol2_file, 'r', way='str').split('@<TRIPOS>MOLECULE')
    results = []
    for i, block in tqdm(enumerate(blocks[1:], 1), total=len(blocks[1:]), leave=False):  # 跳过第一个空块
        if not block.strip():
            continue
        # 重新添加标记头
        mol_block = '@<TRIPOS>MOLECULE' + block
        # 尝试解析分子
        mol = Chem.MolFromMol2Block(mol_block, removeHs=False)
        if mol is not None:
            try:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                try:
                    smiles = standardize_smiles(smiles)
                except Exception as e:
                    print(f'Failed to standardize SMILES for {mol2_file}: {i}: {e}')
                results.append(smiles)
            except Exception as e:
                print(f"Failed to convert SMILES for {mol2_file}: {i}: {e}")
        else:
            print(f"Failed to parse molecule {i} in {mol2_file}")
    
    opts_file(str(output_file), mode='w', way='str', data='\n'.join(results))


def standardize_smiles_parallel(data_root: str, subname: str = '.mol2'):
    paths = get_paths_with_extension(data_root, [subname], name_substr='1pt0LD')
    pbar = tqdm(total=len(paths))
    pool = TaskPool('process', 16).start()
    for mol2_path in paths:
        mol2_path = Path(mol2_path)
        output_path = mol2_path.parent / (mol2_path.stem + '_std.smi')
        if output_path.exists():
            pbar.update(1)
            continue
        pbar.set_description(f'Standardize {mol2_path}')
        pool.add_task(None, convert_mol2_to_smiles, mol2_path, output_path)
        pool.wait_till_free()
        pbar.update(1)
    pool.wait_till_free()
    pool.close(5)


if __name__ == '__main__':
    # standardize_smiles_parallel(f'../EHIGN_dataset/DUDE_Z')
    
    # generate_protein_dataset(f'../EHIGN_dataset/DUDE_Z',
    #                          result_path=Path(f'../EHIGN_dataset/DUDE_Z/protein_esm3-open_split.pt'),
    #                          model_name='esm3-open', device='cpu', chain_process='mean_each_mean', max_len=1024)
    # generate_protein_dataset(f'../EHIGN_dataset/DUDE_Z',
    #                          result_path=Path(f'../EHIGN_dataset/DUDE_Z/protein_SaProt_650M_AF2_mean_each_mean.pt'),
    #                          model_name='SaProt_650M_AF2', device='cpu', chain_process='mean_each_mean', max_len=1024)
    # generate_protein_dataset(f'../EHIGN_dataset/DUDE_Z',
    #                          result_path=Path(f'../EHIGN_dataset/DUDE_Z/protein_ProSST-2048_mean_each_mean.pt'),
    #                          model_name='ProSST-2048', device='cpu', chain_process='mean_each_mean', max_len=1024)
    # generate_protein_dataset(f'../EHIGN_dataset/DUDE_Z',
    #                          result_path=Path(f'../EHIGN_dataset/DUDE_Z/protein_esm2-mean_each_mean.pt'),
    #                          model_name='esm2', device='cpu', chain_process='mean_each_mean', max_len=1024)
    # generate_protein_dataset(f'../EHIGN_dataset/DUDE_Z',
    #                          result_path=Path(f'../EHIGN_dataset/DUDE_Z/protein_esm2-3B-30_mean_each_mean.pt'),
    #                          model_name='esm2-3B-30', device='cpu', chain_process='mean_each_mean', max_len=1024)
    # generate_protein_dataset(f'../EHIGN_dataset/DUDE_Z',
    #                          result_path=Path(f'../EHIGN_dataset/DUDE_Z/protein_esm2-3B_mean_each_mean.pt'),
    #                          model_name='esm2-3B-33', device='cpu', chain_process='mean_each_mean', max_len=1024)
    # generate_protein_dataset(f'../EHIGN_dataset/DUDE_Z',
    #                          result_path=Path(f'../EHIGN_dataset/DUDE_Z/protein_esm2-3B-36_mean_each_mean.pt'),
    #                          model_name='esm2-3B', device='cpu', chain_process='mean_each_mean', max_len=1024)
    
    generate_SMILES_dataset(data_root=f'../EHIGN_dataset/DUDE_Z/AA2AR/Extrema',
                            result_que=Queue(), model_name='SELFormer', subname='_std.smi')
    pass