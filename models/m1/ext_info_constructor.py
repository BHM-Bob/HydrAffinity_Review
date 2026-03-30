import hashlib
import io
import os
import re
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from models._utils.img_pymol_constructor import (
        generate_rotated_views_tensor, process_image_tensors)
    from models._utils.img_rdkit_constructor import \
        smiles_to_hidden_state as fn_MaskMol
except Exception as e:
    print(f'img_pymol_constructor, img_rdkit_constructor are not imported: {e}')

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from Bio.PDB import PDBParser
from mbapy.base import put_err, split_list
from mbapy.file import get_paths_with_extension, opts_file
from mbapy.web import TaskPool
from pymol import cmd
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForMaskedLM, AutoTokenizer, BertTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          RobertaTokenizerFast, pipeline)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')
    from molvs import standardize_smiles
except ImportError as e:
    print(f'Some optional packages are not installed: {e}')
    
try:
    from esm.models.esm3 import ESM3
except ImportError as e:
    print(f'Some optional packages are not installed: {e}') 

np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.WARNING)


FOLDSEEK_PATH = os.path.expanduser('~/Desktop/USER_NAME/LFH/_soft/foldseek/bin/foldseek')
TMALIGN_PATH = os.path.expanduser('~/Desktop/USER_NAME/LFH/_soft/TMalign/TMalign')


def read_cids_from_df(df_file_name: str):
    return pd.read_csv(f'data/{df_file_name}')['pdbid'].tolist()

def get_cid2dir(data_root: str = '../EHIGN_dataset', check_name: str = '_protein.pdb'):
    # LRU cache
    cache_dir = os.path.expanduser('~/.cache/ext_info_constructor')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    ## check whether ~/.cache/ext_info_constructor.pkl exists as a LRU cache
    hash_suffix = hashlib.md5(f'{data_root}{check_name}'.encode('utf-8')).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f'cid2dir_{hash_suffix}.pkl')
    if os.path.exists(cache_path):
        cid2dir = opts_file(cache_path, 'rb', way='pkl')
        for name in ['train', 'valid', 'test2013', 'test2016', 'test2019']:
            df_cids = read_cids_from_df(f'{name}.csv')
            print(f'{name}: load {len(set(cid2dir.keys()) & set(df_cids))}/{len(df_cids)} cids from cache: {cache_path}')
        return cid2dir
    # run and cache the result
    cid2dir = {}
    for d_name in ['PDBbind_2013', 'PDBbind_2016', 'PDBbind_2019']:
        if d_name == 'PDBbind_2019':
            df_cids = read_cids_from_df('train.csv') + read_cids_from_df('valid.csv') + read_cids_from_df('test2019.csv')
        else:
            df_cids = read_cids_from_df(f'test{d_name[-4:]}.csv')
        cids = os.listdir(f'{data_root}/{d_name}/pdb')
        total_valid = 0
        for cid in tqdm(cids, desc=f'Check files for {d_name}', leave=False):
            if cid not in df_cids:
                continue
            if os.path.exists(os.path.join(f'{data_root}/{d_name}/pdb/{cid}', f"{cid}{check_name}")):
                cid2dir[cid] = f'{data_root}/{d_name}/pdb/{cid}/'
                total_valid += 1
        print(f'Find {total_valid}/{len(df_cids)} valid data in {d_name}.')
    opts_file(cache_path, 'wb', way='pkl', data=cid2dir)
    return cid2dir


def MolAI_vectorize(smiles: list[str]|np.ndarray, char_to_int: dict[str, int], max_smi_len: int = 256):
    one_hot = np.zeros((len(smiles), max_smi_len, len(char_to_int)), dtype=np.int8)
    for i, smile in tqdm(enumerate(smiles), disable=True):
        
        # Encode the start char
        one_hot[i, 0, char_to_int["!"]] = 1
        
        # Encode the rest of the chars
        for j, c in enumerate(smile):
            if c in char_to_int:
                one_hot[i, j+1, char_to_int[c]] = 1
            else:
                continue  # Skip if char not in char_to_int
                
        # Encode end char
        one_hot[i, len(smile)+1, char_to_int["$"]] = 1
        one_hot[i, len(smile)+2:, char_to_int["%"]] = 1
        
    # Return two, one for input and the other for output
    return one_hot[:, 1:, :], one_hot[:, :-1, :]


def smiles_to_selfies(smiles, cid, cid2dir):
    """将SMILES转换为SELFIES格式，失败时尝试多种备选方法"""
    import selfies
    try:
        return selfies.encoder(smiles)
    except selfies.EncoderError:
        try:
            from rdkit import Chem
            return selfies.encoder(Chem.MolToSmiles(Chem.MolFromMol2File(os.path.join(cid2dir[cid], f'{cid}.mol2')), 
                    isomericSmiles=True))
        except Exception:
            try:
                from rdkit import Chem
                return selfies.encoder(Chem.MolToSmiles(opts_file(os.path.join(cid2dir[cid], f'{cid}.rdkit'), 'rb', way='pkl')[0], 
                        isomericSmiles=True))
            except Exception as e:
                put_err(f'Error in encoding {cid} to SELFIES, {e}.')
                return None


def load_SMILES_model(model_name: str, **kwargs):
    MODEL_ROOT = Path(os.path.expanduser('~/Desktop/USER_NAME/LFH/250118-complex-MDS/'))
    if model_name == 'ChemBERTa_10M':
        model_path = Path(f'../EHIGN_dataset/_pretrained/PubChem10M_SMILES_BPE_450k')
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif model_name in {'ChemBERTa_100M_MLM', 'ChemBERTa_77M_MLM', 'ChemBERTa_77M_MTR'}:
        model_path = Path(f'../EHIGN_dataset/_pretrained/{model_name.replace("_", "-")}')
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif model_name == 'MoLFormer':
        model_path = Path(f'../EHIGN_dataset/_pretrained/MoLFormer-XL-both-10pct')
        model = AutoModel.from_pretrained(model_path, deterministic_eval=True, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_name in {'PepDoRA', 'PepDoRA-token'}:
        # TODO: do the peft change the base model?
        # peft==0.13.0 will load model exactly as expected, but the result is same
        from peft import PeftConfig, PeftModel
        base_model = str(MODEL_ROOT / 'GNN/pretrained/ChemBERTa-77M-MLM')
        adapter_model = str(MODEL_ROOT / 'GNN/pretrained/PepDoRA')
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, adapter_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    elif model_name == 'SELFormer':
        model_path = Path(f'../EHIGN_dataset/_pretrained/SELFormer/')
        tokenizer = RobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)
        config = RobertaConfig.from_pretrained(model_path, num_labels=2)
        model = RobertaModel.from_pretrained(model_path, config=config)
    elif model_name.startswith('UniMol'):
        os.environ['UNIMOL_WEIGHT_DIR'] = os.path.expanduser('~/Desktop/USER_NAME/LFH/_soft/UniMol')
        from unimol_tools import UniMolRepr
        model = UniMolRepr(data_type='molecule', 
                 remove_hs=False,
                 model_name='unimolv2', # avaliable: unimolv1, unimolv2
                 model_size='84m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                 max_atoms=256, # after github issue#14
                 )
        tokenizer = None
    elif model_name == 'MolAI':
        import tensorflow as tf
        from keras.models import load_model
        model_path = MODEL_ROOT / '../_soft/pretrained_models/MolAI'
        model = load_model(model_path / 'smi2lat_epoch_6.h5')
        char_to_int = opts_file(model_path / 'char_to_int.pkl', 'rb', way='pkl')
        new_model = tf.keras.models.clone_model(model, input_tensors=[tf.keras.Input(shape=(None, 34))])
        new_model.set_weights(model.get_weights())
        model = new_model
        tokenizer = partial(MolAI_vectorize, char_to_int=char_to_int)
    elif model_name == 'GeminiMol':
        from models._utils.geminimol.GeminiMol import BinarySimilarity
        params = opts_file('../EHIGN_dataset/_pretrained/GeminiMol/model_params.json', way='json')
        model = BinarySimilarity(model_name = 'GeminiMol', **params)
        model.load_state_dict(torch.load('../EHIGN_dataset/_pretrained/GeminiMol/GeminiMol.pt', map_location='cpu'))
        tokenizer = None
    elif model_name == '3Dimg':
        from models._utils.img_pymol_constructor import load_image_model
        model, tokenizer, _ = load_image_model('vit')
    elif model_name in {'MaskMol', 'ImageMol'}:
        from models._utils.img_rdkit_constructor import (get_img_transformer,
                                                         load_image_model)
        model = load_image_model(model_name)
        tokenizer = get_img_transformer(resolution=224)
    elif model_name == 'rdkit_vit':
        from models._utils.img_rdkit_constructor import (get_img_transformer,
                                                         load_image_model)
        model = load_image_model('rdkit_vit', resolution=kwargs.get('resolution', 224))
        tokenizer = get_img_transformer(resolution=kwargs.get('resolution', 224))
    else:
        raise ValueError(f'Unknown model name: {model_name}')
    return model, tokenizer


def extract_data_from_unimol_repr(unimol_repr: dict, idx: int = 0):
    length = unimol_repr['atomic_reprs'][idx].shape[0]
    if length <= 256:
        mask = F.pad(torch.ones(length, dtype=torch.long), (0, 256-length))
    else:
        mask = torch.ones(256, dtype=torch.long)
        unimol_repr['atomic_reprs'][idx] = unimol_repr['atomic_reprs'][idx][:256]
    return dict(cls=unimol_repr['cls_repr'][idx],
                hidden_states=unimol_repr['atomic_reprs'][idx], attention_mask=mask)


def _load_smiles_from_file(cid: str, root: str, return_mol: bool = False):
    # special case for ligand: 1-(2-[(R)-2,4-Dihydroxybutoxy]ethyl)-12-(5-ethyl-5-hydroxyheptyl)-1,12-dicarba-closo-dodecaborane
    if cid in {'3vjs', '3vjt'}:
        return '[BH]1234[BH]567[BH]89%10[BH]%11%12%13[BH]%141([BH]1%15%12[C]%12%16%17(CCCCC(O)(CC)CC)[BH]25([BH]68%12[BH]9%111%16)[BH]3%14%15%17)[C]47%10%13CCOC[C@@H](O)CCO', Chem.MolFromMol2File(os.path.join(root, f'{cid}_ligand_fix.mol2'), sanitize=False, removeHs=True)
    # try to load mol2 file first
    ligand_mol = Chem.MolFromMol2File(os.path.join(root, f'{cid}_ligand.mol2'), removeHs=True)
    # try to load fixed mol2 file which fixed manually
    if ligand_mol is None and os.path.exists(os.path.join(root, f'{cid}_ligand_fix.mol2')):
        ligand_mol = Chem.MolFromMol2File(os.path.join(root, f'{cid}_ligand_fix.mol2'), removeHs=True)
    # try to load sdf file
    if ligand_mol is None:
        ligand_mol = Chem.MolFromMolFile(os.path.join(root, f'{cid}_ligand.sdf'), removeHs=True)
    # try to load mol2 via pymol, then convert to rdkit mol through pdb block
    if ligand_mol is None:
        cmd.reinitialize()
        cmd.load(os.path.join(root, f'{cid}_ligand.mol2'), 'lig')
        cmd.remove('resn HOH')
        cmd.remove('hydrogens')
        ligand_mol = Chem.MolFromPDBBlock(cmd.get_pdbstr('lig'), removeHs=True)
    # try to standardize the mol2 file to '{cid}_ligand_std.pdb' via obabel
    if ligand_mol is None:
        os.system(f'obabel -imol2 "{os.path.join(root, f"{cid}_ligand.mol2")}" -O "{os.path.join(root, f"{cid}_ligand_std.pdb")}" --partialcharge gasteiger --canonical')
        if os.path.exists(os.path.join(root, f'{cid}_ligand_std.sdf')) and os.path.getsize(os.path.join(root, f'{cid}_ligand_std.sdf')) > 0:
            ligand_mol = Chem.MolFromMolFile(os.path.join(root, f'{cid}_ligand_std.sdf'), removeHs=True)
    if ligand_mol is None:
        print(f'Failed to read ligand molecule for {cid}, skip.')
        return None
    smiles = Chem.MolToSmiles(ligand_mol, isomericSmiles=True)
    try:
        smiles = standardize_smiles(smiles)
    except Exception as e:
        print(f'Failed to standardize SMILES for {cid}: {e}')
    if return_mol:
        return smiles, ligand_mol
    return smiles


@torch.no_grad()
def generate_SMILES_dataset(data_root: str,
                            result_path: Path, model_name: str = 'PepDoRA', resolution: int = 224,
                            cid2dir: dict = None):
    data = {}
    data_len = {}
    cid2dir = cid2dir or get_cid2dir(check_name='.rdkit')
    if os.path.exists(result_path) and os.path.isfile(result_path):
        print(f'Loading existing dataset from {result_path}')
        data = torch.load(result_path, map_location='cpu')
    if model_name == '3Dimg':
        result_dir = result_path.parent / 'SMILES_3Dimg'
        result_dir.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer = load_SMILES_model(model_name, resolution=resolution)
    if model_name in {'ChemBERTa_10M', 'ChemBERTa_100M_MLM', 'ChemBERTa_77M_MLM', 'ChemBERTa_77M_MTR',
                      'PepDoRA', 'MoLFormer', 'SELFormer', '3Dimg', 'MaskMol', 'ImageMol', 'rdkit_vit'}:
        model = model.to('cuda')
    print(f'Using model {model_name} to generate SMILES dataset (resolution={resolution}).')
    for cid_idx, cid in tqdm(enumerate(cid2dir), total=len(cid2dir)):        
        if cid in data:
            continue
        if model_name == 'UniMol-v3' and (result_path / f'{cid}.pt').exists():
            continue
        if model_name == 'UniMol-v4' and (result_path / f'{cid}_unimol_lig.pt').exists():# and (result_path / f'{cid}_unimol_poc.pt').exists():
            continue
        if model_name == 'UniMol-gen' and (result_path / f'{cid}_ligand.pt').exists():
            continue
        
        path1 = os.path.join(data_root, f'{cid}_str_6A.pt.gz')
        path2 = os.path.join(data_root, f'{cid}_str.pt.gz')
        if not os.path.exists(path1) and not os.path.exists(path2):
            lig_pack = _load_smiles_from_file(cid, cid2dir[cid], return_mol=True)
            if lig_pack is None:
                continue # waring was done in _load_smiles_from_file func
            smiles, ligand_mol = lig_pack
            if smiles is None and ligand_mol is None:
                print(f'Failed to load SMILES and mol for {cid}, skip.')
                continue
            if smiles is None and model_name != 'GeminiMol':
                print(f'Failed to load SMILES for {cid}, skip.')
                continue
        else:
            path = path1 if os.path.exists(path1) else path2
            p = subprocess.Popen(['pigz', '-dc', path], stdout=subprocess.PIPE)
            gz_data = p.stdout.read()  # 一次性读取所有数据
            p.wait()
            all_data = torch.load(io.BytesIO(gz_data))  # 使用BytesIO包装
            smiles = all_data['smiles']
        
        data_len[cid] = len(smiles)
        if model_name in {'UniMol-SMILES'}:
            smiles_list = [smiles]
            unimol_repr = model.get_repr(smiles_list, return_atomic_reprs=True)
            data[cid] = extract_data_from_unimol_repr(unimol_repr)
        elif model_name in {'UniMol-v3'}:
            if cid not in cid2dir:
                continue
            ligand, pocket = opts_file(os.path.join(cid2dir[cid], f'{cid}.rdkit'), 'rb', way='pkl')
            data = {}
            for name , mol in zip(['ligands', 'pockets'], [ligand, pocket]):
                atoms = [[atom.GetSymbol() for atom in mol.GetAtoms()]]
                coords = [mol.GetConformers()[0].GetPositions()]
                unimol_repr = model.get_repr(dict(atoms=atoms, coordinates=coords), return_atomic_reprs=True)
                data[name] = extract_data_from_unimol_repr(unimol_repr)
            torch.save(data, result_path / f'{cid}.pt', pickle_protocol=5)
        elif model_name in {'UniMol-v4'}:
            data = {}
            # load from 6A.pt.gz first
            for name in ['ligands']:#, 'pockets']:
                data[name] = []
                if name not in all_data or all_data[name] is None or len(all_data[name]) == 0:
                    continue
                mols = all_data[name]
                for mol_batch in split_list(mols, 2):
                    atoms = [[atom.GetSymbol() for atom in mol.GetAtoms()] for mol in mol_batch]
                    coords = [mol.GetConformers()[0].GetPositions() for mol in mol_batch]
                    unimol_repr = model.get_repr(dict(atoms=atoms, coordinates=coords), return_atomic_reprs=True)
                    data[name].extend([extract_data_from_unimol_repr(unimol_repr, i) for i in range(len(mol_batch))])
            # if fail, load from cid.rdkit
            if len(data['ligands']) == 0 and cid in cid2dir:
                lig_mol, _ = opts_file(os.path.join(cid2dir[cid], f'{cid}.rdkit'), 'rb', way='pkl')
                atoms = [[atom.GetSymbol() for atom in lig_mol.GetAtoms()]]
                coords = [lig_mol.GetConformers()[0].GetPositions()]
                unimol_repr = model.get_repr(dict(atoms=atoms, coordinates=coords), return_atomic_reprs=True)
                data['ligands'].append(extract_data_from_unimol_repr(unimol_repr, 0))
            if data['ligands']:
                torch.save(data['ligands'], os.path.join(result_path, f'{cid}_unimol_lig.pt'), pickle_protocol=5)
            else:
                print(f'\nNo ligand found for {cid}\n')
            # torch.save(data['pockets'], os.path.join(result_path, f'{cid}_unimol_poc.pt'), pickle_protocol=5)
        elif model_name == 'UniMol-gen':
            if cid not in cid2dir:
                continue
            ligand, pocket = opts_file(os.path.join(cid2dir[cid], f'{cid}.rdkit'), 'rb', way='pkl')
            ligands, data = [], []
            if not (result_path / f'{cid}_ligand_atoms_coords.pt').exists():
                mol_h = Chem.AddHs(ligand)  # 添加氢原子以保证构象完整性
                print(f'Generating conformers for {cid} ligand with {mol_h.GetNumAtoms()} atoms...')
                try:
                    cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=100, pruneRmsThresh=0.5, numThreads=0)
                    mol_noH = Chem.RemoveHs(mol_h)  # 去氢但保留构象坐标
                except Exception as e:
                    put_err(f'Error in generating conformers for {cid} ligand: {e}, use original conformer.')
                    cids = []
                for conf_id in cids:
                    conformer = mol_noH.GetConformer(conf_id)
                    atoms = [[atom.GetSymbol() for atom in mol_noH.GetAtoms()]]
                    coords = [conformer.GetPositions()]
                    ligands.append(dict(atoms=atoms, coordinates=coords))
                if not ligands:
                    ligands.append(dict(atoms=[[atom.GetSymbol() for atom in ligand.GetAtoms()]],
                                        coordinates=[ligand.GetConformers()[0].GetPositions()]))
                torch.save(ligands, result_path / f'{cid}_ligand_atoms_coords.pt', pickle_protocol=5)
            else:
                for lig_data in torch.load(result_path / f'{cid}_ligand_atoms_coords.pt'):
                    unimol_repr = model.get_repr(lig_data, return_atomic_reprs=True)
                    data.append(extract_data_from_unimol_repr(unimol_repr, 0))
            torch.save(data, result_path / f'{cid}_ligand.pt', pickle_protocol=5)
        elif model_name == 'MolAI':
            inputs, _ = tokenizer([smiles], max_smi_len=512)
            data[cid] = model.predict(inputs)
        elif model_name == 'GeminiMol':
            try:
                if smiles is None or Chem.MolFromSmiles(smiles) is None:
                    input_tensor = model.mols2tensor([ligand_mol])
                else:
                    input_tensor = model.sents2tensor([smiles])
                data[cid] = model.Encoder(input_tensor)
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
            data_len[cid] = inputs['attention_mask'].sum().item()        
            data[cid] = inputs['input_ids']
        elif model_name == '3Dimg':
            if (result_dir / f'{cid}_ligand.pt').exists():
                img_tensors = torch.load(result_dir / f'{cid}_ligand.pt')
                for k in img_tensors:
                    img_tensors[k] = [i.to(torch.float32) / 255 for i in img_tensors[k]]
            else:
                pdb_file = os.path.join(cid2dir[cid], f'{cid}_ligand.pdb')
                img_tensors = generate_rotated_views_tensor(
                    pdb_file, image_size=224,  # 为了与图像模型匹配，使用224x224
                    representations=('ball_stick', 'stick', 'surface')
                )
            hidden_states = process_image_tensors(img_tensors, model, tokenizer, 'cuda', batch_size=4)
            if not (result_dir / f'{cid}_ligand.pt').exists():
                # transfer img_tensors to int16 by multiply 255 to save disk space
                for k in img_tensors:
                    img_tensors[k] = [(i*255).to(torch.int16) for i in img_tensors[k]]
                torch.save(img_tensors, result_dir / f'{cid}_ligand.pt', pickle_protocol=5)
            data[cid] = hidden_states
        elif model_name in {'MaskMol', 'ImageMol'}:
            data[cid] = fn_MaskMol(smiles, model, tokenizer, resolution=224).cpu()
        elif model_name == 'rdkit_vit':
            data[cid] = fn_MaskMol(smiles, model, tokenizer, resolution=resolution).cpu()
        else: # ChemBERTa_10M, ChemBERTa_100M_MLM, ChemBERTa_77M_MTR, ChemBERTa_77M_MLM, SELFormer, MoLFormer, PepDoRA
            if model_name == 'SELFormer':
                encoded_smiles = smiles_to_selfies(smiles, cid, cid2dir)
                if encoded_smiles is None:
                    continue
                smiles = encoded_smiles
            inputs = tokenizer(
                [smiles],
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=256
            ).to('cuda')
            data_len[cid] = inputs['attention_mask'].sum().item()
            outputs = model(**inputs, output_hidden_states=True)
        
            data[cid] = dict(
                hidden_states=outputs.hidden_states[-1].cpu(),
                attention_mask=inputs['attention_mask'].cpu(),
            )
            
        # save data every 3000 cids to avoid crash and get nothing
        if cid_idx % 3000 == 0:
            torch.save(data, result_path, pickle_protocol=5)
            
    # report what have done
    if model_name in {'UniMol-v3', 'UniMol-v4', 'UniMol-gen'}:
        data = {k: True for k in list(map(lambda x: x[:4], os.listdir(result_path)))}
    for name in ['train', 'valid', 'test2013', 'test2016', 'test2019']:
        df_cids = read_cids_from_df(f'{name}.csv')
        print(f'{name}: done {len(set(data.keys()) & set(df_cids))}/{len(df_cids)}')
    if model_name in {'PepDoRA', 'ChemBERTa_10M', 'ChemBERTa_100M_MLM', 'ChemBERTa_77M_MLM', 'ChemBERTa_77M_MTR', 
                      'MoLFormer', 'SELFormer', 'MolAI', 'PepDoRA-token', '3Dimg', 'MaskMol', 'ImageMol', 'GenminiMol'}:
        torch.save(data_len, result_path.parent / f'{result_path.stem}_len.pt')
        torch.save(data, result_path, pickle_protocol=5)


def load_ESM_model(model_name: str = 'esm3-open', device: str = 'cuda'):
    if model_name == 'esm2':
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.eval()
        return model, alphabet.get_batch_converter()
    elif model_name in {'esm2-3B', 'esm2-3B-33', 'esm2-3B-30'}:
        import esm
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D() # dim=1280
        model.eval()
        return model, alphabet.get_batch_converter()
    elif model_name == 'esm2-8M':
        import esm
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # dim=320
        model.eval()
        return model, alphabet.get_batch_converter()
    elif model_name == 'esm3-open':
        return ESM3.from_pretrained(model_name).to(device).to(torch.float32)
    elif model_name == 'SaProt_650M_AF2':
        from transformers import EsmForMaskedLM, EsmTokenizer
        model_path = "../EHIGN_dataset/_pretrained/SaProt_650M_AF2"
        tokenizer = EsmTokenizer.from_pretrained(model_path)
        model = EsmForMaskedLM.from_pretrained(model_path)
        model.eval()
        return model, tokenizer
    elif model_name == 'ProSST-2048':
        from models._utils.prosst.structure.get_sst_seq import SSTPredictor
        predictor = SSTPredictor(structure_vocab_size=2048)
        model_path = "../EHIGN_dataset/_pretrained/ProSST-2048"
        model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        return model, tokenizer, predictor
    elif model_name == 'aido_protein_16b':
        from modelgenerator.tasks import Embed
        model = Embed.from_config({"model.backbone": "aido_protein_16b"}).eval()
        return model

def calculate_model_size(model, model_name: str, model_type: str):
    """计算模型参数大小和内存占用"""
    total_params = 0
    trainable_params = 0
    
    # 处理不同类型的模型
    if hasattr(model, 'parameters'):
        # PyTorch模型
        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
    elif hasattr(model, 'trainable_variables'):
        # TensorFlow/Keras模型
        import tensorflow as tf
        for var in model.trainable_variables:
            total_params += var.shape.num_elements()
            trainable_params += var.shape.num_elements()
        # 添加非可训练参数
        for var in model.non_trainable_variables:
            total_params += var.shape.num_elements()
    elif hasattr(model, 'model') and hasattr(model.model, 'parameters'):
        # UniMolRepr等包装类
        for param in model.model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
    else:
        # 无法计算参数
        return {
        }
    
    # 计算内存占用（假设float32，每个参数4字节）
    memory_bytes = total_params * 4
    memory_mb = memory_bytes / (1024 * 1024)
    memory_gb = memory_bytes / (1024 * 1024 * 1024)
    
    return {
        'model_name': model_name,
        'model_type': model_type,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'memory_bytes': memory_bytes,
        'memory_mb': memory_mb,
        'memory_gb': memory_gb
    }


def load_all_models_and_calculate_size(skip_existing: bool = True):
    """加载所有配体受体预训练模型到CPU并计算模型大小
    
    Args:
        skip_existing: 是否跳过已有记录，默认为True
    """
    output_path = Path('../EHIGN_dataset/model_param.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取已有记录
    existing_results = []
    if skip_existing and output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            existing_results = existing_df.to_dict('records')
            print(f"发现已有记录: {len(existing_results)} 条")
        except Exception as e:
            print(f"读取已有记录失败: {e}")
    
    # 配体模型列表（SMILES模型）
    ligand_models = [
        'ChemBERTa_10M',
        'ChemBERTa_100M_MLM', 
        'ChemBERTa_77M_MLM',
        'ChemBERTa_77M_MTR',
        'MoLFormer',
        'PepDoRA',
        'PepDoRA-token',
        'SELFormer',
        'UniMol',
        'MolAI',
        'GeminiMol',
        '3Dimg',
        'MaskMol',
        'ImageMol',
        'rdkit_vit'
    ]
    
    # 受体模型列表（ESM模型）
    receptor_models = [
        'esm2',
        'esm2-3B',
        'esm2-3B-33',
        'esm2-3B-30',
        'esm2-8M',
        'esm3-open',
        'SaProt_650M_AF2',
        'ProSST-2048',
    ]
    
    # 检查哪些模型已有记录
    existing_models = {record['model_name'] for record in existing_results if 'model_name' in record}
    
    results = existing_results.copy()
    processed_models = set(existing_models)
    
    print("开始加载配体模型...")
    for model_name in ligand_models:
        if skip_existing and model_name in existing_models:
            print(f"跳过已有记录: {model_name}")
            continue
            
        try:
            print(f"正在加载 {model_name}...")
            model, tokenizer = load_SMILES_model(model_name)
            
            # 确保模型在CPU上
            if hasattr(model, 'to'):
                model = model.to('cpu')
            
            # 计算模型大小
            size_info = calculate_model_size(model, model_name, 'ligand')
            results.append(size_info)
            processed_models.add(model_name)
            print(f"  {model_name}: {size_info['total_params']:,} 参数, {size_info['memory_gb']:.2f} GB")
            
            # 清理内存
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"加载 {model_name} 失败: {e}")
    
    print("\n开始加载受体模型...")
    for model_name in receptor_models:
        if skip_existing and model_name in existing_models:
            print(f"跳过已有记录: {model_name}")
            continue
            
        try:
            print(f"正在加载 {model_name}...")
            
            # 特殊处理不同的模型返回类型
            if model_name == 'ProSST-2048':
                model, tokenizer, predictor = load_ESM_model(model_name, device='cpu')
            elif model_name == 'esm3-open':
                model = load_ESM_model(model_name, device='cpu')
            else:
                model, tokenizer = load_ESM_model(model_name, device='cpu')
            
            # 计算模型大小
            size_info = calculate_model_size(model, model_name, 'receptor')
            results.append(size_info)
            processed_models.add(model_name)
            print(f"  {model_name}: {size_info['total_params']:,} 参数, {size_info['memory_gb']:.2f} GB")
            
            # 清理内存
            del model
            if tokenizer:
                del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"加载 {model_name} 失败: {e}")
            results.append({
                'model_name': model_name,
                'model_type': 'receptor',
                'total_params': 0,
                'trainable_params': 0,
                'memory_bytes': 0,
                'memory_mb': 0,
                'memory_gb': 0,
                'error': str(e)
            })
            processed_models.add(model_name)
    
    # 保存结果到CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    print(f"\n结果已保存到: {output_path}")
    print(f"总共处理了 {len(results)} 个模型记录")
    print(f"本次新增处理了 {len(processed_models) - len(existing_models)} 个模型")
    
    # 打印统计信息
    ligand_df = df[df['model_type'] == 'ligand']
    receptor_df = df[df['model_type'] == 'receptor']
    
    print(f"\n配体模型统计:")
    print(f"  成功加载: {len(ligand_df[ligand_df['total_params'] > 0])}/{len(ligand_df)}")
    print(f"  总参数数: {ligand_df['total_params'].sum():,}")
    print(f"  总内存占用: {ligand_df['memory_gb'].sum():.2f} GB")
    
    print(f"\n受体模型统计:")
    print(f"  成功加载: {len(receptor_df[receptor_df['total_params'] > 0])}/{len(receptor_df)}")
    print(f"  总参数数: {receptor_df['total_params'].sum():,}")
    print(f"  总内存占用: {receptor_df['memory_gb'].sum():.2f} GB")
    
    return df


@torch.no_grad()
def encode_protein_via_esm3(model: 'ESM3', sequence: str):
    from esm.sdk.api import ESMProtein, LogitsConfig
    protein = ESMProtein(sequence=sequence)
    if len(sequence) <= 512 and model.device.type == 'cpu':
        model = model.to('cuda')
    if len(sequence) > 512 and model.device.type == 'cuda':
        model = model.to('cpu')
    protein_tensor = model.encode(protein)
    logits_output = model.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    return logits_output.embeddings

@torch.no_grad()
def encode_protein_via_esm2(model: 'ESM2', sequence: str, n_layer: int = 33):
    # modified from demo code of https://github.com/facebookresearch/esm
    model, batch_converter = model
    model_device = model.emb_layer_norm_after.weight.device.type
    data = [('chain', sequence.replace('?', '<unk>'))]
    if len(sequence) <= 512 and model_device == 'cpu':
        model = model.to('cuda')
    if len(sequence) > 512 and model_device == 'cuda':
        model = model.to('cpu')
    model_device = model.emb_layer_norm_after.weight.device.type
    _, _, batch_tokens = batch_converter(data)
    results = model(batch_tokens.to(device=model_device), repr_layers=[n_layer], return_contacts=True)
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    return results['representations'][n_layer].squeeze(0)[1:].cpu()

@torch.no_grad()
def encode_protein_via_SaProt(model: 'SaProt_650M_AF2', sequence: str):
    model, tokenizer = model
    model_device = model.lm_head.dense.weight.device.type
    if len(sequence) <= 1024 and model_device == 'cpu':
        model = model.to('cuda')
    if len(sequence) > 1024 and model_device == 'cuda':
        model = model.to('cpu')
    model_device = model.lm_head.dense.weight.device.type
    inputs = tokenizer(sequence, return_tensors="pt").to(model_device)
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1][:, 1:-1].squeeze(0).cpu() # [L, 1280], if is logits: [L, 446]

@torch.no_grad()
def encode_protein_via_ProSST(model: 'ProSST-2048', inputs: dict):
    model, tokenizer, _ = model
    model_device = model.prosst.embeddings.word_embeddings.weight.device.type
    if len(inputs['aa_seq']) <= 512 and model_device == 'cpu':
        model = model.to('cuda')
    if len(inputs['aa_seq']) > 512 and model_device == 'cuda':
        model = model.to('cpu')
    model_device = model.prosst.embeddings.word_embeddings.weight.device.type
    aa_inputs = tokenizer(inputs['aa_seq'], return_tensors="pt").to(model_device)
    aa_inputs = {k: v.to(model_device) for k, v in aa_inputs.items()}
    inputs['2048_sst_seq'] = [i+3 for i in inputs['2048_sst_seq']] # from https://github.com/ai4protein/ProSST/blob/main/zero_shot/score_mutant.ipynb
    ss_input_ids = torch.tensor([1, *inputs['2048_sst_seq'], 2], dtype=torch.long, device=model_device).unsqueeze(0)
    outputs = model(**aa_inputs, ss_input_ids=ss_input_ids, output_hidden_states=True, return_dict=True)
    return outputs['hidden_states'][-1].squeeze(0)[1:-1].cpu() # [L, 768]

def encode_protein_via_esm(model_name, model, sequence):
    if model_name in {'esm2', 'esm2-3B', 'esm2-3B-33', 'esm2-3B-30', 'esm2-8M'}:
        # esm 3.1.1 requires transformers<4.47.0
        name2layer = {'esm2': 33, 'esm2-3B': 36, 'esm2-3B-33': 33, 'esm2-3B-30': 30, 'esm2-8M': 6}
        return encode_protein_via_esm2(model, sequence, n_layer=name2layer[model_name]).cpu() # [L, 1280]
    elif model_name == 'esm3-open':
        # this alse named esm
        return encode_protein_via_esm3(model, sequence).squeeze(0).cpu() # [L, 1536]
    elif model_name == 'SaProt_650M_AF2':
        return encode_protein_via_SaProt(model, sequence).cpu() # [L, 1280]
    elif model_name == 'ProSST-2048':
        return encode_protein_via_ProSST(model, sequence).cpu() # [L, 768]
    else:
        raise ValueError(f'Unknown model_name: {model_name}')

@torch.no_grad()
def generate_protein_dataset(data_root: str, result_path: Path,
                             model_name: str = 'esm3-open', device: str = 'cuda',
                             chain_process: str = 'mean_each_mean', max_len: int = 1024,
                             cid2dir: dict = None):
    cid2dir = cid2dir or get_cid2dir()
    print(f'generate_protein_dataset: {model_name} {chain_process} {max_len} {result_path}')
    model = load_ESM_model(model_name, device)
    if result_path.exists():
        data = torch.load(result_path)
    else:
        data = {}
    lens = []
    CAT_TOKEN = ';' if 'esm3' in model_name else ''
    for idx, cid in tqdm(enumerate(cid2dir), total=len(cid2dir)):
        if cid in data:
            continue
        if model_name in {'esm2', 'esm2-3B', 'esm2-3B-33', 'esm2-3B-30', 'esm2-8M', 'esm3-open'}:
            if not os.path.exists(os.path.join(cid2dir[cid], f'{cid}_protein.pdb')):
                print(f'{cid} not exists, skip')
                continue
            cmd.reinitialize()
            cmd.load(os.path.join(cid2dir[cid], f'{cid}_protein.pdb'), 'rec')
            fasta = cmd.get_fastastr('rec')
            chains = [str(record.seq) for record in SeqIO.parse(StringIO(fasta), "fasta")]
            lens.extend([len(chain) for chain in chains])
        elif model_name in 'SaProt_650M_AF2':
            from models._utils.foldseek_util import get_struc_seq
            pdb_path = os.path.join(cid2dir[cid], f'{cid}_protein.pdb')
            # seq, foldseek_seq, combined_seq = parsed_seqs, we only need combined_seq
            parsed_seqs = get_struc_seq(FOLDSEEK_PATH, pdb_path, None, plddt_mask=False)
            chains = [seq[-1] for seq in parsed_seqs.values()]
            lens.extend([len(chain) for chain in chains])
        elif model_name == 'ProSST-2048':
            cmd.reinitialize()
            cmd.load(os.path.join(cid2dir[cid], f'{cid}_protein.pdb'), 'rec')
            cmd.remove('resn HOH'); cmd.remove('inorganic')
            chains = []
            for chain_code in cmd.get_chains('rec'):
                pdb_io = StringIO(cmd.get_pdbstr(f'rec and chain {chain_code}'))
                try:
                    inputs = model[-1].predict_from_pdb([pdb_io])[0]
                    chains.append(inputs)
                except Exception as e:
                    print(cid, chain_code, e)
            if not chains:
                print(cid, 'no valid chain')
                continue
            lens.extend([len(chain['aa_seq']) for chain in chains])
        else:
            raise ValueError(f'Unknown model_name: {model_name}')
        
        _check_chain_valid = lambda x, min_len: (isinstance(x, str) and len(x) > min_len) or (isinstance(x, dict) and 'aa_seq' in x and len(x['aa_seq']) > min_len)

        if chain_process == 'mean_each_mean':
            # encode chains separately and apply avg to each emebddings
            feats = torch.cat([encode_protein_via_esm(model_name, model, chain).mean(dim=0, keepdim=True) for chain in chains if _check_chain_valid(chain, 16)], dim=0)
            data[cid] = feats.mean(dim=0).cpu()
        elif chain_process == 'cat_each_mean':
            # encode chains separately and apply avg to each emebddings
            feats = torch.cat([encode_protein_via_esm(model_name, model, chain).mean(dim=0, keepdim=True) for chain in chains if _check_chain_valid(chain, 4)], dim=0)
            data[cid] = feats.cpu()
        elif chain_process == 'cat_all':
            # encode chains in one 'protein', and apply avg to the seq
            feats = encode_protein_via_esm(model_name, model, CAT_TOKEN.join(chains)).squeeze(0)
            data[cid] = feats.mean(dim=0).cpu()
        elif chain_process == 'cat_all_truncate':
            # encode chains in one 'protein', but turncate if too long
            feats = encode_protein_via_esm(model_name, model, CAT_TOKEN.join(chains)[:max_len]).squeeze(0)
            data[cid] = feats.cpu() # [L, D]
        else:
            raise ValueError(f'Unknown chain_process: {chain_process}')
        
        # save data every 1000 proteins to avoid crash and get nothing!
        if idx % 1000 == 0:
            torch.save(data, result_path, pickle_protocol=5)
    
    torch.save(data, result_path, pickle_protocol=5)
    torch.save(lens, 'lens.pt')
    
    
def smiles_tokenizer(smiles: str) -> List[str]:
    """使用正则表达式对SMILES字符串进行分词"""
    # 定义SMILES中的特殊符号和双字母元素
    pattern = '|'.join([
        'Cl', 'Br', r'\[nH\]', r'\[NH\+\]', r'\[NH2\+\]', r'\[NH3\+\]', '[^a-zA-Z]',  # 双字母元素和非字母字符
        '[a-zA-Z]'  # 单个字母
    ])
    import re
    tokens = re.findall(pattern, smiles)
    return tokens


def extract_smiles_from_cids(cids: List[str], data_root: str) -> Dict[str, str]:
    """从给定的cid列表中提取对应的SMILES字符串"""
    cid2smiles = {}
    cid2dir = get_cid2dir()
    gz_paths = get_paths_with_extension(data_root, ['pt.gz'], name_substr='_6A')
    cid2gzpath = {Path(path).stem[:4]: path for path in gz_paths}
    
    for cid in tqdm(cids, desc="Extracting SMILES"):
        if cid in cid2dir:
            # 尝试从不同的文件中获取SMILES
            # 1. 先尝试从_pt.gz文件中读取
            if cid in cid2gzpath:
                try:
                    p = subprocess.Popen(['pigz', '-dc', cid2gzpath[cid]], stdout=subprocess.PIPE)
                    gz_data = p.stdout.read()
                    p.wait()
                    all_data = torch.load(io.BytesIO(gz_data))
                    if 'smiles' in all_data:
                        cid2smiles[cid] = all_data['smiles']
                except Exception:
                    continue
            
            # 2. 如果_pt.gz文件中没有，尝试从rdkit文件中读取
            if cid not in cid2smiles:
                rdkit_path = os.path.join(cid2dir[cid], f'{cid}.rdkit')
                if os.path.exists(rdkit_path):
                    try:
                        ligand, _ = opts_file(rdkit_path, 'rb', way='pkl')
                        cid2smiles[cid] = Chem.MolToSmiles(ligand)
                    except Exception:
                        continue
    
    return cid2smiles


def build_smiles_vocab(smiles_list: List[str], tokenizer, special_tokens: List[str] = None) -> Dict[str, int]:
    """根据SMILES列表构建词汇表"""
    if special_tokens is None:
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        
    # 收集所有token
    all_tokens = set()
    for smiles in tqdm(smiles_list, desc="Building vocabulary"):
        tokens = tokenizer(smiles)
        all_tokens.update(tokens)
    
    # 创建词汇表，包含特殊标记
    vocab = {token: i for i, token in enumerate(special_tokens)}
    
    # 添加从训练数据中提取的标记
    for token in all_tokens:
        vocab[token] = len(vocab)
    
    return vocab


def convert_smiles_to_indices(smiles: str, vocab: Dict[str, int], tokenizer, max_length: int = 256) -> List[int]:
    """将SMILES字符串转换为索引序列"""
    tokens = tokenizer(smiles)
    
    # # 添加开始标记
    # indices = [vocab['<SOS>']]
    indices = []
    
    # 转换每个token为索引，如果token不在词汇表中，使用<UNK>标记
    for token in tokens:
        indices.append(vocab.get(token, vocab['<UNK>']))
    
    # # 添加结束标记
    # indices.append(vocab['<EOS>'])
    
    # # 如果长度超过最大长度，截断
    # if len(indices) > max_length:
    #     indices = indices[:max_length]
    
    # # 如果长度不足最大长度，用PAD填充
    # elif len(indices) < max_length:
    #     indices.extend([vocab['<PAD>']] * (max_length - len(indices)))
    
    return indices


def process_smiles_tokenization(data_root: str = '../EHIGN_dataset', 
                                result_path: str = '../EHIGN_dataset/smiles_tokenized', 
                                max_length: int = 256):
    """主函数：从train和valid表格中提取SMILES，分词，构建词典，并转换为索引"""
    # 创建结果目录
    os.makedirs(result_path, exist_ok=True)
    
    # 从train和valid表格中读取cid
    read_cids = lambda df_file_name: pd.read_csv(f'data/{df_file_name}')['pdbid'].tolist()
    train_cids = read_cids('train.csv')
    valid_cids = read_cids('valid.csv')
    
    # 合并cid列表
    all_cids = list(set(train_cids + valid_cids))
    # all_cids = read_cids('test2013.csv')
    print(f"Total unique cids from train and valid: {len(all_cids)}")
    
    # 提取SMILES
    cid2smiles = extract_smiles_from_cids(all_cids, data_root)
    print(f"Successfully extracted {len(cid2smiles)} SMILES(per cid) from train and valid")
    
    # 构建词汇表
    smiles_list = list(cid2smiles.values())
    vocab = build_smiles_vocab(smiles_list, smiles_tokenizer)
    print(f"Vocabulary size: {len(vocab)}")
    
    # 保存词汇表
    vocab_path = os.path.join(result_path, 'smiles_vocab.pkl')
    opts_file(vocab_path, 'wb', way='pkl', data=vocab)
    
    # 将所有SMILES转换为索引
    test2013_cids = read_cids('test2013.csv')
    test2016_cids = read_cids('test2016.csv')
    test2019_cids = read_cids('test2019.csv')
    cid2smiles.update(extract_smiles_from_cids(test2013_cids+test2016_cids+test2019_cids, data_root))
    print(f"Total unique cids from train, valid, test2013, test2016, test2019: {len(cid2smiles)}")
    
    cid2indices = {}
    for cid, smiles in tqdm(cid2smiles.items(), desc="Converting SMILES to indices"):
        try:
            indices = convert_smiles_to_indices(smiles, vocab, smiles_tokenizer, max_length)
            cid2indices[cid] = indices
        except Exception as e:
            print(f"Error processing {cid}: {e}")
            continue
    
    # 保存索引数据
    indices_path = os.path.join(result_path, 'smiles_indices.pt')
    torch.save(cid2indices, indices_path)
    
    print(f"SMILES tokenization completed. Results saved to {result_path}")


def generate_method_id():
    """Generate method id (Ki, Kd, IC50) for each cid."""
    all_data = opts_file('data/clean_split/PDBbind_data_dict.json', way='json')
    text_source = opts_file('../EHIGN_dataset/PDBbind_2019/refined-set/index/INDEX_general_PL.2019', errors='ignore').split('\n')[6:]
    real_data = torch.load('../EHIGN_dataset/SMILES_PepDoRA.pt')
    text_source = {line[:4]: line for line in text_source}
    method_id_map = {'Ki': 0, 'Kd': 1, 'IC50': 2}
    cid2mid = {}
    for dataset_name in ['train', 'valid', 'test2013', 'test2016', 'test2019', 'clean_split/affinity']:
        df = pd.read_csv(f'data/{dataset_name}.csv')
        for cid in tqdm(df['pdbid'], desc=f'Processing {dataset_name}', leave=False):
            record = all_data.get(cid, text_source.get(cid, None))
            if record is None:
                if cid not in real_data:
                    continue
                raise ValueError(f"CID {cid} not found in dataset {dataset_name}")
            for method in method_id_map:
                if (isinstance(record, str) and f'{method}=' in record) or (isinstance(record, dict) and method in record):
                    cid2mid[cid] = method_id_map[method]
            if cid not in cid2mid:
                raise ValueError(f"CID {cid} not found method id in dataset {dataset_name}")
        print(f'Finish processing {dataset_name}, {len(df)} cids in df, {len(set(cid2mid.keys()) & set(df["pdbid"].tolist()))} cids have processed')
    opts_file('data/cid2mid.pkl', 'wb', way='pkl', data=cid2mid)
    


def run_tmalign(pair, tm_re, rmsd_re):
    pdb1, pdb2 = pair
    cmd = [TMALIGN_PATH, pdb1, pdb2]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)  # 5 min timeout
    except subprocess.TimeoutExpired:
        return (pdb1, pdb2, None, None, "timeout")
    out = p.stdout + p.stderr
    # 解析第一个 TM-score 和 RMSD
    m = tm_re.search(out)
    r = rmsd_re.search(out)
    tm = float(m.group(1)) if m else None
    rmsd = float(r.group(1)) if r else None
    return (pdb1, pdb2, tm, rmsd, None)


def get_tm_score():
    cid2dir = get_cid2dir()
    cids = list(cid2dir.keys())
    family_df = pd.read_excel('../EHIGN_dataset/protein_family/pfam_domain_hits1.xlsx')
    cid2family = dict(zip(family_df['seq_id'], family_df['pfam_name']))
    family2cids = defaultdict(list)
    for cid, family in cid2family.items():
        family2cids[family].append(cid)
    family2cids['UNK'] = list(set(cids) - set(cid2family.keys()))
    for cid in family2cids['UNK']:
        cid2family[cid] = 'UNK'
    if os.path.exists(f'../EHIGN_dataset/protein_family/tm_scores.pt'):
        tm_results = torch.load(f'../EHIGN_dataset/protein_family/tm_scores.pt')
        rmsd_results = torch.load(f'../EHIGN_dataset/protein_family/rmsd_scores.pt')
    else:
        tm_results = {}
        rmsd_results = {}
    tm_re = re.compile(r"TM-score=\s*([0-9.]+)")
    rmsd_re = re.compile(r"RMSD=\s*([0-9.]+)")
    taskpool = TaskPool('threads', n_worker=4, report_error=True).start()
    for cid_i in tqdm(cids, desc='Processing TM-align'):
        tm_results[cid_i] = tm_results.get(cid_i, {})
        rmsd_results[cid_i] = rmsd_results.get(cid_i, {})
        family = cid2family[cid_i]
        tasks = []
        for cid_j in tqdm(family2cids[family], desc=f'Processing TM-align {cid_i}: family {family}', leave=False):
            tm_results[cid_j] = tm_results.get(cid_j, {})
            rmsd_results[cid_j] = rmsd_results.get(cid_j, {})
            if cid_i == cid_j:
                continue
            if cid_i in tm_results and cid_j in tm_results[cid_i] and tm_results[cid_i][cid_j]:
                continue
            pdb1 = os.path.join(cid2dir[cid_i], f'{cid_i}_protein.pdb')
            pdb2 = os.path.join(cid2dir[cid_j], f'{cid_j}_protein.pdb')
            tasks.append(taskpool.add_task((cid_i, cid_j), run_tmalign, (pdb1, pdb2), tm_re, rmsd_re))
            taskpool.wait_till(lambda: taskpool.count_waiting_tasks() == 0, 0.001)
        for task in tqdm(tasks, desc=f'Query TM-align {cid_i}: family {family}', leave=False):
            cid_i, cid_j = task
            pdb1, pdb2, tm, rmsd, err = taskpool.query_task(task, block=True, timeout=999)
            if err is None:
                tm_results[cid_i][cid_j] = tm_results[cid_j][cid_i] = tm
                rmsd_results[cid_i][cid_j] = rmsd_results[cid_j][cid_i] = rmsd
            else:
                print(f"Error aligning {pdb1} and {pdb2}: {err}")
        # save result per 1000 cids
        if cids.index(cid_i) % 1000 == 0:
            torch.save(tm_results, '../EHIGN_dataset/protein_family/tm_scores.pt')
            torch.save(rmsd_results, '../EHIGN_dataset/protein_family/rmsd_scores.pt')
    torch.save(tm_results, '../EHIGN_dataset/protein_family/tm_scores.pt')
    torch.save(rmsd_results, '../EHIGN_dataset/protein_family/rmsd_scores.pt')
    
    

if __name__ == '__main__':
    # dev code
    # load_all_models_and_calculate_size()
    # generate_method_id()
    # get_tm_score()
    # process_smiles_tokenization(f'../EHIGN_dataset/MISATO')
    # for name in ['MolAI', 'ChemBERTa_77M_MTR',
    #              'MoLFormer', 'SELFormer', 'PepDoRA-token']:
    # name = 'PepDoRA'
    # generate_SMILES_dataset(data_root=f'../EHIGN_dataset/MISATO', result_path=Path(f'../EHIGN_dataset/SMILES_{name}_fix.pt'),
    #                             model_name=name)
    # generate_SMILES_dataset(data_root=f'../EHIGN_dataset/MISATO', result_path=Path(f'../EHIGN_dataset/SMILES_GeminiMol.pt'),
    #                             model_name='GeminiMol')
    generate_protein_dataset(data_root=os.path.expanduser('path-to-your-data-folder/MISATO'),
                             result_path=Path(os.path.expanduser('path-to-your-data-folder/protein_esm2-3B-33_cat_all_truncate.pt')),
                             model_name='esm2-3B-33', device='cuda', chain_process='cat_all_truncate', max_len=1024)
    exit(0)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='生成蛋白质或SMILES特征数据'
    )
    subparsers = parser.add_subparsers(dest='command', help='可用子命令')
    
    # 蛋白质数据生成子命令
    protein_parser = subparsers.add_parser('protein', help='生成蛋白质特征数据')
    protein_parser.add_argument('--data-root', required=False, default=os.path.expanduser('path-to-your-data-folder/MISATO'), help='包含蛋白质数据的根目录')
    protein_parser.add_argument('--result-path', required=False, default=os.path.expanduser('path-to-your-data-folder/MISATO/protein_esm3-open_cat_each_mean.pt'), help='结果文件保存路径')
    protein_parser.add_argument('--model-name', default='esm3-open', 
                               choices=['esm3-open', 'esm2', 'esm2-3B', 'esm2-3B-33', 'SaProt_650M_AF2', 'ProSST-2048'], help='使用的蛋白质模型名称')
    protein_parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='计算设备')
    protein_parser.add_argument('--chain-process', default='mean_each_mean', 
                               choices=['mean_each_mean', 'cat_each_mean', 'cat_all', 'cat_all_truncate'],
                               help='蛋白质链处理方式')
    protein_parser.add_argument('--max-len', type=int, default=1024, help='最大序列长度（仅在cat_all_truncate模式下使用）')
    
    # SMILES数据生成子命令
    smiles_parser = subparsers.add_parser('smiles', help='生成SMILES特征数据')
    smiles_parser.add_argument('--data-root', required=False, default='../EHIGN_dataset/MISATO', help='包含SMILES数据的根目录')
    smiles_parser.add_argument('--result-path', required=False, default='../EHIGN_dataset/SMILES_PepDoRA.pt', help='结果文件保存路径')
    smiles_parser.add_argument('--model-name', default='PepDoRA', 
                              choices=['ChemBERTa_10M', 'ChemBERTa_100M_MLM', 'ChemBERTa_77M_MLM', 'ChemBERTa_77M_MTR',
                                       'MoLFormer', 'PepDoRA', 'PepDoRA-token', 'SELFormer', 'UniMol-SMILES', 'UniMol-Mol',
                                       'MolAI', '3Dimg', 'MaskMol', 'ImageMol', 'rdkit_vit', 'GenminiMol'],
                              help='使用的SMILES模型名称')
    
    # SMILES tokenization
    token_parser = subparsers.add_parser('smiles_tokenize', help='SMILES分词和索引转换')
    token_parser.add_argument('--data-root', required=False, default='../EHIGN_dataset', help='包含SMILES数据的根目录')
    token_parser.add_argument('--result-path', required=False, default='../EHIGN_dataset/smiles_tokenized', help='结果保存路径')
    token_parser.add_argument('--max-length', type=int, default=256, help='SMILES索引序列的最大长度')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据子命令执行相应功能
    if args.command == 'protein':
        generate_protein_dataset(
            data_root=args.data_root,
            result_path=Path(args.result_path),
            model_name=args.model_name,
            device=args.device,
            chain_process=args.chain_process,
            max_len=args.max_len
        )
    elif args.command == 'smiles':
        generate_SMILES_dataset(
            data_root=args.data_root,
            result_path=Path(args.result_path),
            model_name=args.model_name
        )
    elif args.command == 'smiles_tokenize':
        process_smiles_tokenization(args.data_root, args.result_path, args.max_length)
    else:
        raise ValueError(f'Unknown command: {args.command}')


# import torch
# import matplotlib.pyplot as plt
# a = torch.load('../EHIGN_dataset/ChemBERTa-100M_len.pt')
# plt.figure(figsize=(10, 6))
# plt.hist(list(a.values()), bins=256, density=True, alpha=0.6, color='b')
# plt.savefig('ChemBERTa_100M_len.png', dpi=600)