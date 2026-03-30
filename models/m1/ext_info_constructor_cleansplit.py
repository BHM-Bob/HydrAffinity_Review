import os
from pathlib import Path

from mbapy.file import opts_file

from models.m1.ext_info_constructor import (generate_SMILES_dataset, generate_protein_dataset,
                                            get_cid2dir)


def get_PDBBind2020_cid2dir():
    cid2dir: dict[str, str] = get_cid2dir()
    all_data = opts_file('./data/clean_split/PDBbind_data_dict.json', way='json')
    ref_cids = os.listdir('../EHIGN_dataset/PDBBind_2020/PDBbind_v2020_refined/refined-set')
    pl_cids = os.listdir('../EHIGN_dataset/PDBBind_2020/v2020-other-PL')
    if set(all_data.keys()) - set(ref_cids + pl_cids):
        print(set(all_data.keys()) - set(ref_cids + pl_cids))
        raise ValueError('There are some cids in PDBbind_data_dict.json that are not in refined-set or v2020-other-PL')
    for cid in ref_cids:
        cid2dir[cid] = f'../EHIGN_dataset/PDBBind_2020/PDBbind_v2020_refined/refined-set/{cid}'
    for cid in pl_cids:
        cid2dir[cid] = f'../EHIGN_dataset/PDBBind_2020/v2020-other-PL/{cid}'
    cid2dir.pop('index')
    cid2dir.pop('readme')
    return cid2dir


def generate_protein_dataset_cleansplit(data_root: str, result_path: Path,
                                        model_name: str = 'esm3-open', device: str = 'cuda',
                                        chain_process: str = 'mean_each_mean', max_len: int = 1024):
    cid2dir = get_PDBBind2020_cid2dir()
    generate_protein_dataset(data_root, result_path, model_name, device, chain_process, max_len, cid2dir)
    
    
def generate_SMILES_dataset_cleansplit(data_root: str,
                                        result_path: Path, model_name: str = 'PepDoRA', resolution: int = 224):
    cid2dir = get_PDBBind2020_cid2dir()
    generate_SMILES_dataset(data_root, result_path, model_name, resolution, cid2dir)

    
if __name__ == '__main__':
    # generate_protein_dataset_cleansplit(None, result_path=Path('../EHIGN_dataset/protein_esm3-open_split.pt'),
    #                                      model_name='esm3-open', device='cpu', chain_process='mean_each_mean', max_len=1024)
    # generate_protein_dataset_cleansplit(None, result_path=Path('../EHIGN_dataset/protein_SaProt_650M_AF2_mean_each_mean.pt'),
    #                                      model_name='SaProt_650M_AF2', device='cpu', chain_process='mean_each_mean', max_len=1024)
    # generate_protein_dataset_cleansplit(None, result_path=Path('../EHIGN_dataset/protein_ProSST-2048_mean_each_mean.pt'),
    #                                      model_name='ProSST-2048', device='cpu', chain_process='mean_each_mean', max_len=1024)
    
    for name in ['ChemBERTa_10M', 'ChemBERTa_100M_MLM', 'ChemBERTa_77M_MLM', 'ChemBERTa_77M_MTR', 
                'MoLFormer', 'SELFormer', 'MolAI', 'PepDoRA-token']:
        generate_SMILES_dataset_cleansplit(data_root=f'../EHIGN_dataset/MISATO', result_path=Path(f'../EHIGN_dataset/SMILES_{name}.pt'),
                                            model_name=name)