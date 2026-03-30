'''
self info removed
'''
import gc
import logging
import os

import dgl
import natsort
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mbapy.base import put_log
from mbapy.file import opts_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.s1.data_loader import GraphDataset as S1DataLoader


def load_lig_data_by_name(data_names: list[str], logger: logging.Logger = None):
    """
    support data names: PepDoRA, ChemBERTa, MolAI, UniMol-v3, UniMol-v4
    """
    assert set(data_names).issubset({'PepDoRA', 'ChemBERTa_10M', 'ChemBERTa_100M_MLM',
                                     'ChemBERTa_77M_MLM', 'ChemBERTa_77M_MTR', 'MolFormer',
                                     'MolAI', 'GeminiMol', 'SELFormer', 'UniMol-v3', 'UniMol-v3-570m', 'UniMol-v4', 'PepDoRA-token', 'token',
                                     'MaskMol_224', 'rdkit_vit_224', 'rdkit_vit_512', 'ImageMol', '3Dimg', '3Dimg_vit'}), \
        f'lig_data_names {data_names} not supported'
    datas = {}
    for name in data_names:
        if name in {'PepDoRA', 'ChemBERTa_10M', 'ChemBERTa_100M_MLM',
                    'ChemBERTa_77M_MLM', 'ChemBERTa_77M_MTR', 'MolFormer', 'MolAI', 'GeminiMol', 'SELFormer',
                    'MaskMol_224', 'rdkit_vit_224', 'rdkit_vit_512', 'ImageMol', '3Dimg', '3Dimg_vit'}:
            datas[name] = torch.load(os.path.expanduser(f'path-to-your-data-folder/SMILES_{name}.pt'), map_location='cpu', weights_only=False)
        elif name in {'UniMol-v3', 'UniMol-v3-570m', 'UniMol-v4'}:
            datas[name] = os.path.expanduser(f'path-to-your-data-folder/{name}')
        elif name == 'token':
            datas[name] = torch.load('../EHIGN_dataset/smiles_tokenized/smiles_indices.pt', map_location='cpu', weights_only=False)
        elif name == 'PepDoRA-token':
            datas[name] = torch.load('../EHIGN_dataset/SMILES_PepDoRA-token.pt', map_location='cpu', weights_only=False)
    return datas


def load_rec_data_by_name(data_names: list[str], logger: logging.Logger = None):
    """
    support data names: ems3, esm2, UniMol-v3, UniMol-v4
    """
    datas = {}
    for name in data_names:
        if name == 'esm2-mean_each_mean':
            datas[name] = torch.load(os.path.expanduser(f'path-to-your-data-folder/protein_esm2-mean_each_mean.pt'), weights_only=False)
        elif name == 'esm2-3B-36_mean_each_mean':
            datas[name] = torch.load(os.path.expanduser(f'path-to-your-data-folder/protein_esm2-3B-36_mean_each_mean.pt'), weights_only=False)
        elif name == 'esm2-3B-30_mean_each_mean':
            datas[name] = torch.load(os.path.expanduser(f'path-to-your-data-folder/protein_esm2-3B-30_mean_each_mean.pt'), weights_only=False)
        elif name == 'esm3-open_split':
            datas[name] = torch.load(os.path.expanduser(f'path-to-your-data-folder/protein_esm3-open_split.pt'), weights_only=False)
        elif name == 'SaProt':
            datas[name] = torch.load(os.path.expanduser(f'path-to-your-data-folder/protein_SaProt_650M_AF2_mean_each_mean.pt'), weights_only=False)
        elif name == 'esm2-3B':
            datas[name] = torch.load(os.path.expanduser(f'path-to-your-data-folder/protein_esm2-3B_mean_each_mean.pt'), weights_only=False)
        elif name == 'ProSST-2048':
            datas[name] = torch.load(os.path.expanduser(f'path-to-your-data-folder/protein_ProSST-2048_mean_each_mean.pt'), weights_only=False)
        elif name in {'UniMol-v3', 'UniMol-v3-570m', 'UniMol-v4'}:
            datas[name] = os.path.expanduser(f'path-to-your-data-folder/{name}')
        else:
            raise ValueError(f'rec_data_name {name} not supported')
    return datas


class GraphDataset(S1DataLoader):
    def __init__(self, lig_data: dict[str, dict[str, torch.Tensor]], rec_data: dict[str, dict[str, torch.Tensor]],
                 df: pd.DataFrame, prot_transform: list[str]=None, prot_max_len: int = 1024, lig_seq_reduce: str=None,
                 load_ratio: float = 0.1, load_order: str = 'random', device: str='cpu', logger: logging.Logger = None):
        """
        Parameters
        ----------
            - lig_data: dict[str, dict[str, torch.Tensor]], data dict
                - key: feature data name, such as PepDoRA, ChemBERTa, MolAI, UniMol-Mol
                - value: feature data tensor dict
                    - key2: cid
                        - key3: feature name, such as `hidden_states`
                        - value3: feature tensor
            - rec_data: dict[str, dict[str, torch.Tensor]], data dict
                - key: feature data name, such as ems3, esm2
                - value: feature data tensor dict
                    - key2: cid
                        - key3: feature name, such as `hidden_states`
                        - value3: feature tensor
            - df: pd.DataFrame, dataframe with cid and pKa columns
            - prot_transform: list[str], transform prot feature after loaded, default None
                - prot_trans_start: float, start point of prot data value
                - prot_trans_method: str, transform method of prot data
                    - int or float value: scale prot feature by this value
                    - 'log2', 'log10', 'sqrt', 'sqrt0.667': transform prot feature by this method
            - prot_max_len: int, default 1024, max length of prot data
            - lig_seq_reduce: str, default None, reduce lig data along seq dim
                - 'sum': sum lig data along seq dim
                - 'mean': mean lig data along seq dim
            - load_ratio: float, default 0.1, load ratio of data. ONLY works with UniMol-v4.
            - load_order: str, default 'random', load order of data. ONLY works with UniMol-v4.
                - 'random': random load data
                - 'front': load data from front
                - 'back': load data from back
            - lig_type: str, default 'PepDoRA'
                - 'PepDoRA': 
                - 'ChemBERTa': 
            - device: str, default 'cpu'
                - 'cpu': keep graph on cpu memory
            - logger: logging.Logger, default None
        """
        self.data = []
        self.df = df.astype({'pdbid': str, '-logKd/Ki': float})
        self.cid2mid = opts_file(f'data/cid2mid.pkl', 'rb', way='pkl')
        self.cids = None
        self.prot_transform = prot_transform
        self.prot_max_len = prot_max_len      
        self.lig_seq_reduce = lig_seq_reduce
        self.device = device
        self.dyn_padding = False
        # store mean and std
        self.mean = None
        self.std = None
        
        # check args       
        assert self.device in ['cpu'], f'load2mem must be one of cpu, cuda, disk, but got {self.device}'
        log_fn = logger.info if logger else put_log
        if self.prot_transform is not None:
            prot_trans_start, prot_trans_method = self._check_prot_transform()
            
        # load umol data if path is given
        lig_umol = list(filter(lambda x: x.startswith('UniMol'), list(lig_data.keys())))
        rec_umol = list(filter(lambda x: x.startswith('UniMol'), list(rec_data.keys())))
        assert len(lig_umol) <= 1, f'only support one version of UniMol in one time for ligand, but got {lig_umol}'
        assert len(rec_umol) <= 1, f'only support one version of UniMol in one time for pocket, but got {rec_umol}'
        if lig_umol and rec_umol:
            _lig_data, _rec_data = {k: v for k, v in lig_data.items() if k != lig_umol[0]}, \
                                   {k: v for k, v in rec_data.items() if k != rec_umol[0]}
            _lig_data[lig_umol[0]], _rec_data[rec_umol[0]] = self.load_umol_data(lig_data[lig_umol[0]],
                                                                               rec_data[rec_umol[0]], df,
                                                                               load_ratio=load_ratio, load_order=load_order)
        elif lig_umol and not rec_umol:
            _lig_data, _rec_data = {k: v for k, v in lig_data.items() if k != lig_umol[0]}, rec_data
            _lig_data[lig_umol[0]] = self.load_umol_data(lig_data[lig_umol[0]], None, df,
                                                        load_ratio=load_ratio, load_order=load_order)[0]
        elif not lig_umol and rec_umol:
            _lig_data, _rec_data = lig_data, {k: v for k, v in rec_data.items() if k != rec_umol[0]}
            _rec_data[rec_umol[0]] = self.load_umol_data(None, rec_data[rec_umol[0]], df,
                                                        load_ratio=load_ratio, load_order=load_order)[1]
        else:
            _lig_data, _rec_data = lig_data, rec_data
        
        exists_keys = self._extract_cids(list(_lig_data.values()) + list(_rec_data.values()))
        self.valid_key = [(cid, pKa) for cid, pKa in self.df[['pdbid', '-logKd/Ki']].values if cid in exists_keys]
        log_fn(f'{len(self.df)} cids in dataframe, {len(self.valid_key)} cid found in file.')
        if len(self.valid_key) == 0:
            raise ValueError(f'No valid data found')
        
        # setup prot data transform
        if self.prot_transform is not None:
            prot_transform_fn = self._get_prot_tranform_fn(prot_trans_method)
                
        # load data
        self.umol_v4_pool = {}
        for idx, (cid, pKa) in tqdm(enumerate(self.valid_key), desc='Pre-Process data', leave=False):
            item_data = {'idx': torch.LongTensor([idx], device=device),
                         'mid': torch.LongTensor([self.cid2mid[cid]], device=device),
                         'pKa': torch.FloatTensor([pKa], device=device)}
            # add rec data first
            for name, data in _rec_data.items():
                name = f'{name}-p' if 'UniMol' in name else name # lig and prot both can use UniMol-v3, so just add a suffix
                rec_feat, rec_mask = self._extract_feature_mask(name, data, cid)       
                # transform prot data
                if name == 'esm3' and self.prot_transform is not None:
                    rec_feat = torch.sign(rec_feat) * torch.where(rec_feat.abs() <= prot_trans_start,
                                                                rec_feat.abs(),
                                                                prot_transform_fn(rec_feat.abs(), prot_trans_start, prot_trans_method))
                item_data[name] = [rec_feat, rec_mask]
            # add lig data after rec data
            for name, data in _lig_data.items():
                name = f'{name}-l' if 'UniMol' in name else name # lig and prot both can use UniMol-v3, so just add a suffix
                item_data[name] = self._extract_feature_mask(name, data, cid)
            # check whether has UniMol-v4 data to be extend
            if 'UniMol-v4-l' in item_data:
                self.umol_v4_pool[cid] = [item_data.pop('UniMol-v4-l'), item_data]
                for i in range(len(self.umol_v4_pool[cid][0])):
                    self.data.append((cid, i))
            else:
                self.data.append(item_data)
            
        gc.collect()
        log_fn(f'{len(self.data)} data loaded into RAM with(load_ratio={load_ratio}, load_order={load_order}).')
        
    @staticmethod
    def collate_fn(batch):
        """
        Parameters
        ----------
            - batch: list[dict[str, torch.Tensor]], batch data
                - key: feature data name, such as PepDoRA, ChemBERTa, MolAI, UniMol-Mol, esm2, esm3
                - value: feature tensor or mask tensor
        Returns
        -------
            - dict[str, torch.Tensor]: collated batch data
        """
        out = {}
        for k in batch[0].keys():
            if k in {'pKa', 'idx', 'mid'}:
                out[k] = torch.stack([b[k] for b in batch], dim=0)
            else:
                out[k] = [torch.stack([b[k][0] for b in batch], dim=0),
                          torch.stack([b[k][1] for b in batch], dim=0)]
        return out
        
    def _pad_prot_feat(self, rec_feat: torch.tensor, prot_n_chains: int):
        if rec_feat.size(0) <= prot_n_chains:
            rec_mask = (torch.arange(prot_n_chains, device=rec_feat.device) < rec_feat.size(0)).long()
            rec_feat = F.pad(rec_feat, (0, 0, 0, prot_n_chains - rec_feat.size(0)), 'constant', 0)
        elif rec_feat.size(0) > prot_n_chains:
            rec_feat = rec_feat[:prot_n_chains] + rec_feat[prot_n_chains:].mean(dim=0)
            rec_mask = torch.ones(prot_n_chains, dtype=torch.long, device=rec_feat.device)
        return rec_feat, rec_mask
        
    def _extract_feature_mask(self, name: str, data: dict[str, torch.Tensor], cid: str) -> list[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
            - name: feature data name, such as PepDoRA, ChemBERTa, MolAI, UniMol-Mol, esm2, esm3
            - data: feature data tensor dict
                - key1: cid
                    - key2: feature name, such as `hidden_states`
                    - value2: feature tensor
            - cid: str, cid of data
        Returns
        -------
            - tuple[torch.Tensor, torch.Tensor]: feature tensor and mask tensor
        """
        if name in {'PepDoRA', 'ChemBERTa_10M', 'ChemBERTa_100M_MLM', 'ChemBERTa_77M_MLM',
                    'ChemBERTa_77M_MTR', 'MolFormer', 'SELFormer'}:
            feature = data[cid]['hidden_states'].to(device=self.device).squeeze(0) # [L, D]
            mask = data[cid]['attention_mask'].to(device=self.device).squeeze(0) # [L]
        elif name == 'GeminiMol':
            feature = data[cid].to(device=self.device) # [1, 2048]
            mask = torch.ones((1,), dtype=torch.long) # [1]
        elif name == 'MolAI':
            feature = torch.from_numpy(data[cid]).to(device=self.device) # [1, D]
            mask = torch.ones((1,), dtype=torch.long) # [1]
        elif name in {'token'}:
            if len(data[cid]) < 256:
                feature = F.pad(torch.tensor(data[cid]), (0, 256 - len(data[cid])), value=0)
                mask = (torch.arange(256) <= len(data[cid]))
            else:
                feature = torch.tensor(data[cid])[:256]
                mask = torch.ones_like(feature, dtype=torch.bool)
        elif name in {'PepDoRA-token'}:
            feature = torch.tensor(data[cid]).squeeze()
            mask = (feature != 0).bool()
        elif name in {'3Dimg', '3Dimg_vit'}:
            feature = torch.cat([torch.stack(i) for i in data[cid].values()], dim=0)
            mask = torch.ones(feature.shape[0], dtype=torch.bool)
        elif name in {'MaskMol_224', 'rdkit_vit_224', 'rdkit_vit_512', 'ImageMol'}:
            feature = data[cid]
            mask = torch.ones(feature.shape[0], dtype=torch.bool)
        elif name.startswith('UniMol-v3'):
            # force padding to avoid hybrid data with both v4 and v3 to be hard to deal with
            feature, mask = self.merge_pad_unimol_data(data[cid], force_pad=True)
        elif name.startswith('UniMol-v4'):
            # just return a list of feature tensor and mask tensor
            data_lst = [self.merge_pad_unimol_data(data_i) for data_i in data[cid]]
            if self.lig_seq_reduce is not None:
                data_lst = list(map(lambda x: self._apply_lig_data_reduce(*x), data_lst))
            return data_lst  # pyright: ignore[reportReturnType]
        elif name in {'esm2-mean_each_mean', 'esm2-3B', 'esm2-3B-36_mean_each_mean',
                      'esm2-3B-30_mean_each_mean', 'esm3-open_split', 'SaProt', 'ProSST-2048'}:
            feature = data[cid].to(device=self.device)
            if len(feature.shape) == 1:
                feature = feature.unsqueeze(0) # [1, D]
                mask = torch.ones((1,), dtype=torch.long) # [1]
            else:
                feature, mask = self._pad_prot_feat(feature, self.prot_max_len) # [L, D], [L]
        else:
            raise ValueError(f'Unknown feature data name: {name}')
        if self.lig_seq_reduce is not None:
            feature, mask = self._apply_lig_data_reduce(feature, mask)
        return [feature, mask]
        
    def __getitem__(self, idx):
        if self.umol_v4_pool:
            cid, i = self.data[idx]
            umol_v4, other_data = self.umol_v4_pool[cid]
            umol_v4_feat, umol_v4_mask = umol_v4[i]
            if self.dyn_padding and self.lig_seq_reduce is None:
                padded_umol_v4_feat = F.pad(umol_v4_feat, (0, 0, 0, 257 - umol_v4_feat.size(0)), value=0)
                return {'UniMol-v4-l': [padded_umol_v4_feat, umol_v4_mask], **other_data}
            return {'UniMol-v4-l': [umol_v4_feat, umol_v4_mask], **other_data}
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
    def __del__(self):
        if self.umol_v4_pool:
            for cid in list(self.umol_v4_pool.keys()):
                del self.umol_v4_pool[cid]
        else:
            for i in self.data:
                for k in list(i.keys()):
                    del i[k]


def get_data_loader(lig_data: dict[str, dict[str, torch.Tensor]], rec_data: dict[str, dict[str, torch.Tensor]],
                    df: pd.DataFrame, prot_transform: list[str], prot_max_len: int, lig_seq_reduce: str, load_ratio: float,
                    load_order: str, device: str, batch_size: int,
                    shuffle: bool, num_workers: int=0, logger: logging.Logger = None):
    dataset = GraphDataset(lig_data, rec_data, df, prot_transform, prot_max_len, lig_seq_reduce, load_ratio, load_order, device, logger)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                num_workers=num_workers, collate_fn=GraphDataset.collate_fn)
    return dataloader
    
    
if __name__ == '__main__':
    name = 'train'
    df = pd.read_csv(f'./data/{name}.csv')
    lig_data = load_lig_data_by_name(['MaskMol_224'])
    prot_data = load_rec_data_by_name([])
    # dataloader = get_data_loader(lig_data, prot_data, df, None, 16, 0.1, 'random', 'cpu', 128, True, 0, None)
    valid_dataloader = get_data_loader(lig_data, prot_data, pd.read_csv(f'./data/valid.csv'), None, 16, 'mean', 0.1, 'random', 'cpu', 128, True, 0, None)
    # dataloader.dataset.calcu_mean_std(data_idx=4)
    # dataloader.dataset.apply_mean_std(data_idx=4)
    # for data_batch in tqdm(dataloader):
    #     pass
    # del dataloader.dataset
    for data_batch in tqdm(valid_dataloader):
        [print(k, v[0].shape, v[1].shape) for k, v in data_batch.items()]
        break