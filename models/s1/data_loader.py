'''
self info removed
'''
import gc
import logging
import os
import random
from typing import Union

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


class GraphDataset(object):
    def __init__(self, lig_data: Union[dict[str, torch.tensor], str], rec_data: Union[dict[str, torch.tensor], str],
                 df: pd.DataFrame, prot_transform: list[str]=None, lig_seq_reduce: str=None, load_ratio: float=1.0,
                 load_order: str='random', cat_v4_n: int = None, device: str='cpu', logger: logging.Logger = None):
        """
        Parameters
        ----------
            - lig_data: dict[str, torch.tensor] | str, data dict or path to UniMol-v3/4 dir
            - rec_data: dict[str, torch.tensor] | str, data dict or path to UniMol-v3/4 dir
            - df: pd.DataFrame, dataframe with cid and pKa columns
            - prot_transform: list[str], transform prot feature after loaded, default None
                - prot_trans_start: float, start point of prot data value
                - prot_trans_method: str, transform method of prot data
                    - int or float value: scale prot feature by this value
                    - 'log2', 'log10', 'sqrt', 'sqrt0.667': transform prot feature by this method
            - lig_seq_reduce: str, default None
                - reduce lig data along seq dim, default None
                    - 'sum': sum lig data along seq dim
                    - 'mean': mean lig data along seq dim
                    - 'zero': set padding token to zeros in lig seq
            - load_ratio: float, default 1.0
                - ratio of data to load, default 1.0
            - load_order: str, default 'random'
                - order of data to load, default 'random'
                    - 'random': load data randomly
                    - 'front': load data from front
                    - 'back': load data from back
                    - 'more': NOT SUPPORTED NOW
            - cat_v4_n: int, default None
                - concat multiple v4 data into one data at dim 0, work with lig_seq_reduce
            - device: str, default 'cpu'
                - 'cpu': keep graph on cpu memory
            - logger: logging.Logger, default None
        """
        self.data = []
        self.df = df.astype({'pdbid': str, '-logKd/Ki': float})
        self.cid2mid = opts_file(f'data/cid2mid.pkl', 'rb', way='pkl')
        self.cids = None
        self.prot_transform = prot_transform        
        self.lig_seq_reduce = lig_seq_reduce
        self.cat_v4_n = cat_v4_n
        self.device = device
        self.dyn_padding = False
        # store mean and std
        self.mean = None
        self.std = None
        
        # check args       
        assert self.device in ['cpu'], f'load2mem must be one of cpu, cuda, disk, but got {self.device}'
        assert isinstance(lig_data, (dict, str)) and isinstance(rec_data, (dict, str)), f'lig_data and rec_data must be dict or str, but got {type(lig_data)} and {type(rec_data)}'
        log_fn = logger.info if logger else put_log
        if self.prot_transform is not None and self.prot_transform:
            prot_trans_start, prot_trans_method = self._check_prot_transform()
        else:
            prot_trans_start, prot_trans_method = None, None
                
        # load umol data if path is given
        if isinstance(lig_data, str) or isinstance(rec_data, str):
            lig_data, rec_data = self.load_umol_data(lig_data, rec_data, df, load_ratio=load_ratio, load_order=load_order)
                
        # check and align cids, generate valid keys
        exists_keys = self._extract_cids([lig_data, rec_data])
        self.valid_key = [(cid, pKa) for cid, pKa in self.df[['pdbid', '-logKd/Ki']].values if cid in exists_keys]
        log_fn(f'{len(self.df)} cids in dataframe, {len(self.valid_key)} cid found in file.')
        if len(self.valid_key) == 0:
            raise ValueError(f'No valid data found')
        
        # setup prot data transform
        if self.prot_transform is not None and self.prot_transform:
            self.prot_transform_fn = self._get_prot_tranform_fn(prot_trans_method)

        # load data
        self.umol_v4_pool = {}
        for idx, (cid, pKa) in tqdm(enumerate(self.valid_key), desc='Pre-Process data', leave=False):
            idx = torch.LongTensor([idx])
            mid = torch.LongTensor([self.cid2mid[cid]])
            pKa = torch.FloatTensor([pKa]).to(device=device)
            rec_feat = self._process_rec_ori_data(cid, rec_data, device, prot_trans_start, prot_trans_method)
            # UniMol-v4 data
            if isinstance(lig_data[cid], list) and isinstance(lig_data[cid][0], dict):
                lig_data[cid] = [self._process_lig_ori_data(lig_data_i, device) for lig_data_i in lig_data[cid]]
                if self.cat_v4_n is None:
                    self.umol_v4_pool[cid] = [idx, mid, lig_data[cid], rec_feat, pKa]
                    for i in range(len(lig_data[cid])):
                        self.data.append([cid, i])
                else:
                    lig_feat = torch.cat([i[0] for i in lig_data[cid][:self.cat_v4_n]], dim=0)
                    lig_mask = torch.cat([i[1] for i in lig_data[cid][:self.cat_v4_n]], dim=0)
                    # in case there are less than self.cat_v4_n v4 data, [n_data, D], pad to [self.cat_v4_n, D]
                    if lig_feat.shape[0] < self.cat_v4_n:
                        lig_feat = F.pad(lig_feat, (0, 0, 0, self.cat_v4_n - lig_feat.shape[0]), value=0)
                        lig_mask = F.pad(lig_mask, (0, self.cat_v4_n - lig_mask.shape[0]), value=0)
                    self.data.append([idx, mid, lig_feat, lig_mask, rec_feat, pKa])
            else:
                lig_feat, lig_mask = self._process_lig_ori_data(lig_data[cid], device)
                self.data.append([idx, mid, lig_feat, lig_mask, rec_feat, pKa])
        gc.collect()
        log_fn(f'{len(self.data)} data loaded into RAM with(load_ratio={load_ratio}, load_order={load_order}).')
        
    def _process_lig_ori_data(self, lig_data: dict[str, torch.tensor], device: str):
        """process original data into the form that can be appended into self.data"""
        # UniMol data
        if isinstance(lig_data, dict) and 'cls' in lig_data:
            lig_feat, lig_mask = self.merge_pad_unimol_data(lig_data)
        # 3Dimg data
        elif isinstance(lig_data, dict) and 'x_rotation_stick' in lig_data:
            lig_feat = torch.cat([torch.stack(i) for i in lig_data.values()], dim=0).to(device=device)
            lig_mask = torch.ones(lig_feat.shape[0], dtype=torch.bool).to(device=device)
        # MaskMol, rdkit_vit_224, rdkit_vit_512; ImageMol
        elif isinstance(lig_data, torch.Tensor) and lig_data.shape in {(4, 768), (4, 913)}:
            lig_feat = lig_data.to(device=device)
            lig_mask = torch.ones(lig_feat.shape[0], dtype=torch.bool).to(device=device)
        # self tokenized
        elif isinstance(lig_data, list):
            if len(lig_data) < 256:
                lig_feat = F.pad(torch.tensor(lig_data).to(device=device), (0, 256 - len(lig_data)), value=0)
                lig_mask = (torch.arange(256) <= len(lig_data)).to(device=device)
            else:
                lig_feat = torch.tensor(lig_data)[:256].to(device=device)
                lig_mask = torch.ones_like(lig_feat, dtype=torch.bool).to(device=device)
        # PepDoRA-token
        elif isinstance(lig_data, torch.Tensor) and lig_data.shape == (1, 256):
            lig_feat = lig_data.reshape(-1).to(device=device)
            lig_mask = lig_feat != 0
        # GeminiMol & MolAI
        elif isinstance(lig_data, (torch.Tensor, np.ndarray)) and lig_data.shape in [(1, 2048), (1, 512)]:
            if isinstance(lig_data, np.ndarray):
                lig_data = torch.from_numpy(lig_data)
            lig_feat = lig_data.to(device=device)
            lig_mask = torch.ones((1,), dtype=torch.bool).to(device=device)
        # PepDoRA, ChemBERTa_10M, ChemBERTa_77M_MLM, ChemBERTa_77M_MTR, ...
        else:
            lig_feat = lig_data['hidden_states'].to(device=device).squeeze(0)
            lig_mask = lig_data['attention_mask'].bool().to(device=device).squeeze(0)
        # apply lig data reduce
        if self.lig_seq_reduce is not None:
            lig_feat, lig_mask = self._apply_lig_data_reduce(lig_feat, lig_mask)
        return lig_feat, lig_mask
    
    def _process_rec_ori_data(self, cid: str, rec_data: dict[str, torch.tensor],
                              device: str, prot_trans_start: float, prot_trans_method: str):
        """process original data into the form that can be appended into self.data"""
        if isinstance(rec_data[cid], dict) and 'cls' in rec_data[cid]:
            rec_feat = torch.from_numpy(rec_data[cid]['cls']).to(device=device)
        else:
            rec_feat = rec_data[cid].to(device=device)
        # transform prot data
        if self.prot_transform is not None and self.prot_transform:
            rec_feat = torch.sign(rec_feat) * torch.where(rec_feat.abs() <= prot_trans_start,
                                                          rec_feat.abs(),
                                                          self.prot_transform_fn(rec_feat.abs(), prot_trans_start, prot_trans_method))
        return rec_feat        
        
    def _check_prot_transform(self):
        prot_trans_start, prot_trans_method = eval(self.prot_transform[0]), self.prot_transform[1]
        assert isinstance(prot_trans_start, (float, int)), f'prot_transform start must be float or int, but got {type(prot_trans_start)}'
        if ('.' in prot_trans_method and prot_trans_method[0].isdigit()) or prot_trans_method.isdigit():
            prot_trans_method = float(prot_trans_method)
        elif prot_trans_method in {'log2', 'log10', 'sqrt', 'sqrt0.667'}:
            pass
        else:
            raise ValueError(f'prot_transform method must be float, int, log2, log10, sqrt, sqrt0.667, but got {prot_trans_method}')
        return prot_trans_start, prot_trans_method
        
    def _get_prot_tranform_fn(self, prot_trans_method: str):
        if isinstance(prot_trans_method, float) or isinstance(prot_trans_method, int):
            prot_transform_fn = lambda x, start, method: method * (x.abs()-start) + start
        if prot_trans_method == 'log2':
            prot_transform_fn = lambda x, start, method: torch.log2(x-start+1) + start
        elif prot_trans_method == 'log10':
            prot_transform_fn = lambda x, start, method: torch.log10(x-start+1) + start
        elif prot_trans_method == 'sqrt':
            prot_transform_fn = lambda x, start, method: torch.sqrt(x-start) + start
        elif prot_trans_method == 'sqrt0.667':
            prot_transform_fn = lambda x, start, method: (x-start)**0.667 + start
        return prot_transform_fn
    
    def merge_pad_unimol_data(self, unimol_data: dict[str, torch.Tensor], force_pad: bool = False):
        """unimol-tools 0.1.4.post1 has a limit of 128 max output token(or input?)
        fix it, so pad it to 257 tokens"""
        for k in ['cls', 'hidden_states']:
            unimol_data[k] = torch.from_numpy(unimol_data[k])
        feat = torch.cat([unimol_data['cls'].unsqueeze(0), unimol_data['hidden_states']], dim=0)
        # NOTE: use dynamic padding in __getitem__ to avoid OOM
        if not self.dyn_padding or force_pad:
            feat = F.pad(feat, (0, 0, 0, 257 - feat.size(0)), value=0) # [L, D] -> [257, D]
        mask = torch.cat([torch.ones(1, dtype=torch.bool), unimol_data['attention_mask'].bool()], dim=0) # [256] -> [257]
        return feat, mask
    
    def load_umol_data(self, lig_data: str, rec_data: str, df: pd.DataFrame,
                       load_ratio: float, load_order: str):
        assert isinstance(lig_data, str) or isinstance(rec_data, str), \
            f'at least one is umol data, but got {type(lig_data)} and {type(rec_data)}'
        kwargs = {'load_ratio': load_ratio, 'load_order': load_order}
        # only lig data is umol data
        if isinstance(lig_data, str) and not isinstance(rec_data, str):
            self.dyn_padding = 'v4' in lig_data
            lig_data = self._load_umol_data_for_single_mol(lig_data, df, 'ligands', **kwargs)
        # only rec data is umol data
        elif isinstance(rec_data, str) and not isinstance(lig_data, str):
            self.dyn_padding = 'v4' in rec_data
            rec_data = self._load_umol_data_for_single_mol(rec_data, df, 'pockets', **kwargs)
        # both lig and rec data are umol-v3 data
        elif 'UniMol-v3' in lig_data and 'UniMol-v3' in rec_data:
            unimol_data = self._load_umol_v3_data(lig_data, df)
            lig_data = {cid: unimol_data[cid]['ligands'] for cid in unimol_data}
            rec_data = {cid: unimol_data[cid]['pockets'] for cid in unimol_data}
        # both lig and rec data are umol-v4 data
        elif 'UniMol-v4' in lig_data and 'UniMol-v4' in rec_data:
            self.dyn_padding = True
            data = self._load_umol_v4_data(lig_data, df, **kwargs, mols=('lig', 'poc'))
            lig_data = {cid: data[cid]['ligands'] for cid in data}
            rec_data = {cid: data[cid]['pockets'] for cid in data}
        # lig data is UniMol-v4, rec data is UniMol-v3
        elif 'UniMol-v4' in lig_data and 'UniMol-v3' in rec_data:
            self.dyn_padding = True
            lig_data = self._load_umol_v4_data(lig_data, df, **kwargs, mols='lig')
            data = self._load_umol_v3_data(rec_data, df, **kwargs)
            rec_data = {cid: data[cid]['pockets'] for cid in data}
        else:
            raise NotImplementedError(f'Not support loading umol data {lig_data} and {rec_data}')
        return lig_data, rec_data
        
    def _load_umol_data_for_single_mol(self, umol_dir: str, df: pd.DataFrame, mol: str, **kwargs):
        """mol: str, 'ligands' or 'pockets'"""
        assert 'UniMol-v3' in umol_dir or 'UniMol-v4' in umol_dir, \
            f'umol_dir must be UniMol-v3 or UniMol-v4, but got {umol_dir}'
        if 'UniMol-v3' in umol_dir:
            data = self._load_umol_v3_data(umol_dir, df, **kwargs)
            data = {cid: data[cid][mol] for cid in data}
        elif 'UniMol-v4' in umol_dir:
            data = self._load_umol_v4_data(umol_dir, df, **kwargs, mols='lig' if mol == 'ligands' else 'poc')
        return data
    
    def _load_umol_v3_data(self, umol_dir: str, df: pd.DataFrame, **kwargs):
        data_dict = {}
        for cid in tqdm(df['pdbid'].values, desc='load umol v3 data', leave=False):
            if os.path.exists(os.path.join(umol_dir, f'{cid}.pt')):
                try:
                    data_dict[cid] = torch.load(os.path.join(umol_dir, f'{cid}.pt'), weights_only=False)
                except:
                    put_log(f'load {os.path.join(umol_dir, f"{cid}.pt")} failed')
        return data_dict
    
    def _load_umol_v4_data(self, umol_dir: str, df: pd.DataFrame, load_ratio: float,
                           load_order: str, mols: Union[str, tuple[str, str]] = ('lig', 'poc')):
        """
        load_ratio: float, ratio of data to load, default 1.0
        load_order: str, order of loading data, default 'random', 'front', 'back', 'more'
        mols: Union[str, tuple[str, str]], mols to load, default ('lig', 'poc')
        """
        if isinstance(mols, tuple):
            raise NotImplementedError(f'_load_umol_v4_data only support loading lig or poc, but got {mols}')
        if 'gen' in umol_dir:
            umol_dir = umol_dir.replace('UniMol-v4-gen', 'UniMol-gen')
        data_dict = {}
        for cid in tqdm(df['pdbid'].values, desc='load umol v4 data', leave=False):
            if 'gen' in umol_dir:
                file_name = f'{cid}_ligand.pt'
            else:
                file_name = f'{cid}_unimol_{mols}.pt'
            if os.path.exists(os.path.join(umol_dir, file_name)):
                data_lst = torch.load(os.path.join(umol_dir, file_name), weights_only=False)
                if load_ratio < 1.0:
                    load_num = max(1, int(len(data_lst) * load_ratio))
                    if load_order == 'random':
                        data_lst = random.sample(data_lst, load_num)
                    elif load_order == 'front':
                        data_lst = data_lst[:load_num]
                    elif load_order == 'back':
                        data_lst = data_lst[-load_num:]
                    elif load_order == 'more' and mols == 'poc':
                        raise NotImplementedError(f'_load_umol_v4_data do not support `more` option now')
                data_dict[cid] = data_lst
        return data_dict
    
    def _extract_cids(self, data_dicts: list[dict[str, torch.tensor]]):
        assert len(data_dicts) >= 1, f'_extract_cids must receive at least 1 data dict, but got {len(data_dicts)}'
        cids = set(data_dicts[0].keys())
        for d in data_dicts[1:]:
            cids &= set(d.keys())
        return list(cids)
        
    def calcu_mean_std(self, data_idx: int = 2):
        # 初始化增量统计量
        self.count = 0
        self.mean = torch.zeros(self.data[0][data_idx].shape[-1])
        self.M2 = torch.zeros(self.data[0][data_idx].shape[-1])  # 平方差累积

        # 逐样本增量计算
        for i in tqdm(range(len(self.data)), total=len(self.data), desc='calcu mean, std'):
            # 加载单个样本特征（避免内存爆炸）
            sample_feat = self.data[i][data_idx].float()
            batch_count = sample_feat.size(0)
            
            # Welford算法增量更新
            delta = sample_feat.mean(dim=0) - self.mean
            total_count = self.count + batch_count
            
            self.mean = (self.count * self.mean + batch_count * sample_feat.mean(dim=0)) / total_count
            self.M2 += sample_feat.var(dim=0) * (batch_count - 1) + delta**2 * self.count * batch_count / total_count
            self.count = total_count

        # 计算最终标准差
        self.std = torch.sqrt(self.M2 / (self.count - 1))  # 无偏估计

        # 添加数值稳定性保护
        self.std = torch.clamp(self.std, min=1e-8)
        
    def apply_mean_std(self, mean: torch.Tensor = None, std: torch.Tensor = None, data_idx: int = 2):
        if mean is None:
            mean = self.mean
        if std is None:
            std = self.std
        for i in range(len(self.data)):
            self.data[i][data_idx] = ((self.data[i][data_idx] - mean) / std).to(dtype=self.data[i][data_idx].dtype)
            
    def _apply_lig_data_reduce(self, lig_data: torch.tensor, lig_mask: torch.tensor):
        if self.lig_seq_reduce == 'sum' and lig_data.dim() == 2:
            lig_data = lig_data.sum(dim=0, keepdim=True)
        elif self.lig_seq_reduce == 'mean' and lig_data.dim() == 2:
            lig_data = lig_data.mean(dim=0, keepdim=True)
        elif self.lig_seq_reduce == 'zero' and lig_data.dim() == 2:
            lig_data[~lig_mask.bool()] = 0
            return lig_data, lig_mask
        return lig_data, torch.ones([1], dtype=lig_mask.dtype)
    
    def __getitem__(self, idx):
        if self.umol_v4_pool:
            cid, i = self.data[idx]
            _idx, mid, umol_v4, rec_feat, pKa = self.umol_v4_pool[cid]
            lig_feat, lig_mask = umol_v4[i]
            if self.dyn_padding and self.lig_seq_reduce is None:
                lig_feat = F.pad(lig_feat, (0, 0, 0, 257 - lig_feat.size(0)), value=0) # [L, D] -> [257, D]
            return [_idx, mid, lig_feat, lig_mask, rec_feat, pKa]
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
    def __del__(self):
        if self.umol_v4_pool:
            for i in self.umol_v4_pool.values():
                del i[0], i[0], i[0], i[0], i[0]
        else:
            for i in self.data:
                del i[0], i[0], i[0], i[0], i[0], i[0]
            
    def get_batch_data(self, idx: int, batch_size: int):
        """Return a batch data with orignal order, which same with shuffle=False.
        Parameters:
            idx: index of batch
            batch_size: batch size
            
        Returns:
            data: tuple of (lig_feats, lig_masks, rec_feats, pKas)
        """
        data = self.data[idx * batch_size: (idx + 1) * batch_size]
        mid, lig_feats, lig_masks, rec_feats, pKas = zip(*data)
        mid = torch.stack(mid, dim=0)
        lig_feats = torch.stack(lig_feats, dim=0)
        lig_masks = torch.stack(lig_masks, dim=0)
        rec_feats = torch.stack(rec_feats, dim=0)
        pKas = torch.stack(pKas, dim=0)        
        return mid, lig_feats, lig_masks, rec_feats, pKas
    
def collate_fn(batch: list, dynamic_turncate: bool = False):
    mid, lig_feats, lig_masks, rec_feats, pKas = zip(*batch)
    mid = torch.stack(mid, dim=0)
    lig_feats = torch.stack(lig_feats, dim=0)
    lig_masks = torch.stack(lig_masks, dim=0)
    rec_feats = torch.stack(rec_feats, dim=0)
    pKas = torch.stack(pKas, dim=0)
    if dynamic_turncate:
        max_len = lig_masks.sum(dim=1).max()
        lig_feats = lig_feats[:, :max_len, :]
        lig_masks = lig_masks[:, :max_len]
    return mid, lig_feats, lig_masks, rec_feats, pKas
            
            
class DANNDataset:
    def __init__(self, data: list):
        self.data = data
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)


def get_data_loader(lig_data: dict[str, torch.tensor], rec_data: dict[str, torch.tensor],
                    df: pd.DataFrame, prot_transform: list[str], lig_seq_reduce: str, load_ratio: float,
                    load_order: str, cat_v4_n: int, device: str, batch_size: int,
                    shuffle: bool, num_workers: int=0, logger: logging.Logger = None, drop_last: bool=False):
    dataset = GraphDataset(lig_data, rec_data, df, prot_transform, lig_seq_reduce, load_ratio, load_order, cat_v4_n, device, logger)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
                            pin_memory=True)
    return dataloader
    
    
if __name__ == '__main__':
    name = 'train'
    df = pd.read_csv(f'./data/{name}.csv')
    lig_data = torch.load(os.path.expanduser(f'path-to-your-data-folder/SMILES_PepDoRA.pt'), weights_only=False)
    rec_data = torch.load(os.path.expanduser(f'path-to-your-data-folder/protein_esm3-open_split.pt'), weights_only=False)
    # lig_data = os.path.expanduser(f'path-to-your-data-folder/UniMol-v4-gen')
    # lig_data = torch.load(os.path.expanduser('../EHIGN_dataset/smiles_tokenized/smiles_indices.pt'), weights_only=False)
    # rec_data = lig_data
    dataloader = get_data_loader(lig_data, rec_data, df, [], 'zero', 0.5, 'front', 64, 'cpu', 128, True, 0, None, False)
    # dataloader.dataset.calcu_mean_std()
    # dataloader.dataset.apply_mean_std()
    for idx, mid, lig_feat, lig_mask, rec_feat, pKa in tqdm(dataloader):
        # lig_feat: [b, 256, 384]
        # lig_mask: [b, 256]
        # rec_feat: [b, 1536]
        # pKa: [b, 1]
        pass