import argparse
import os
from glob import glob
from queue import Queue
from threading import Thread

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from config.config_dict import *
from log.train_logger_v1 import *
from models._utils.arg import *
from models.m1.ext_info_constructor_LIT_PCBA import generate_SMILES_dataset
from models.s1.test import _setup
from models.s1.train import Arch1, get_dataset, val
from models.s1.data_loader import GraphDataset
from utils import load_model_dict

warnings.filterwarnings('ignore')


class BatchFromQueue:
    def __init__(self, queue: Queue, prot_data: torch.Tensor, batch_size: int,
                 standerd_loader: GraphDataset, device: str = 'cpu',
                 prot_type: str = 'all_pdbid'):
        self.queue = queue
        self.prot_data = prot_data
        self.prot_keys = list(self.prot_data.keys())
        self.prot_names = list(map(lambda x: '_'.join(x.split('_')[:-1]), self.prot_data.keys()))
        self.batch_size = batch_size
        self.std_loader = standerd_loader.dataset
        self.device = device
        self.mid = torch.zeros(1, dtype=torch.long, device=self.device)
        self.item_id = torch.ones(1, dtype=torch.long, device=self.device)
        # pre-process rec feat
        for k in self.prot_keys:
            self.prot_data[k] = self.std_loader._process_rec_ori_data(k, self.prot_data, self.device, None, None)
        # assign prot_type
        self.prot_type = prot_type
        if self.prot_type == 'all_pdbid':
            self.push_item_to_batch = self.push_item_to_batch_multi_prot
        else:
            self.push_item_to_batch = self.push_item_to_batch_single_prot
        
    def push_item_to_batch_multi_prot(self, pack, batch, item_idx: int):
        cid, lig_data, label = pack
        prot_name = '_'.join(cid.split('_')[:-3])
        prot_feats = [v for k, v in self.prot_data.items() if k.startswith(prot_name)]
        for prot_feat in prot_feats:
            lig_feat, lig_mask = self.std_loader._process_lig_ori_data(lig_data, self.device)
            batch.append([*item_idx, lig_feat, lig_mask, prot_feat, torch.FloatTensor([float(label)])])
        return batch
    
    def push_item_to_batch_single_prot(self, pack, batch, item_idx: int):
        cids, ligs_data, labels = pack
        # process prot_data
        prot_names = list(map(lambda x: '_'.join(x.split('_')[:-3]), cids))
        prot_feats = []
        for prot_name in prot_names:
            prot_feats.extend([v for k, v in self.prot_data.items() if k.startswith(prot_name)])
        if len(prot_feats) != len(cids):
            pass
        prot_feats = torch.stack(prot_feats, dim=0)
        # process lig_data
        lig_feat, lig_mask = self.std_loader._process_lig_ori_data(ligs_data, self.device, batchsize=prot_feats.size(0))
        # process labels
        labels = torch.FloatTensor(labels)
        return cids, self.mid, self.item_id.repeat(len(cids)), lig_feat, lig_mask, prot_feats, labels
    
    def pack_batch(self, batch: list[torch.Tensor]):
        cids, mid, item_id, lig_feat, lig_mask, prot_feat, label = zip(*batch)
        mid = torch.stack(mid, dim=0)
        item_id = torch.stack(item_id, dim=0)
        lig_feat = torch.stack(lig_feat, dim=0)
        lig_mask = torch.stack(lig_mask, dim=0)
        prot_feat = torch.stack(prot_feat, dim=0)
        label = torch.stack(label, dim=0)
        return mid, item_id, lig_feat, lig_mask, prot_feat, label
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch, item_idx = [], 0
        while len(batch) < self.batch_size:
            pack = self.queue.get()
            if pack is None:
                raise StopIteration
            batch = self.push_item_to_batch(pack, batch, item_idx)
            # fast path for push_item_to_batch_single_prot
            if self.prot_type != 'all_pdbid':
                return batch
            item_idx += 1
        if not batch:
            raise StopIteration
        return self.pack_batch(batch)


def calculate_ef(y_true, y_scores, percentage=1.0):
    """
    Enrichment Factor: EF_x% = (前x%中的活性数 / 总活性数) × 100
    """
    y_true = np.array(y_true, dtype=int)
    y_scores = np.array(y_scores, dtype=float)
    n = len(y_true)
    n_actives = int(np.sum(y_true == 1))
    if n_actives == 0:
        return 0.0
    sorted_y = y_true[np.argsort(y_scores)[::-1]]       # 降序排序
    n_top = max(1, int(np.ceil(n * percentage / 100.0)))  # 前x%的个数
    return (np.sum(sorted_y[:n_top]) / n_actives) * 100.0


def calculate_bedroc(y_true, y_scores, alpha=80.5):
    """
    BEDROC: 用指数权重 α 对排名靠前的分子赋予更高重要性
    
    Φ_obs  = (1/N_a) Σ exp(-α·r_i/N)        ← 观测值
    Φ_rand = (1/N)   Σ_{r=1}^{N} exp(-α·r/N) ← 随机期望(等比数列)
    Φ_ideal = (1/N_a) Σ_{i=1}^{N_a} exp(-α·i/N) ← 完美期望
    BEDROC = (Φ_obs - Φ_rand) / (Φ_ideal - Φ_rand)
    """
    y_true = np.array(y_true, dtype=int)
    y_scores = np.array(y_scores, dtype=float)
    n = len(y_true)
    n_actives = int(np.sum(y_true == 1))
    if n_actives == 0 or n == 0:
        return 0.0
    
    # 降序排序, 获取活性分子排名 (1-indexed)
    sorted_y = y_true[np.argsort(y_scores)[::-1]]
    active_ranks = np.where(sorted_y == 1)[0] + 1
    
    phi_obs = np.sum(np.exp(-alpha * active_ranks / n)) / n_actives
    
    factor = np.exp(-alpha / n)
    geo_all = factor * (1.0 - np.exp(-alpha)) / (1.0 - factor)
    phi_rand = geo_all / n
    
    geo_ideal = factor * (1.0 - np.exp(-alpha * n_actives / n)) / (1.0 - factor)
    phi_ideal = geo_ideal / n_actives
    
    denom = phi_ideal - phi_rand
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip((phi_obs - phi_rand) / denom, 0.0, 1.0))


def evaluate_vs_metrics(y_true, y_scores, alpha=80.5):
    """
    一键计算四个虚拟筛选指标
    
    ⚠️ y_scores 应为 "越大越好":
       - pIC50 / pKd / pKi → 直接传入
       - IC50 / Kd (nM)   → 先取 -log10() 或取负值
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    return {
        'EF0.5%': calculate_ef(y_true, y_scores, 0.5),
        'EF1%': calculate_ef(y_true, y_scores, 1.0),
        'EF5%': calculate_ef(y_true, y_scores, 5.0),
        'AUROC': float(roc_auc_score(y_true, y_scores)),
        f'BEDROC(α={alpha})': calculate_bedroc(y_true, y_scores, alpha)
    }


@torch.no_grad()
def run_one_config(cfg: str, batch_loader: DataLoader, models: list[nn.Module],
                   prot_type: str = 'all_pdbid', agg: str = 'mean',
                   data_root: str = '../EHIGN_dataset/LIT-PCBA/AVE_unbiased/'):
    if isinstance(cfg, str):
        config = Config(cfg).get_config()
    else:
        config = cfg
    # validate on LIT-PCBA dataset
    if agg == 'mean':
        pred_lst, label_lst = [], []
    else:
        pred_lst, label_lst = [[], [], []], [[], [], []]
    device, cid_lst = config['training']['device'], []
    for batch in tqdm(batch_loader, desc="validating"):
        cids, mid, item_id, lig_feat, lig_mask, prot_feat, label = batch
        mid = mid.to(device)
        lig_feat = lig_feat.to(device)
        lig_mask = lig_mask.to(device)
        prot_feat = prot_feat.to(device)
        pred = torch.stack([model(mid, lig_feat, lig_mask, prot_feat) for model in models], dim=0)
        cid_lst.extend(cids)
        if agg == 'mean':
            pred = pred.mean(dim=0).reshape(-1)
        if prot_type == 'all_pdbid':
            num_items = item_id.max().item() + 1
            pred_out = torch.zeros(num_items, dtype=pred.dtype, device=device)
            pred_out = pred_out.scatter_reduce(dim=0, index=item_id.to(device).reshape(-1), src=pred, reduce="mean", include_self=False)
            label_out = torch.zeros(num_items, dtype=label.dtype, device=device)
            label_out = label_out.scatter_reduce(dim=0, index=item_id.to(device).reshape(-1), src=label.to(device).reshape(-1), reduce="mean", include_self=False)
        elif agg == 'mean':
            pred_out = pred.reshape(-1)
            label_out = label.reshape(-1)
        if agg == 'mean':
            pred_lst.extend(pred_out.cpu().numpy().tolist())
            label_lst.extend(label_out.cpu().numpy().tolist())
        else:
            [pred_lst[i].extend(pred_out.cpu().numpy().tolist()) for i in range(len(pred_lst))]
            [label_lst[i].extend(label_out.cpu().numpy().tolist()) for i in range(len(label_lst))]
    metrics = evaluate_vs_metrics(label_lst, pred_lst)
    record_df = pd.DataFrame({'cid': cid_lst, 'pred': pred_lst, 'label': label_lst})
    print(metrics)
       # free cuda mem
    if 'cuda' in config['training']['device']:
        torch.cuda.empty_cache()
    return metrics, record_df


def run_one_root(args: argparse.Namespace):
    if not os.path.exists(f'./checkpoints/{args.path}/randomseed0/log/train'):
        print(f'./checkpoints/{args.path}/randomseed0/log/train/ not exists')
        exit(1)
    if os.path.exists(f'./checkpoints/{args.path}/LIT-PCBA_records.parquet'):
        print(f'./checkpoints/{args.path}/LIT-PCBA_records.parquet already exists')
        return
    else:        
        print(f'processing ./checkpoints/{args.path}')
    # return 
    
    path = os.path.join(f'./checkpoints/{args.path}/randomseed0/log/train/*.json')
    path = glob(path)[0]
    config = opts_file(path, way='json')
    config['data']['batch_size'] = args.batch_size
    config['training']['random_seed'] = [0, 3407, 777]
    
    this_run_dir = f'./checkpoints/{args.path}'
    # get PDBBind dataloader
    _, valid_loader, _, _, _ = get_dataset(config['data'], None, val_mode=True)
    # load models
    models = []
    for seed in config['training']['random_seed']:
        ckp_path = os.path.join(this_run_dir, f"randomseed{seed}", "model", "*.pt")
        best_model_list = glob(ckp_path)
        if len(best_model_list) == 0:
            continue
        model, device = _setup(config, seed, valid_loader, best_model_list)
        models.append(model)
    # get LIT-PCBA dataset
    targets = pd.read_excel('data/LIT_PCBA.xlsx', sheet_name='Sheet2')['folder_name'].tolist()
    prot_data = torch.load(f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased/{args.prot_type}/protein_{config["data"]["prot_type"]}.pt', weights_only=False)
    root_queue, queue = Queue(), Queue()
    thread = Thread(target=generate_SMILES_dataset, args=(root_queue, queue,
                                                          config['data']['lig_type'], '.smi_std',
                                                          224, len(targets), args.batch_size))
    thread.start()
    batch_loader = BatchFromQueue(queue, prot_data, config['data']['batch_size'],
                                  valid_loader, prot_type=args.prot_type)

    metrics, dfs = {}, []
    for target_name in targets:
        data_root = f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased/{target_name}/'
        root_queue.put(data_root)
        metrics[target_name], record_df = run_one_config(config, batch_loader, models,
                                                        prot_type=args.prot_type, agg='mean', data_root=data_root)
        dfs.append(record_df)
    df = pd.DataFrame(metrics).T
    df.to_excel(f'./checkpoints/{args.path}/LIT-PCBA_metrics.xlsx', index=True)
    record_df = pd.concat(dfs, axis=0)
    record_df.to_parquet(f'./checkpoints/{args.path}/LIT-PCBA_records.parquet', index=True)
    

if __name__ == '__main__':
    # command launch
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument("-p", "--path", type=str,
                            default=None, help='20251213-115323-s1_arch1MLP,12,1|sum_adamw1e-4_PepDoRA_2h512do0.1ns0.05lns0.05bs196')
    args_paser.add_argument("-bs", "--batch-size", type=int, default=256,
                            help="batch size, default is %(default)s")
    args_paser.add_argument("--prot-type", type=str, choices=['all_pdbid', 'PLANET'], default='PLANET',
                            help="prot type, default is %(default)s")
    
    args = args_paser.parse_args()
    if args.path is not None:
        run_one_root(args)