import argparse
import os
from glob import glob
from pathlib import Path
from queue import Queue
from threading import Thread

from mbapy.file import get_paths_with_extension
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
from models.m1.ext_info_constructor_DUDEZ import generate_SMILES_dataset
from models.s1.test import _setup
from models.s1.train import Arch1, get_dataset, val
from models.s1.data_loader import GraphDataset
from models.s1.test_LIT_PCBA import evaluate_vs_metrics
from utils import load_model_dict

warnings.filterwarnings('ignore')


class BatchFromQueue:
    def __init__(self, queue: Queue, prot_data: torch.Tensor, batch_size: int,
                 standerd_loader: GraphDataset, device: str = 'cpu'):
        self.queue = queue
        self.prot_data = prot_data
        self.batch_size = batch_size
        self.std_loader = standerd_loader.dataset
        self.device = device
        self.mid = torch.zeros(1, dtype=torch.long, device=self.device)
        self.item_id = torch.ones(1, dtype=torch.long, device=self.device)
    
    def push_item_to_batch_single_prot(self, pack):
        cids, ligs_data, labels = pack
        prot_feats = self.prot_data.repeat(len(cids), 1)
        # process lig_data
        lig_feat, lig_mask = self.std_loader._process_lig_ori_data(ligs_data, self.device, batchsize=prot_feats.size(0))
        # process labels
        labels = torch.FloatTensor(labels)
        return self.mid, lig_feat, lig_mask, prot_feats, labels
    
    def __iter__(self):
        return self
    
    def __next__(self):
        pack = self.queue.get()
        if pack is None:
            raise StopIteration
        batch = self.push_item_to_batch_single_prot(pack)
        if not batch:
            raise StopIteration
        return batch


@torch.no_grad()
def run_one_config(cfg: str, models: list[nn.Module], device: str, valid_loader: DataLoader, target_name: str, agg: str = 'mean',
                   data_root: str = '../EHIGN_dataset/DUDE_Z/'):
    if isinstance(cfg, str):
        config = Config(cfg).get_config()
    else:
        config = cfg
    # get DUDE-Z dataset
    prot_data = torch.load(f'../EHIGN_dataset/DUDE_Z/protein_{config["data"]["prot_type"]}.pt',
                           weights_only=False)
    if target_name not in prot_data.keys():
        print(f'prot_key {target_name} not in prot_data')
        return {}
    prot_data = valid_loader.dataset._process_rec_ori_data(target_name, prot_data, device, None, None)
    queue = Queue()
    thread = Thread(target=generate_SMILES_dataset, args=(data_root, queue,
                                                          config['data']['lig_type'], '_std.smi'))
    thread.start()
    batch_loader = BatchFromQueue(queue, prot_data, config['data']['batch_size'], valid_loader)
    # validate on DUDE-Z dataset
    if agg == 'mean':
        pred_lst, label_lst = [], []
    else:
        pred_lst, label_lst = [[], [], []], [[], [], []]
    for batch in tqdm(batch_loader, desc="validating"):
        mid, lig_feat, lig_mask, prot_feat, label = batch
        mid = mid.to(device)
        lig_feat = lig_feat.to(device)
        lig_mask = lig_mask.to(device)
        prot_feat = prot_feat.to(device)
        pred = torch.stack([model(mid, lig_feat, lig_mask, prot_feat) for model in models], dim=0)
        if agg == 'mean':
            pred = pred.mean(dim=0).reshape(-1)
            pred_out = pred.reshape(-1)
            label_out = label.reshape(-1)
        if agg == 'mean':
            pred_lst.extend(pred_out.cpu().numpy().tolist())
            label_lst.extend(label_out.cpu().numpy().tolist())
        else:
            [pred_lst[i].extend(pred_out.cpu().numpy().tolist()) for i in range(len(pred_lst))]
            [label_lst[i].extend(label_out.cpu().numpy().tolist()) for i in range(len(label_lst))]
    metrics = evaluate_vs_metrics(label_lst, pred_lst)
    print(metrics)
    # free cuda mem
    torch.cuda.empty_cache()
    return metrics


def run_one_root(args: argparse.Namespace):
    if not os.path.exists(f'./checkpoints/{args.path}/randomseed0/log/train'):
        print(f'./checkpoints/{args.path}/randomseed0/log/train/ not exists')
        exit(1)
    # if os.path.exists(f'./checkpoints/{args.path}/DUDEZ_metrics.xlsx'):
    #     print(f'./checkpoints/{args.path}/DUDEZ_metrics.xlsx already exists')
    #     return 
    
    path = os.path.join(f'./checkpoints/{args.path}/randomseed0/log/train/*.json')
    path = glob(path)[0]
    config = opts_file(path, way='json')
    config['data']['batch_size'] = 256
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

    metrics = {}
    for _type in ['ALL', 'DUDE_Z', 'Extrema', 'Goldilocks']:
        result_path = f'./checkpoints/{args.path}/DUDEZ_{_type}_metrics.xlsx'
        if os.path.exists(result_path):
            print(f'{result_path} already exists')
            continue 
        for rec_path in get_paths_with_extension('../EHIGN_dataset/DUDE_Z', ['rec.crg.pdb']):
            if _type in {'ALL'}:
                data_root = Path(rec_path).parent
                target_name = data_root.name
            else:
                data_root = Path(rec_path).parent / _type
                target_name = data_root.parent.name
            metrics[target_name] = run_one_config(config, models, device, valid_loader, target_name,
                                                agg='mean', data_root=str(data_root))
        df = pd.DataFrame(metrics).T
        df.to_excel(result_path, index=True)
    

if __name__ == '__main__':
    # command launch
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument("-p", "--path", type=str,
                            default=None,
                            help="checkpoint path, 20251213-115323-s1_arch1MLP,12,1|sum_adamw1e-4_PepDoRA_2h512do0.1ns0.05lns0.05bs196")
    
    args = args_paser.parse_args()
    if args.path is not None:
        run_one_root(args)