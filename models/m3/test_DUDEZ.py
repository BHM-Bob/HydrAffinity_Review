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
from mbapy.dl_torch.utils import set_random_seed

from config.config_dict import *
from log.train_logger_v1 import *
from models._utils.arg import *
from models.m1.ext_info_constructor_DUDEZ import generate_SMILES_dataset
from models.m3.train import get_dataset, get_model, get_data_shape_from_dataset
from models.m3.data_loader import GraphDataset
from models.s1.test_LIT_PCBA import evaluate_vs_metrics
from models.m3.test_LIT_PCBA import _setup, BatchFromQueue

warnings.filterwarnings('ignore')


@torch.no_grad()
def run_one_config(cfg: str, valid_loader: DataLoader, models: list, device: torch.device, agg: str = 'mean',
                   data_root: str = '../EHIGN_dataset/DUDE_Z/'):
    if isinstance(cfg, str):
        config = Config(cfg).get_config()
    else:
        config = cfg
    # get LIT-PCBA dataset
    queue = Queue()
    assert len(config['data']['lig_type']) == 1, f'Only support one ligand, found {config["data"]["lig_type"]}'
    thread = Thread(target=generate_SMILES_dataset, args=(data_root, queue,
                                                          config['data']['lig_type'][0], '_std.smi'))
    thread.start()
    batch_loader = BatchFromQueue({config['data']['lig_type'][0]: queue},
                                  config['data']['batch_size'], valid_loader, {})
    # validate on LIT-PCBA dataset
    if agg == 'mean':
        pred_lst, label_lst = [], []
    else:
        pred_lst, label_lst = [[], [], []], [[], [], []]
    for batch in tqdm(batch_loader, desc="validating"):
        pack, label = batch
        data = {k: [v[0].to(device), v[1].to(device)] for k, v in pack.items()}
        pred = torch.stack([model(data) for model in models], dim=0)
        if agg == 'mean':
            pred = pred.mean(dim=0).reshape(-1)
            pred_out = pred.reshape(-1)
            label_out = label.reshape(-1)
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
    
    if all([os.path.exists(f'./checkpoints/{args.path}/DUDEZ_{_type}_metrics.xlsx') for _type in ['ALL', 'DUDE_Z', 'Extrema', 'Goldilocks']]):
        print(f'{args.path} already evaluated')
        return 
    
    path = os.path.join(f'./checkpoints/{args.path}/randomseed0/log/train/*.json')
    path = glob(path)[0]
    config = opts_file(path, way='json')
    config['data']['batch_size'] = 256
    config['training']['random_seed'] = [0, 3407, 777]
    
    # get PDBBind dataloader
    _, valid_loader, _, _, _ = get_dataset(config['data'], None, val_mode=True)
    # load models
    this_run_dir = f'./checkpoints/{args.path}'
    print(this_run_dir)
    models, device = _setup(config, this_run_dir)

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
            metrics[target_name] = run_one_config(config, valid_loader, models, device,
                                                  agg='mean', data_root=str(data_root))
        df = pd.DataFrame(metrics).T
        df.to_excel(result_path, index=True)


if __name__ == '__main__':
    # command launch
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument("-p", "--path", type=str,
                            default=None,
                            help="checkpoint path, default is %(default)s")
    args_paser.add_argument("--prot-type", type=str, choices=['all_pdbid', 'PLANET'], default='PLANET',
                            help="prot type, default is %(default)s")
    
    args = args_paser.parse_args()
    if args.path is not None:
        run_one_root(args)