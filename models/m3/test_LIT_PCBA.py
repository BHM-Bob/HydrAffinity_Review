import argparse
import os
import time
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
from mbapy.dl_torch.utils import set_random_seed

from config.config_dict import *
from log.train_logger_v1 import *
from models._utils.arg import *
from models.m1.ext_info_constructor_LIT_PCBA import generate_SMILES_dataset
from models.m3.train import get_dataset, get_model, get_data_shape_from_dataset, load_rec_data_by_name
from models.m3.data_loader import GraphDataset
from models.s1.test_LIT_PCBA import evaluate_vs_metrics
from utils import load_model_dict

warnings.filterwarnings('ignore')


def load_rec_data_by_name(data_names: list[str], prot_type: str):
    """
    support data names: ems3, esm2, SaProt, ProSST-2048
    """
    datas, root = {}, f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased/{prot_type}/'
    for name in data_names:
        if name == 'esm2-mean_each_mean':
            datas[name] = torch.load(os.path.join(root, f'protein_esm2-mean_each_mean.pt'), weights_only=False)
        elif name == 'esm2-3B-36_mean_each_mean':
            datas[name] = torch.load(os.path.join(root, f'protein_esm2-3B-36_mean_each_mean.pt'), weights_only=False)
        elif name == 'esm2-3B-30_mean_each_mean':
            datas[name] = torch.load(os.path.join(root, f'protein_esm2-3B-30_mean_each_mean.pt'), weights_only=False)
        elif name == 'esm3-open_split':
            datas[name] = torch.load(os.path.join(root, f'protein_esm3-open_split.pt'), weights_only=False)
        elif name == 'SaProt':
            datas[name] = torch.load(os.path.join(root, f'protein_SaProt.pt'), weights_only=False)
        elif name == 'esm2-3B':
            datas[name] = torch.load(os.path.join(root, f'protein_esm2-3B_mean_each_mean.pt'), weights_only=False)
        elif name == 'ProSST-2048':
            datas[name] = torch.load(os.path.join(root, f'protein_ProSST-2048_mean_each_mean.pt'), weights_only=False)
        else:
            raise ValueError(f'rec_data_name {name} not supported')
    return datas


class BatchFromQueue:
    def __init__(self, queue: dict[str, Queue], batch_size: int, standerd_loader: DataLoader,
                 prot_data: dict[str, torch.Tensor], device: str = 'cpu'):
        self.queue = queue
        self.batch_size = batch_size
        self.std_loader: GraphDataset = standerd_loader.dataset
        self.prot_data = prot_data
        self.device = device
        self.mid = torch.zeros(1, dtype=torch.long, device=self.device)
        self.item_id = torch.ones(1, dtype=torch.long, device=self.device)
    
    def push_item_to_batch_single_prot(self, pack: dict[str, torch.Tensor]):
        data_pack = {}
        for lig_key, lig_pack in pack.items():
            cids, ligs_data, labels = lig_pack
            # process lig_data
            lig_feat, lig_mask = self.std_loader._process_lig_ori_data(ligs_data, self.device, batchsize=len(cids))
            data_pack[lig_key] = [lig_feat, lig_mask]
            # process labels
            labels = torch.FloatTensor(labels)
        # add prot data
        prot_names = list(map(lambda x: '_'.join(x.split('_')[:-3]), cids))
        for k, v in self.prot_data.items():
            prot_feats = []
            for prot_name in prot_names:
                prot_feats.extend([v for k, v in self.prot_data[k].items() if k.startswith(prot_name)])
            prot_feats = torch.stack(prot_feats, dim=0)
            prot_mask = torch.ones((len(cids), 1), dtype=torch.long) # [len(cids), 1]
            data_pack[k] = [prot_feats, prot_mask]
        return data_pack, labels
    
    def __iter__(self):
        return self
    
    def __next__(self):
        pack = {k: v.get() for k, v in self.queue.items()}
        if all(v is None for v in pack.values()):
            raise StopIteration
        return self.push_item_to_batch_single_prot(pack)


def _setup(config, this_run_dir: str):
    # get dataloader
    _, valid_loader, _, _, _ = get_dataset(config['data'], None, val_mode=True)
    # load models
    models = []
    for seed in config['training']['random_seed']:
        ckp_path = os.path.join(this_run_dir, f"randomseed{seed}", "model", "*.pt")
        best_model_list = glob(ckp_path)
        if len(best_model_list) == 0:
            continue
        # set random seed
        config['training']['now_random_seed'] = seed
        set_random_seed(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        # model
        device = torch.device(config['training']['device'])
        data_shapes = get_data_shape_from_dataset(valid_loader.dataset)
        model =  get_model(config['model'], data_shapes, None).to(device)
        if not (config['training']['no_compile']):
            model: nn.Module = torch.compile(model)
        # final testing
        load_model_dict(model, best_model_list[-1])
        model = model.to(device)
        model.eval()
        models.append(model)
    return models, device

@torch.no_grad()
def run_one_config(cfg: str, batch_loader: DataLoader, models: list[nn.Module], agg: str = 'mean'):
    if isinstance(cfg, str):
        config = Config(cfg).get_config()
    else:
        config = cfg
    # validate on LIT-PCBA dataset
    if agg == 'mean':
        pred_lst, label_lst = [], []
    else:
        pred_lst, label_lst = [[], [], []], [[], [], []]
    for batch in tqdm(batch_loader, desc="validating"):
        pack, label = batch
        data = {k: [v[0].to('cuda'), v[1].to('cuda')] for k, v in pack.items()}
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
    # if os.path.exists(f'./checkpoints/{args.path}/LIT-PCBA_metrics.xlsx'):
    #     print(f'./checkpoints/{args.path}/LIT-PCBA_metrics.xlsx already exists')
    #     return 
    
    path = os.path.join(f'./checkpoints/{args.path}/randomseed0/log/train/*.json')
    path = glob(path)[0]
    config = opts_file(path, way='json')
    config['data']['batch_size'] = 256
    config['training']['random_seed'] = [0, 3407, 777]
    
    # get PDBBind dataloader
    _, valid_loader, _, _, _ = get_dataset(config['data'], None, val_mode=True)
    # load models
    models, device = _setup(config, f'./checkpoints/{args.path}')
    # get LIT-PCBA dataset
    prot_data = load_rec_data_by_name(config['data']['prot_type'], args.prot_type)
    targets = pd.read_excel('data/LIT_PCBA.xlsx', sheet_name='Sheet2')['folder_name'].tolist()
    data_ques, root_ques = {}, []
    for lig_name in config['data']['lig_type']:
        queue, data_root_queue = Queue(), Queue()
        data_ques[lig_name] = queue
        root_ques.append(data_root_queue)
        thread = Thread(target=generate_SMILES_dataset, args=(data_root_queue, queue, lig_name, '.smi_std', 224, len(targets)))
        thread.start()
        time.sleep(10)
    batch_loader = BatchFromQueue(data_ques, config['data']['batch_size'], valid_loader, prot_data)

    metrics = {}
    for target_name in targets:
        data_root = f'../EHIGN_dataset/LIT-PCBA/AVE_unbiased/{target_name}/'
        [i.put(data_root) for i in root_ques]
        metrics[target_name] = run_one_config(config, batch_loader, models, agg='mean')
    df = pd.DataFrame(metrics).T
    df.to_excel(f'./checkpoints/{args.path}/LIT-PCBA_metrics.xlsx', index=True)
    

if __name__ == '__main__':
    # command launch
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument("-p", "--path", type=str,
                            default='20251130-000800-m3_arch1MLP,12,6|weighted_adamw1e-4_ChemBERTa_77M_MTR,GeminiMol,MolFormer_2h512do0.1ns0.05lns0.05bs128',
                            help="checkpoint path, default is %(default)s")
    args_paser.add_argument("--prot-type", type=str, choices=['all_pdbid', 'PLANET'], default='PLANET',
                            help="prot type, default is %(default)s")
    
    args = args_paser.parse_args()
    run_one_root(args)