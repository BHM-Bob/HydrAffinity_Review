import argparse
import os
from glob import glob

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import warnings
import pandas as pd
import torch
import torch.nn as nn
from mbapy.dl_torch.utils import set_random_seed

from config.config_dict import *
from log.train_logger_v1 import *
from models._utils.arg import *
from models.s1.train import Arch1, get_dataset, get_model, val
from utils import load_model_dict

warnings.filterwarnings('ignore')


def _setup(config, seed: int, valid_loader: torch.utils.data.DataLoader, best_model_list: list):
    # set random seed
    config['training']['now_random_seed'] = seed
    set_random_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    
    if config['data']['lig_type'] in {'token', 'PepDoRA-token'}:
        lig_dim = 1
        prot_dim = valid_loader.dataset.data[0][4].shape[-1]
    elif valid_loader.dataset.umol_v4_pool:
        # umol_v4_pool: List[List[idx, mid, List[feat, mask]], rec_feat, pKa]
        lig_dim = valid_loader.dataset.umol_v4_pool[valid_loader.dataset.data[0][0]][2][0][0].shape[-1]
        prot_dim = valid_loader.dataset.umol_v4_pool[valid_loader.dataset.data[0][0]][3].shape[-1]
    else:
        lig_dim = valid_loader.dataset.data[0][2].shape[-1]
        prot_dim = valid_loader.dataset.data[0][4].shape[-1]
    # model
    device = torch.device(config['training']['device'])
    model =  get_model(config['model'], lig_dim, prot_dim, None).to(device)
    # if not config['training']['no_compile']:
    #     model: nn.Module = torch.compile(model)
    # load_model_dict(model, best_model_list[-1])
    ckp = torch.load(best_model_list[-1])
    ckp = {k.replace('_orig_mod.', ''): v for k, v in ckp.items()}
    model.load_state_dict(ckp)
    model = model.to(device)
    model.eval()
    return model, device

def run_one_config(cfg: str, this_run_dir: str):
    if isinstance(cfg, str):
        config = Config(cfg).get_config()
    else:
        config = cfg
    _log_fn = print
    _log_fn(__file__)
    _log_fn(config)
    # get dataloader
    _, valid_loader, test2013_loader, test2016_loader, test2019_loader = get_dataset(config['data'], None, val_mode=True)
    # train for each random seed
    for seed in config['training']['random_seed']:
        ckp_path = os.path.join(this_run_dir, f"randomseed{seed}", "model", "*.pt")
        best_model_list = glob(ckp_path)
        if len(best_model_list) == 0:
            continue
        # start test
        model, device = _setup(config, seed, valid_loader, best_model_list)
        with torch.no_grad():
            _, valid_rmse, valid_pr = val(model, valid_loader, device, config)
            _, test2013_rmse, test2013_pr = val(model, test2013_loader, device, config)
            _, test2016_rmse, test2016_pr = val(model, test2016_loader, device, config)
            _, test2019_rmse, test2019_pr = val(model, test2019_loader, device, config)
        msg = "valid_rmse:%.4f, valid_pr:%.4f, test2013_rmse:%.4f, test2013_pr:%.4f, test2016_rmse:%.4f, test2016_pr:%.4f, test2019_rmse:%.4f, test2019_pr:%.4f," \
                    % (valid_rmse, valid_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, test2019_rmse, test2019_pr)
        print(msg)
        # append the msg at the last line in the log file
        with open(os.path.join(this_run_dir, f"randomseed{seed}", "log", "train", "Train.log"), "a") as f:
            f.write(msg + "\n")
    # free cuda mem
    if 'cuda' in config['training']['device']:
        torch.cuda.empty_cache()
        
@torch.no_grad()
def eval_moe_activation(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, config):
    lig_fc_act, prot_fc_act, pred_act, idx_lst = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval moe act", leave=False):
            batch = [b.to(device) for b in batch]
            idx, mid, lig, mask, rec, _ = batch
            model(mid, lig, mask, rec)
            if hasattr(model.lig_fc, 'topk_idx'):
                if model.lig_fc.topk_idx.shape[0] != idx.shape[0]:
                    # TokenMoE
                    lig_fc_act.append(torch.cat([idx.repeat(1, lig.size(1)).reshape(-1, 1),
                                                 model.lig_fc.topk_idx], dim=-1).cpu())
                else:
                    lig_fc_act.append(torch.cat([idx, model.lig_fc.topk_idx], dim=-1).cpu())
            if hasattr(model.prot_fc, 'topk_idx'):
                prot_fc_act.append(torch.cat([idx, model.prot_fc.topk_idx], dim=-1).cpu())
            if hasattr(model.predictor, 'topk_idx'):
                pred_act.append(torch.cat([idx, model.predictor.topk_idx], dim=-1).cpu())
            idx_lst.append(idx.cpu())
    idx = torch.cat(idx_lst, dim=0)
    cids = [loader.dataset.valid_key[i][0] for i in idx]
    pKa = [loader.dataset.valid_key[i][1] for i in idx]
    lig_act = torch.cat(lig_fc_act, dim=0) if lig_fc_act else None
    prot_act = torch.cat(prot_fc_act, dim=0) if prot_fc_act else None
    pred_act = torch.cat(pred_act, dim=0) if pred_act else None
    return lig_act, prot_act, pred_act, cids, pKa


def test_moe_exp_dis(cfg: str, this_run_dir: str):
    if isinstance(cfg, str):
        config = Config(cfg).get_config()
    else:
        config = cfg
    print(__file__)
    print(config)
    # get dataloader
    train_loader, valid_loader, test2013_loader, test2016_loader, test2019_loader = get_dataset(config['data'], None)
    # train for each random seed
    result = {}
    for seed in config['training']['random_seed']:
        result[seed] = {'lig_act': [], 'prot_act': [], 'pred_act': [], 'cids': [], 'pKa': []}
        ckp_path = os.path.join(this_run_dir, f"randomseed{seed}", "model", "*.pt")
        best_model_list = glob(ckp_path)
        if len(best_model_list) == 0:
            continue
        # start test
        model, device = _setup(config, seed, valid_loader, best_model_list)
        # enable MoE act record
        if hasattr(model.lig_fc, 'balance_loss'):
            model.lig_fc._RECORD_EXPERT_ACTIVATION = True
        if hasattr(model.prot_fc, 'balance_loss'):
            model.prot_fc._RECORD_EXPERT_ACTIVATION = True
        if hasattr(model.predictor, 'balance_loss'):
            model.predictor._RECORD_EXPERT_ACTIVATION = True
        for loader in [train_loader, valid_loader, test2013_loader, test2016_loader, test2019_loader]:
            lig_act, prot_act, pred_act, cids, pKa = eval_moe_activation(model, loader, device, config)
            result[seed]['lig_act'].append(lig_act)
            result[seed]['prot_act'].append(prot_act)
            result[seed]['pred_act'].append(pred_act)
            result[seed]['cids'].extend(cids)
            result[seed]['pKa'].extend(pKa)
        result[seed]['lig_act'] = torch.cat(result[seed]['lig_act'], dim=0) if result[seed]['lig_act'][0] is not None else None
        result[seed]['prot_act'] = torch.cat(result[seed]['prot_act'], dim=0) if result[seed]['prot_act'][0] is not None else None
        result[seed]['pred_act'] = torch.cat(result[seed]['pred_act'], dim=0) if result[seed]['pred_act'][0] is not None else None
    # free cuda mem
    if 'cuda' in config['training']['device']:
        torch.cuda.empty_cache()
    # save the result
    torch.save(result, os.path.join(this_run_dir, "prot_moe_act.pt"))

def get_mask(df: pd.DataFrame, kwargs: dict, ext_mask: np.ndarray = None) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for k, v in kwargs.items():
        if v is None:
            mask &= df[k].isna()
        elif isinstance(v, list):
            mask_i = np.zeros(len(df), dtype=bool)
            for v_i in v:
                if v_i is None:
                    mask_i |= df[k].isna()
                else:
                    mask_i |= (df[k] == v_i)
            mask &= mask_i
        else:
            mask &= (df[k] == v)
    return mask & ext_mask if ext_mask is not None else mask

if __name__ == '__main__':
    # command launch
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument("-p", "--path", type=str, nargs='+', default=None,
                            help="checkpoint path, default is %(default)s")
    
    args = args_paser.parse_args()


    for i, path in enumerate(args.path):
        args.path = path
        print(f'rocessing {i}/{len(args.path)}: {path}')
        if os.path.exists(f'./checkpoints/{args.path}/prot_moe_act.pt'):
            d = torch.load(f'./checkpoints/{args.path}/prot_moe_act.pt', weights_only=False)
            if all(i in d for i in [0, 3407, 777]):
                print(f'./checkpoints/{args.path}/prot_moe_act.pt already exists')
                continue
        if not os.path.exists(f'./checkpoints/{args.path}/randomseed0/log/train'):
            print(f'./checkpoints/{args.path}/randomseed0/log/train/ not exists')
            continue
        
        path = os.path.join(f'./checkpoints/{args.path}/randomseed0/log/train/*.json')
        path = glob(path)[0]
        config = opts_file(path, way='json')
        config['data']['batch_size'] = 256
        config['training']['random_seed'] = [0, 3407, 777]
        # run_one_config(config, f'./checkpoints/{args.path}')
        test_moe_exp_dis(config, f'./checkpoints/{args.path}')
