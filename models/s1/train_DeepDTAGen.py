
import argparse
import os
from glob import glob
from typing import Union

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score

from config.config_dict import *
from log.train_logger_v1 import *
from models._utils.arg import *
from models.s1.data_loader import (GraphDatasetCompress,
                                   get_data_loader_compress)
from models.s1.test import _setup, concordance_index
from models.s1.train import make_args, run_one_config

warnings.filterwarnings('ignore')


def get_dataset(config: dict, logger, val_mode: bool = False, debug: bool = False,
                train_split: Union[str, dict[str, list[str]]] = None, seed: int = None):   
    # replace ~ in paths
    for n in ['data_root', 'bin_root']:
        config[n] = config[n].replace('/home/SERVER/', '~/')
        config[n] = os.path.expanduser(config[n])
    # load data and do pre-process
    ds = GraphDatasetCompress()
    prot_data_path = os.path.join(config['bin_root'], f'DeepDTAGen/protein_{config["prot_type"]}.pt')
    prot_data = torch.load(prot_data_path, map_location='cpu', weights_only=False)
    for cid in tqdm(prot_data.keys(), desc='pre-process prot_data'):
        prot_data[cid] = ds._process_rec_ori_data(cid, prot_data, 'cpu', None, None)
    lig_data_path = os.path.join(config['bin_root'], f'DeepDTAGen/SMILES_{config["lig_type"]}.pt')
    lig_data = torch.load(lig_data_path, map_location='cpu', weights_only=False)
    for cid in tqdm(lig_data.keys(), desc='pre-process lig_data'):
        lig_data[cid] = ds._process_lig_ori_data(lig_data[cid], 'cpu')
    # load datasets
    num_workers = 0
    if logger is not None:
        logger.info(f'num_workers: {num_workers}')
    else:
        print(f'num_workers: {num_workers}')
    if val_mode:
        train_loader = None
    else:
        df = pd.read_csv(f'./data/DeepDTAGen/{train_split}_train.csv')
        if debug:
            df = df.sample(frac=0.1, random_state=42)
        # torch compile use dynamic shape, drop_last will make shape consistent in whole epoch
        train_loader = get_data_loader_compress(lig_data, prot_data, df, config['batch_size'], True, num_workers, logger, drop_last=False)
    df = pd.read_csv(f'./data/DeepDTAGen/{train_split}_test.csv')
    test_loader = get_data_loader_compress(lig_data, prot_data, df, config['batch_size'], False, num_workers, logger, drop_last=False)
    # train, valid, test2013, test2016, test2019
    return train_loader, None, None, test_loader, None


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)

    mult = sum((y_pred - np.mean(y_pred)) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - np.mean(y_pred)) ** 2)

    return mult / float(y_obs_sq * y_pred_sq)

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)

    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)

    return 1 - (upp / down)

def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(abs((r2 * r2) - (r02 * r02))))

def get_aupr(predictions, true_labels, threshold):
    binary_pred = (predictions > threshold).astype(int)
    binary_true = (true_labels > threshold).astype(int)
    return average_precision_score(binary_true, binary_pred)

def val_on_metrics(model, loader, device, config):
    model.eval()
    pred_list = []
    label_list = []
    for data in tqdm(loader, desc="Validating", leave=False):
        data = [i.to(device) for i in data]
        inputs, label = data[1:-1], data[-1]
        with torch.no_grad():
            pred = model(*inputs)
            pred_list.append(pred.detach().cpu().view(-1).numpy())
            label_list.append(label.detach().cpu().view(-1).numpy())
    pred = np.concatenate(pred_list, axis=0)
    if pred.mean() < 0:
        pred = -pred
    label = np.concatenate(label_list, axis=0)
    
    mse = np.mean((pred - label) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - label))
    pr: float = pearsonr(pred, label)[0]
    ci = concordance_index(label, pred)
    rm2 = get_rm2(label, pred)
    if config['data']['dataset'] == 'kiba':
        # thresholds = [10.0, 10.50, 11.0, 11.50, 12.0, 12.50]
        aupr_threshold = 12.1
    elif config['data']['dataset'] in ['davis', 'bindingdb']:
        # thresholds = [5.0, 5.50, 6.0, 6.50, 7.0, 7.50, 8.0, 8.50]
        aupr_threshold = 7.0
    aupr = get_aupr(pred, label, aupr_threshold)
    return mse, rmse, mae, pr, ci, rm2, aupr


def test_from_root(root):
    path = os.path.join(f'./checkpoints/{root}/randomseed0/log/train/*.json')
    this_run_dir = f'./checkpoints/{root}'
    path = glob(path)[0]
    config = opts_file(path, way='json')
    config['data']['batch_size'] = 256
    config['training']['random_seed'] = [0, 3407, 777]
    _log_fn = print
    _log_fn(__file__)
    _log_fn(config)
    # get dataloader
    _, _, _, test_loader, _ = get_dataset(config['data'], None, val_mode=True, train_split=config['data']['dataset'])
    # train for each random seed
    results = []
    for seed in config['training']['random_seed']:
        ckp_path = os.path.join(this_run_dir, f"randomseed{seed}", "model", "*.pt")
        best_model_list = glob(ckp_path)
        if len(best_model_list) == 0:
            continue
        # start test
        model, device = _setup(config, seed, test_loader, best_model_list)
        with torch.no_grad():
            v_mse, v_rmse, v_mae, v_pr, v_ci, v_rm2, v_aupr = val_on_metrics(model, test_loader, device, config)
        msg = "mse:%.4f, rmse:%.4f, mae:%.4f, pr:%.4f, ci:%.4f, rm2:%.4f, aupr:%.4f," \
                    % (v_mse, v_rmse, v_mae, v_pr, v_ci, v_rm2, v_aupr)
        results.append(np.array([v_mse, v_rmse, v_mae, v_pr, v_ci, v_rm2, v_aupr]))
        print(msg)
        # append the msg at the last line in the log file
        with open(os.path.join(this_run_dir, f"randomseed{seed}", "log", "train", "test.log"), "a") as f:
            f.write(msg + "\n")
            
    # calcu mean and std
    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    msg = "mean_mse:%.4f (%.4f), mean_rmse:%.4f (%.4f), mean_mae:%.4f (%.4f), mean_pr:%.4f (%.4f), mean_ci:%.4f (%.4f), mean_rm2:%.4f (%.4f), mean_aupr:%.4f (%.4f)" \
                % (mean[0], std[0], mean[1], std[1], mean[2], std[2], mean[3], std[3], mean[4], std[4], mean[5], std[5], mean[6], std[6])
    print(msg)
    # append the msg at the last line in the log file
    with open(os.path.join(this_run_dir, "test.log"), "a") as f:
        f.write(msg + "\n")
    # free cuda mem
    if 'cuda' in config['training']['device']:
        torch.cuda.empty_cache()


def approximate_ci_loss(pred, target, scale=1.0):
    """
    越小越好，相当于最大化 CI
    """
    i, j = torch.triu_indices(pred.size(0), pred.size(0), offset=1)

    pred_diff = pred[i] - pred[j]
    target_diff = target[i] - target[j]

    # 去掉相等标签的对
    mask = target_diff != 0
    pred_diff = pred_diff[mask]
    target_diff = target_diff[mask]

    score = pred_diff * target_diff * scale
    ci_approx = torch.sigmoid(score)

    # 1 - CI 作为 loss
    return (1.0 - ci_approx).mean()


def pairwise_ranking_loss(pred, target, margin=0.1):
    """
    pred: [batch_size]
    target: [batch_size]
    """
    # 构造所有样本对 (i > j)
    i, j = torch.triu_indices(pred.size(0), pred.size(0), offset=1)

    pred_diff = pred[i] - pred[j]
    target_diff = target[i] - target[j]

    # 只在真实值不相等时计算
    mask = target_diff != 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    sign = torch.sign(target_diff[mask])
    loss = torch.clamp(margin - sign * pred_diff[mask], min=0.0)

    return loss.mean()
    

if __name__ == '__main__':
    # command launch
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument('--dataset', type=str, default='bindingdb', choices=['bindingdb', 'davis', 'kiba'])
    args_paser.add_argument('--test', type=str, nargs='+', default=None)
    args_paser.add_argument('--extra-loss-fn', type=str, nargs='+', default=None, choices=['approximate_ci_loss', 'pairwise_ranking_loss'])
    args_paser.add_argument('--extra-loss-fn-weight', type=float, nargs='+', default=None)
    args = make_args(args_paser)
    args.config['name'] = args.config['name'].replace('s1_', f's1DeepDTAGen_')
    args.config['data']['dataset'] = args.dataset
    args.config['training']['extra_loss_fn'] = args.extra_loss_fn
    args.config['training']['extra_loss_fn_weight'] = args.extra_loss_fn_weight
    
    if args.extra_loss_fn is not None and args.extra_loss_fn_weight is not None and len(args.extra_loss_fn) == len(args.extra_loss_fn_weight):
        extra_loss = [(globals()[fn], w) for fn, w in zip(args.extra_loss_fn, args.extra_loss_fn_weight)]
    else:
        extra_loss = None
    
    if args.test is not None:
        for root in args.test:
            test_from_root(root)
    elif args.config['training']['resume_dir'] is None:
        run_one_config(args.config, train_split=args.dataset, _get_dataset=get_dataset,
                       extra_loss=extra_loss)
    else:
        args.config['training']['random_seed'] = args.config['training']['resume_seeds']
        run_one_config(args.config, args.config['training']['resume_dir'], train_split=args.dataset, _get_dataset=get_dataset,
                       extra_loss=extra_loss)
