import argparse
import os
import warnings
from glob import glob
from typing import Optional

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
from mbapy.dl_torch.utils import set_random_seed
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from config.config_dict import *
from log.train_logger_v1 import *
from models._utils.arg import *
from models.m3.model import Arch1
from models.m3.train import get_data_shape_from_dataset, get_dataset, get_model
from models.s1.test import concordance_index

warnings.filterwarnings('ignore')


def val(model: Arch1, dataloader: Optional[DataLoader], device: str) -> tuple[float, float, float, float, float]:
    if dataloader is None:
        return -1, -1, -1, -1, -1
    model.eval()
    if model.use_DGL in {'re_weight', 're_pred'}:
        kwgs = {'DGL_forward_method': model.use_DGL, 'hybrid': False, 'low_mem': True, 'eval_mode': True}
    else:
        kwgs = {}
    pred_list = []
    label_list = []
    for data in dataloader:
        _ = data.pop('idx')
        mid = data.pop('mid').to(device)
        label = data.pop('pKa').to(device)
        data = {k: [v[0].to(device), v[1].to(device)] for k, v in data.items()}

        with torch.no_grad():
            pred = model(data, mid=mid, **kwgs)
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0).reshape(-1)
    label = np.concatenate(label_list, axis=0).reshape(-1)
    # check whether pred contains nan
    if np.isnan(pred).any() or np.isnan(label).any():
        return 100, 100, 100, 100, 100
    pr: float = pearsonr(pred, label)[0]
    loss: float = mean_squared_error(label, pred)
    rmse: float = np.sqrt(loss)
    mae: float = np.mean(np.abs(pred - label))
    ci: float = concordance_index(label, pred)
    return loss, rmse, mae, pr, ci


def run_one_config(cfg: str, this_run_dir: str):
    if isinstance(cfg, str):
        config = Config(cfg).get_config()
    else:
        config = cfg
    # get dataloader
    _, valid_loader, test2013_loader, test2016_loader, test2019_loader = get_dataset(config['data'], None, val_mode=True)
    # train for each random seed
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
        logger = None
        _log_fn = print
        _log_fn(__file__)
        _log_fn(str(config))
        # model
        device = torch.device(config['training']['device'])
        data_shapes = get_data_shape_from_dataset(valid_loader.dataset)
        model =  get_model(config['model'], data_shapes, logger).to(device)
        if not (config['training']['no_compile']):
            model: nn.Module = torch.compile(model)
        # final testing
        ckp = torch.load(best_model_list[-1])
        ckp = {k.replace('_orig_mod.', ''): v for k, v in ckp.items()}
        model.load_state_dict(ckp)
        model = model.to(device)
        model.eval()
        _, valid_rmse, valid_mae, valid_pr, valid_ci = val(model, valid_loader, device)
        _, test2013_rmse, test2013_mae, test2013_pr, test2013_ci = val(model, test2013_loader, device)
        _, test2016_rmse, test2016_mae, test2016_pr, test2016_ci = val(model, test2016_loader, device)
        _, test2019_rmse, test2019_mae, test2019_pr, test2019_ci = val(model, test2019_loader, device)
        msg = "valid_rmse:%.4f, valid_mae:%.4f, valid_pr:%.4f, valid_ci:%.4f, test2013_rmse:%.4f, test2013_mae:%.4f, test2013_pr:%.4f, test2013_ci:%.4f, test2016_rmse:%.4f, test2016_mae:%.4f, test2016_pr:%.4f, test2016_ci:%.4f, test2019_rmse:%.4f, test2019_mae:%.4f, test2019_pr:%.4f, test2019_ci:%.4f," \
                    % (valid_rmse, valid_mae, valid_pr, valid_ci, test2013_rmse, test2013_mae, test2013_pr, test2013_ci, test2016_rmse, test2016_mae, test2016_pr, test2016_ci, test2019_rmse, test2019_mae, test2019_pr, test2019_ci)
        _log_fn(msg)
        # append the msg at the last line in the log file
        with open(os.path.join(this_run_dir, f"randomseed{seed}", "log", "train", "Train.log"), "a") as f:
            f.write(msg + "\n")


if __name__ == '__main__':
    # command launch
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument("-p", "--path", type=str, default=None,
                            help="checkpoint path, default is %(default)s")
    
    args = args_paser.parse_args()
    
    path = os.path.join(f'./checkpoints/{args.path}/randomseed0/log/train/*.json')
    path = glob(path)[0]
    config = opts_file(path, way='json')

    run_one_config(config, f'./checkpoints/{args.path}')
