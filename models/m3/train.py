
import argparse
import os
import random
import sys
from glob import glob

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from mbapy.base import get_fmt_time, put_log, split_list
from mbapy.dl_torch.optim import LrScheduler, str2scheduleF
from mbapy.dl_torch.utils import (AverageMeter, ProgressMeter,
                                  init_model_parameter, set_random_seed)
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config.config_dict import *
from log.train_logger_v1 import *
from models._utils.arg import *
from models._utils.meter import FakeSummaryWriter, Meters, MeterType
from models._utils.scheduler import *
from models.m3.data_loader import (DANNDataset, get_data_loader,
                                   load_lig_data_by_name,
                                   load_rec_data_by_name)
from models.m3.model import (Arch1, Arch14, get_data_shape_from_dataset)
from models.s1.train import get_loss
from utils import BestMeter, load_model_dict, save_model_dict

warnings.filterwarnings('ignore')

# %%
def val(model: Arch1, dataloader: DataLoader, device: str) -> tuple[float, float, float]:
    model.eval()
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
        return 100, 100, 100
    pr: float = pearsonr(pred, label)[0]
    loss: float = mean_squared_error(label, pred)
    rmse: float = np.sqrt(loss)

    return loss, rmse, pr

def get_dataset(config: dict, logger, val_mode: bool = False, debug: bool = False, shuffle: bool = True, train_split: dict[str, list[str]] = None):
    # replace ~ in paths
    for n in ['data_root', 'bin_root']:
        config[n] = os.path.expanduser(config[n])
    # load bin data
    lig_data = load_lig_data_by_name(config['lig_type'], logger)
    prot_data = load_rec_data_by_name(config['prot_type'], logger)
    # load datasets
    df = pd.read_csv('./data/train.csv')
    if train_split is not None:
        df = pd.read_csv('./data/clean_split/affinity.csv')
        df = df[df['pdbid'].isin(train_split['train'])]
    if debug:
        df = df.sample(frac=0.05, random_state=42)
    train_loader = get_data_loader(lig_data, prot_data, df,
                                    config['prot_transform'], config['prot_max_len'],
                                    config['lig_seq_reduce'], config['load_ratio'], config['load_order'],
                                    config['device'], config['batch_size'], shuffle, 1, logger)
    df = pd.read_csv('./data/valid.csv')
    if train_split is not None:
        df = pd.read_csv('./data/clean_split/affinity.csv')
        df = df[df['pdbid'].isin(train_split['validation'])]
    valid_loader = get_data_loader(lig_data, prot_data, df,
                                   config['prot_transform'], config['prot_max_len'],
                                   config['lig_seq_reduce'], config['load_ratio'], config['load_order'],
                                   config['device'], config['batch_size'], False, 1, logger)
    test2013_loader = get_data_loader(lig_data, prot_data, pd.read_csv('./data/test2013.csv'),
                                      config['prot_transform'], config['prot_max_len'],
                                      config['lig_seq_reduce'], 0.1, 'front',
                                      config['device'], config['batch_size'], False, 1, logger)
    test2016_loader = get_data_loader(lig_data, prot_data, pd.read_csv('./data/test2016.csv'),
                                      config['prot_transform'], config['prot_max_len'],
                                      config['lig_seq_reduce'], 0.1, 'front',
                                      config['device'], config['batch_size'], False, 1, logger)
    test2019_loader = get_data_loader(lig_data, prot_data, pd.read_csv('./data/test2019.csv'),
                                      config['prot_transform'], config['prot_max_len'],
                                      config['lig_seq_reduce'], 0.1, 'front',
                                      config['device'], config['batch_size'], False, 1, logger)
    if config['use_prot_std']:
        train_loader.dataset.calcu_mean_std()
        train_loader.dataset.apply_mean_std()
        valid_loader.dataset.apply_mean_std(train_loader.dataset.mean, train_loader.dataset.std)
        test2013_loader.dataset.apply_mean_std(train_loader.dataset.mean, train_loader.dataset.std)
        test2016_loader.dataset.apply_mean_std(train_loader.dataset.mean, train_loader.dataset.std)
        test2019_loader.dataset.apply_mean_std(train_loader.dataset.mean, train_loader.dataset.std)
        log_fn = logger.info if logger else put_log
        log_fn(f'protein feature normalized by mean and std of training set (mean={train_loader.dataset.mean}, std={train_loader.dataset.std}).')
    return train_loader, valid_loader, test2013_loader, test2016_loader, test2019_loader


def merge_dataloader(*args: DataLoader) -> DataLoader:
    data, umol_v4_pool = [], {}
    for loader in args:
        data += loader.dataset.data
        umol_v4_pool.update(loader.dataset.umol_v4_pool)
    data_loader = DataLoader(DANNDataset(data, umol_v4_pool, args[0].dataset.dyn_padding),
                             batch_size=args[0].batch_size, shuffle=True, num_workers=1)
    return data_loader


def get_model(config: dict, data_shapes: dict[str, torch.Size], logger):
    if logger is None:
        log_fn = print
    else:
        log_fn = logger.info
        
    _str2arch = {'arch1': Arch1, 'arch14': Arch14}
    
    if config['arch'] in {'arch1', 'arch14'}:
        model = _str2arch[config['arch']](data_shapes, config['hidden_feat_size'], config['RMSNorm'], config['use_rope'], False, config['feat_mini_mhsa'],
                      config.get('softmax_partition', False), config['layer_num'],
                      config['pred'], config['shared_exp'], config['gatter'], config['router_noise'], config['router_act'],
                      config['prot_scale'], config['n_head'], config['drop_out'],
                      lig_token_moe=config.get('lig_token_moe', False), lig_pred=config['lig_pred'], prot_pred=config['prot_pred'],
                      hydraformer=config.get('hydraformer', False), moe_ffn=config.get('moe_ffn', False), use_method_id=config.get('use_method_id', []),
                      low_mem_transformer=config.get('low_mem_transformer', False), gated_sdpa=config.get('gated_sdpa', False),
                      lig_emb_dim=config.get('lig_emb_dim', 384))
    else:
        raise ValueError(f"Unknown architecture: {config['arch']}")
    log_fn(str(model))
    return model

def run_one_config(cfg: str|dict[str, bool|str|dict[str, float]], this_run_wd: Path = None, train_split: dict[str, list[str]] = None):
    if isinstance(cfg, str):
        config = Config(cfg).get_config()
    else:
        config = cfg
    cfg_name = config['name']
    if not config['debug'] and this_run_wd is None:
        this_run_wd = Path(config['training']['save_dir']) / f'{get_fmt_time("%Y%m%d-%H%M%S")}-{cfg_name}'
        os.makedirs(this_run_wd, exist_ok=True)
        import fnmatch
        def ignore_static_files(_, files):
            return [f for f in files if any(fnmatch.fnmatch(f, fmt) for fmt in ['*.pt', '*.joblib'])]
        shutil.copytree('models', this_run_wd / f'src', ignore=ignore_static_files)
    # train for each random seed
    for seed in config['training']['random_seed']:
        # set random seed
        config['training']['now_random_seed'] = seed
        set_random_seed(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        # logger
        if not config['debug']:
            logger = TrainLogger(config, cfg_name, this_run_wd, create=True)
            _log_fn = logger.info
            if config['training']['no_tensorboard']:
                logger.info('tensorboard disabled because no_tensorboard option is True.')
                writer = FakeSummaryWriter()
            else:
                writer = SummaryWriter(comment=f"{cfg_name}_seed{seed}")
        else:
            logger = None
            _log_fn = print
        _log_fn(f'command line input: {" ".join(sys.argv)}') # log command line input
        _log_fn(__file__)
        _log_fn(str(config))
        # get dataloader
        train_loader, valid_loader, test2013_loader, test2016_loader, test2019_loader = get_dataset(config['data'], logger,
                                                                                                    debug=config['debug'], train_split=train_split)
        # model
        device = torch.device(config['training']['device'])
        data_shapes = get_data_shape_from_dataset(train_loader.dataset)
        model =  get_model(config['model'], data_shapes, logger).to(device)
        if not (config['training']['no_compile']):
            model: nn.Module = torch.compile(model)
        # optimizer
        lr, wd = config['training']['lr'], config['training']['weight_decay']
        if config['training']['optimizer'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif config['training']['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
        # scheduler
        min_lr = 0.5 * lr if config['training']['scheduler'] in str2scheduleF else 0.01 * lr
        model_scheduler = get_model_scheduler(optimizer, scheduler_type=config['training']['scheduler'], initial_value=lr,
                                            total_epochs=config['training']['epochs'], scheduler_T=config['training']['scheduler_T'],
                                            min_value=min_lr, patience=10, max_value=lr)
        # criterion
        criterion = nn.MSELoss()
        criterion_moe = nn.CrossEntropyLoss()
        criterion_cl = nn.CrossEntropyLoss()
        # torch.autograd.set_detect_anomaly(True)
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # meters
        meters = Meters()
        meters.add_meters(MeterType('train', ':.4f'), MeterType('valid', ':.4f'),
                          MeterType('test2013', ':.4f'), MeterType('test2016', ':.4f'),
                          MeterType('best_2016', ':.4f', 'min'),
                          MeterType('MoE_act', ':.4f'), MeterType('MoE_loss', ':.4f'),
                          *[MeterType(f'G-{n}', ':.4f') for n in model.feat_encoders])
        meters.make_progress(1, mp = logger.info if not config['debug'] else print)
        best_model_list = []
        # scheduler manager
        scheduler_manager = SchedulerManager({
            'model': model_scheduler,
        })
        # start training
        valid_rmse, test2016_rmse = 0, 0
        # is_freeze_disabled = False
        model.train()
        for epoch in range(config['training']['epochs']):

            # train for one epoch
            model.train()
            for b_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}', leave=False):
                _ = data.pop('idx')
                mid = data.pop('mid').to(device)            
                label = data.pop('pKa').to(device)
                data = {k: [v[0].to(device), v[1].to(device)] for k, v in data.items()}
                
                # add input noise
                if config['model']['noise_rate'] >= 0: # make sure even noise_rate is 0, the data order is same because the random is also called
                    for k, v in data.items():
                        if k not in {'token', 'PepDoRA-token'}:
                            data[k][0] += torch.randn_like(data[k][0]) * config['model']['noise_rate']
                        
                # add label noise
                if config['model']['label_noise_rate'] >= 0: # make sure even label_noise_rate is 0, the data order is same because the random is also called
                    label = label + torch.randn_like(label) * config['model']['label_noise_rate']
                    
                optimizer.zero_grad()
                
                # scheduler step
                scheduler_manager.step_in_epoch('lr', epoch)
                if config['debug']:
                    print(f'INPUT: ', ', '.join([f'{k} range {v[0].min():7.0f}~{v[0].max():7.0f}' for k, v in data.items()]))
                    data = model.feat_encode(data)
                    print(f'feat_encode: ', ', '.join([f'{k} range {v[0].min():7.0f}~{v[0].max():7.0f}' for k, v in data.items()]))
                    print(f'feat_encode: ', ', '.join([f'{k} shape {v[0].shape}' for k, v in data.items()]))
                    if hasattr(model, 'scale'):
                        print(f'scale: {", ".join([f"{k}: {v:.4f}" for k, v in model.scale.items()])}')
                    x = model.feat_forward(data, mid=mid)  # [N, 1, D]
                    logits = model.predict_from_feat(x)
                    loss = get_loss(model, logits.reshape(-1), label.reshape(-1), criterion, criterion_moe, config)
                else:
                    logits = model(data, mid=mid, noise_rate=config['model']['noise_rate'],
                                   shuffle=config['training']['modal_shuffle'])
                    loss = get_loss(model, logits.reshape(-1), label.reshape(-1), criterion, criterion_moe, config)
                    
                loss.backward()
                # record modal encoders gradients
                model.record_modal_enc_grad()
                for n in model.feat_encoders:
                    meters.update(f'G-{n}', model.modal_enc_grad_record[n], label.size(0))
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                meters.update('train', loss.detach().cpu().item(), label.size(0))
                if model.is_MoE:
                    meters.update('MoE_loss', model.MoE_balance_loss.item(), label.size(0))
                if model.pred_is_MoE:
                    meters.update('MoE_act', model.predictor.expert_activation.std() / model.predictor.expert_activation.mean(), label.size(0))
                
            # log loss
            epoch_rmse = np.sqrt(meters.get('train').avg)
            moe_banlance_loss = model.MoE_balance_loss if model.is_MoE else 0
            if config['debug']:
                meters.display(epoch)
                if model.is_MoE:
                    if config['model']['router_noise']:
                        print(f'MoE balance_loss: {model.MoE_balance_loss.item():.4f}')
                    print(f'MoE expert_activation: {model.predictor.expert_activation.cpu().numpy().astype(int).tolist()}')
                    print(f'MoE gate_density: {[f"{i.cpu().item():.4f}" for i in model.predictor.gate_density]}')
                print(torch.cuda.memory_summary())
            else:
                # log loss
                writer.add_scalar('train_rmse', epoch_rmse, epoch)
                writer.add_scalar('MoE_act', meters.get('MoE_act').avg, epoch)
                writer.add_scalar('MoE_loss', moe_banlance_loss, epoch)
                # validating
                _, valid_rmse, valid_pr = val(model, valid_loader, device)
                _, test2013_rmse, test2013_pr = val(model, test2013_loader, device)
                _, test2016_rmse, test2016_pr = val(model, test2016_loader, device)
                writer.add_scalar('valid_rmse', valid_rmse, epoch)
                writer.add_scalar('test2013_rmse', test2013_rmse, epoch)
                writer.add_scalar('test2016_rmse', test2016_rmse, epoch)
                meters.update('valid', valid_rmse, 1)
                meters.update('test2013', test2013_rmse, 1)
                meters.update('test2016', test2016_rmse, 1)
                # save best model
                if test2016_rmse < meters.get('best_2016').get_best():
                    meters.update('best_2016', test2016_rmse, 1)
                    msg = "epoch:%4d, train_rmse:%7.4f, valid_rmse:%7.4f, test2013_rmse:%7.4f, test2016_rmse:%7.4f" \
                        % (epoch, epoch_rmse, valid_rmse, test2013_rmse, test2016_rmse)
                    model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                    best_model_list.append(model_path)
                    save_model_dict(model, logger.get_model_dir(), msg)
                    print(f'Now is No.{meters.get("best_2016").sum_best} best model')
                else:
                    no_improve_count = meters.get("best_2016").counter() # easy understand
                    # make sure train more than leat epochs
                    if no_improve_count > config['training']['early_stop_epoch'] and epoch > config['training']['least_epochs']:
                        best_mse = meters.get("best_2016").get_best()
                        msg = "best_rmse: %.4f" % best_mse
                        logger.info(f"early stop in epoch {epoch}")
                        logger.info(msg)
                        break
                meters.display(epoch)
                
            # scheduler step
            scheduler_manager.step_after_epoch('lr', epoch, valid_rmse)
                    
            meters_need_reset = list(meters.meters.keys())
            meters_need_reset.remove('best_2016')
            meters.resets(*meters_need_reset)
            if model.pred_is_MoE:
                model.predictor.reset_expert_activation()
                
            # check whether exists `quit.txt` in model dir
            if logger and os.path.exists(os.path.join(logger.get_model_dir(), 'quit.txt')):
                break
            
        if not config['debug']:
            # close this run logger
            writer.close()
            # final testing
            load_model_dict(model, best_model_list[-1])
            _, valid_rmse, valid_pr = val(model, valid_loader, device)
            _, test2013_rmse, test2013_pr = val(model, test2013_loader, device)
            _, test2016_rmse, test2016_pr = val(model, test2016_loader, device)
            _, test2019_rmse, test2019_pr = val(model, test2019_loader, device)
            msg = "valid_rmse:%.4f, valid_pr:%.4f, test2013_rmse:%.4f, test2013_pr:%.4f, test2016_rmse:%.4f, test2016_pr:%.4f, test2019_rmse:%.4f, test2019_pr:%.4f," \
                        % (valid_rmse, valid_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, test2019_rmse, test2019_pr)
            logger.info(msg)
            # delete non-best model
            for path in best_model_list[:-1]:
                os.remove(path)
            # free cuda mem
            if 'cuda' in config['training']['device']:
                torch.cuda.empty_cache()
        # free dataset mem
        try:
            del train_loader.dataset, valid_loader.dataset, test2013_loader.dataset, test2016_loader.dataset, test2019_loader.dataset
            del train_loader, valid_loader, test2013_loader, test2016_loader, test2019_loader
        except:
            pass


def make_args():
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument("-c", "--config", type=str, default=None,
                            help="config file name in task/configs dir, default is %(default)s")
    args_paser.add_argument("-a", type=str, choices=['arch1', 'arch11', 'arch12', 'arch13', 'arch14', 'arch2', 'arch21', 'arch22'], default='arch1',
                            help="architecture, default is %(default)s")
    args_paser.add_argument("--prot-n", type=int, default=8,
                            help="number of prot token, default is %(default)s")
    args_paser.add_argument("--modals-n-token", type=str, default=None, nargs='+',
                            help="number of token for each modal, such as GeminiMol 1, default is %(default)s")
    args_paser.add_argument("-l", type=int, default=2,
                            help="number of layer, default is %(default)s")
    args_paser.add_argument("-d", type=int, default=512,
                            help="number of hidden, default is %(default)s")
    args_paser.add_argument("--mox-topk", type=int, default=1,
                            help="topk for MoMixture, default is %(default)s")
    args_paser.add_argument("--mox-gatter", type=str, default='weighted', choices=['mean', 'sum', 'weighted'],
                            help="gatter for MoMixture, default is %(default)s")
    args_paser.add_argument("--lig-token-moe", action="store_true", default=False,
                            help="use token moe for lig, default is %(default)s")
    args_paser.add_argument("--lig-emb-dim", type=int, default=384,
                            help="lig embedding dimension, default is %(default)s")
    add_pred_args(args_paser)
    add_transformer_args(args_paser)
    add_noise_args(args_paser)
    args_paser.add_argument("--modal-drop-num", type=int, default=0,
                            help="modal drop number, default is %(default)s")
    args_paser.add_argument("--modal-del-num", type=int, default=0,
                            help="modal del number, default is %(default)s")
    args_paser.add_argument("--modal-shuffle", action="store_true", default=False,
                            help="modal shuffle, default is %(default)s")
    add_train_control_args(args_paser)
    args_paser.add_argument("-lig", type=str, default=['PepDoRA'], nargs='+', choices=['PepDoRA', 'ChemBERTa_10M', 'ChemBERTa_100M_MLM',
                                                                                       'ChemBERTa_77M_MTR', 'ChemBERTa_77M_MLM', 'MolFormer',
                                                                                       'SELFormer', 'MolAI', 'GeminiMol',
                                                                                       'UniMol-v3', 'UniMol-v3-570m', 'UniMol-v4', 'PepDoRA-token', 'token',
                                                                                       'MaskMol_224', 'rdkit_vit_224', 'rdkit_vit_512', 'ImageMol',
                                                                                       '3Dimg', '3Dimg_vit', ''],
                            help="lig type, default is %(default)s")
    args_paser.add_argument("-r", type=float, default=0.01,
                            help="dataset load ratio, default is %(default)s")
    args_paser.add_argument("-lo", type=str, default='front', choices=['front', 'back', 'random', 'more'],
                            help="dataset load order, default is %(default)s")
    args_paser.add_argument("-prot", type=str, default=['esm3-open_split'], nargs='+', choices=['esm2-mean_each_mean', 'esm2-3B',
                                                                                                'esm2-3B-36_mean_each_mean', 'esm2-3B-30_mean_each_mean',
                                                                                                'esm3-open_split',
                                                                                                'ProSST-2048', 'UniMol-v3', 'SaProt', ''],
                            help="prot type, default is %(default)s")
    args_paser.add_argument("--prot-std", action="store_true", default=False,
                            help="use mean, std to normalize prot, default is %(default)s")
    args_paser.add_argument("--prot-scale", action='store_true', default=False,
                            help="scale prot feature after loaded, default is %(default)s")
    args_paser.add_argument("--prot-transform", type=str, nargs='+', default=None,
                            help="transform prot feature after loaded, default is %(default)s")
    args_paser.add_argument("--prot-max-len", type=int, default=128,
                            help="max prot len, default is %(default)s")
    args_paser.add_argument("--lig-seq-reduce", type=str, default=None, choices=['sum', 'mean', 'zero'],
                            help="reduce lig data along seq dim, default is %(default)s")
    
    args = args_paser.parse_args()
    
    if args.config is None:
        args.config = Config('s1_EHIGN_base').get_config()
        
        # name
        model_data = f'{",".join(args.lig) if isinstance(args.lig, list) else args.lig}{"std" if args.prot_std else ""}'
        model_detail = f'{args.l}h{args.d}do{args.do}ns{args.ns}lns{args.lns}bs{args.bs}'
        if args.a in {'arch1', 'arch11', 'arch12', 'arch13', 'arch14'}:
            args.config['name'] = f'm3_{args.a}{",".join(args.pred)}|{args.gatter}_{args.optim}{args.lr}_{model_data}_{model_detail}'
        elif args.a == 'arch2':
            args.config['name'] = f'm3_{args.a}{",".join(args.pred)}PN{args.prot_n}|{args.gatter}_{args.optim}{args.lr}_{model_data}_{model_detail}'
        elif args.a in {'arch21', 'arch22'}:
            args.config['name'] = f'm3_{args.a}{",".join(args.pred)}|{args.gatter}_{args.optim}{args.lr}_{model_data}_{model_detail}'
            
        # data
        args.config['data']['lig_type'] = args.lig
        args.config['data']['load_ratio'] = args.r
        args.config['data']['load_order'] = args.lo
        args.config['data']['prot_type'] = args.prot
        args.config['data']['use_prot_std'] = args.prot_std
        args.config['data']['prot_transform'] = args.prot_transform
        args.config['data']['prot_max_len'] = args.prot_max_len
        args.config['data']['lig_seq_reduce'] = args.lig_seq_reduce
        ## special case
        if '' in args.prot:
            args.config['data']['prot_type'].remove('')
        if '' in args.lig:
            args.config['data']['lig_type'].remove('')
        
        # arch
        args.config['model']['arch'] = args.a
        args.config['model']['prot_n'] = args.prot_n
        args.config['model']['modals_n_token'] = {}
        if args.modals_n_token is not None:
            args.config['model']['modals_n_token'] = {k: int(v) for k, v in split_list(args.modals_n_token, 2)}
        args.config['model']['layer_num'] = args.l
        args.config['model']['hidden_feat_size'] = args.d
        args.config['model']['mox_topk'] = args.mox_topk
        args.config['model']['mox_gatter'] = args.mox_gatter   
        args.config['model']['lig_emb_dim'] = args.lig_emb_dim  
        args.config['model']['lig_token_moe'] = args.lig_token_moe   
        args.config['model']['use_rope'] = not args.no_rope 
        args.config['model']['hydraformer'] = args.hydraformer
        args.config['model']['prot_scale'] = args.prot_scale
        
        # unify
        assign_pred_args(args)
        assign_transformer_args(args)
        assign_noise_args(args)
        assign_train_control_args(args)
    
    return args


if __name__ == '__main__':
    # command launch
    args = make_args()
    run_one_config(args.config)
