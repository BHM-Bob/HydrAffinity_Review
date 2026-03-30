
import argparse
import os
import sys

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from mbapy.base import get_fmt_time, put_log
from mbapy.dl_torch.optim import str2scheduleF
from mbapy.dl_torch.utils import init_model_parameter, set_random_seed
from mbapy.plot import save_show
from mbapy.web import TaskPool
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config.config_dict import *
from log.train_logger_v1 import *
from models._utils.arg import *
from models._utils.meter import FakeSummaryWriter, Meters, MeterType
from models._utils.scheduler import *
from models.s1.data_loader import DANNDataset, get_data_loader
from models.s1.model import (Arch1, Arch2, Arch3, Arch4, Arch11, Arch21,
                             Arch22, Arch23, Arch24, Arch31, Arch41, Arch42,
                             Arch43, Arch44, DANNWarpper)
from utils import (get_model_state_dict_copy, load_model_dict,
                   save_state_dict_in_thread)

warnings.filterwarnings('ignore')


# %%
def val(model: Arch1, dataloader: DataLoader, device: str, config: dict) -> tuple[float, float, float]:
    model.eval()
    is_BiLevel = False
    if (isinstance(config['model']['pred'], str) and 'BiLevel' in config['model']['pred']) or \
       (isinstance(config['model']['pred'], list) and any('BiLevel' in n for n in config['model']['pred'])):
        is_BiLevel = True
    pred_list = []
    label_list = []
    for data in tqdm(dataloader, desc='Validating', leave=False):
        data = [i.to(device) for i in data]
        inputs, label = data[1:-1], data[-1]

        with torch.no_grad():
            pred = model(*inputs)
            if is_BiLevel:
                pred = pred.reshape(label.shape[0], -1)
                pred = pred[:, :-1].argmax(dim=-1) + pred[:, -1]
            pred_list.append(pred.detach().cpu().view(-1).numpy())
            label_list.append(label.detach().cpu().view(-1).numpy())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    # check whether pred contains nan
    if np.isnan(pred).any() or np.isnan(label).any():
        return 100, 100, 100
    pr: float = pearsonr(pred, label)[0]
    loss: float = mean_squared_error(label, pred)
    rmse: float = np.sqrt(loss)

    return loss, rmse, pr

def get_dataset(config: dict, logger, val_mode: bool = False, debug: bool = False, train_split: dict[str, list[str]] = None):
    # replace ~ in paths
    for n in ['data_root', 'bin_root']:
        config[n] = config[n].replace('/home/SERVER/', '~/')
        config[n] = os.path.expanduser(config[n])
    # load bin data or make UniMol-v3 data path
    if config['lig_type'] == 'UniMol-v3':
        lig_data = os.path.join(config['bin_root'], 'UniMol-v3')
    elif config['lig_type'] == 'UniMol-v3-570m':
        lig_data = os.path.join(config['bin_root'], 'UniMol-v3-570m')
    elif config['lig_type'] == 'UniMol-v4-gen':
        lig_data = os.path.join(config['bin_root'], 'UniMol-v4-gen')
    elif config['lig_type'] == 'UniMol-v4':
        lig_data = os.path.join(config['bin_root'], 'UniMol-v4')
    elif config['lig_type'] == 'token':
        lig_data = torch.load('../EHIGN_dataset/smiles_tokenized/smiles_indices.pt', map_location='cpu', weights_only=False)
    elif config['lig_type'] == 'PepDoRA-token':
        lig_data = torch.load('../EHIGN_dataset/SMILES_PepDoRA-token.pt', map_location='cpu', weights_only=False)
    else:
        lig_data_path = os.path.join(config['bin_root'], f'SMILES_{config["lig_type"]}.pt')
        lig_data = torch.load(lig_data_path, map_location='cpu', weights_only=False)
    if config['prot_type'] == 'UniMol-v3':
        prot_data = os.path.join(config['bin_root'], 'UniMol-v3')
    elif config['prot_type'] == 'SaProt':
        prot_data_path = os.path.join(config['bin_root'], f'protein_SaProt_650M_AF2_mean_each_mean.pt')
        prot_data = torch.load(prot_data_path, map_location='cpu', weights_only=False)
    else:
        prot_data_path = os.path.join(config['bin_root'], f'protein_{config["prot_type"]}.pt')
        prot_data = torch.load(prot_data_path, map_location='cpu', weights_only=False)
    # load datasets
    num_workers = 0
    load_ratio, load_order = config.get('load_ratio', 1.0), config.get('load_order', 'front')
    cat_v4_n = config.get('cat_v4_n', None)
    testset_lr = 0.1 if cat_v4_n is None else load_ratio
    if logger is not None:
        logger.info(f'num_workers: {num_workers}')
    else:
        print(f'num_workers: {num_workers}')
    if val_mode:
        train_loader = None
    else:
        if train_split is not None:
            df = pd.read_csv('./data/clean_split/affinity.csv')
            df = df[df['pdbid'].isin(train_split['train'])]
        else:
            df = pd.read_csv('./data/train.csv')
        if debug:
            df = df.sample(frac=0.1, random_state=42)
        # torch compile use dynamic shape, drop_last will make shape consistent in whole epoch
        train_loader = get_data_loader(lig_data, prot_data, df, config['prot_transform'], config.get('lig_seq_reduce', None),
                                        load_ratio, load_order, cat_v4_n, config['device'], config['batch_size'], True, num_workers, logger, drop_last=False)
    if train_split is not None:
        df = pd.read_csv('./data/clean_split/affinity.csv')
        df = df[df['pdbid'].isin(train_split['validation'])]
    else:
        df = pd.read_csv('./data/valid.csv')
    valid_loader = get_data_loader(lig_data, prot_data, df,
                                   config['prot_transform'], config.get('lig_seq_reduce', None),
                                   load_ratio, load_order, cat_v4_n, config['device'], config['batch_size'], False, num_workers, logger)
    test2013_loader = get_data_loader(lig_data, prot_data, pd.read_csv('./data/test2013.csv'),
                                      config['prot_transform'], config.get('lig_seq_reduce', None),
                                      testset_lr, 'front', cat_v4_n, config['device'], config['batch_size'], False, num_workers, logger)
    test2016_loader = get_data_loader(lig_data, prot_data, pd.read_csv('./data/test2016.csv'),
                                      config['prot_transform'], config.get('lig_seq_reduce', None),
                                      testset_lr, 'front', cat_v4_n, config['device'], config['batch_size'], False, num_workers, logger)
    test2019_loader = get_data_loader(lig_data, prot_data, pd.read_csv('./data/test2019.csv'),
                                      config['prot_transform'], config.get('lig_seq_reduce', None),
                                      testset_lr, 'front', cat_v4_n, config['device'], config['batch_size'], False, num_workers, logger)
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
    data = []
    for loader in args:
        data += loader.dataset.data
    data_loader = DataLoader(DANNDataset(data), batch_size=args[0].batch_size, shuffle=True, num_workers=1)
    return data_loader


def get_model(config: dict, lig_dim: int, prot_dim: int, logger):
    if logger is None:
        log_fn = print
    else:
        log_fn = logger.info
        
    _str2arch = {'arch1': Arch1, 'arch11': Arch11,
                 'arch2': Arch2, 'arch21': Arch21, 'arch22': Arch22, 'arch23': Arch23, 'arch24': Arch24,
                 'arch3': Arch3, 'arch31': Arch31,
                 'arch4': Arch4, 'arch41': Arch41, 'arch42': Arch42, 'arch43': Arch43, 'arch44': Arch44}
    ext_kwgs = {}
    # assign arch
    if config['arch'] == 'arch1':
        model = Arch1(lig_dim, prot_dim, config['hidden_feat_size'], config.get('RMSNorm', False), config.get('use_rope', False),
                      config['lig_pred'], config['prot_pred'], config.get('lig_moe_n_head', 1), config.get('prot_moe_n_head', 1),
                      config.get('pred_n', 1), config['layer_num'], config.get('modal_token', False), config.get('softmax_partition', False),
                      config['pred'], config['shared_exp'], config['gatter'], config['router_noise'], config['router_act'],
                      config['prot_scale'], config['n_head'], config['drop_out'], config.get('lig_emb_dim', 384),
                      lig_token_moe=config.get('lig_token_moe', None), moe_ffn=config.get('moe_ffn', None), hydraformer=config.get('hydraformer', False),
                      norm_first=config.get('norm_first', False), use_method_id=config.get('use_method_id', []),
                      gated_sdpa=config.get('gated_sdpa', False), token_moe_mask=config.get('token_moe_mask', False), **ext_kwgs)
    elif config['arch'] in {'arch11', 'arch3', 'arch31', 'arch4', 'arch41', 'arch42', 'arch43', 'arch44'}:
        model = _str2arch[config['arch']](lig_dim, prot_dim, config['hidden_feat_size'], config['RMSNorm'], config['use_rope'],
                                          config['lig_pred'], config['prot_pred'], config['layer_num'],
                                          config['pred'], config['shared_exp'], config['gatter'], config['router_noise'], config['router_act'],
                                          config['prot_scale'], config['n_head'], config['drop_out'], config.get('lig_emb_dim', 384), **ext_kwgs)
    elif config['arch'] == 'arch2':
        model = Arch2(lig_dim, prot_dim, config['prot_n'], config['hidden_feat_size'],
                      config['RMSNorm'], config['use_rope'], config['feat_mini_mhsa'], config['norm_first'],
                      config['lig_pred'], config['prot_pred'], config['lig_moe_n_head'], config['prot_moe_n_head'], config['layer_num'],
                      config['pred'], config['shared_exp'], config['gatter'], config['router_noise'], config['router_act'],
                      config['prot_scale'], config['n_head'], config['drop_out'],
                      lig_token_moe=config['lig_token_moe'], moe_ffn=config['moe_ffn'], hydraformer=config['hydraformer'],
                      use_method_id=config['use_method_id'],
                      gated_sdpa=config['gated_sdpa'], token_moe_mask=config['token_moe_mask'], **ext_kwgs)
    elif config['arch'] == 'arch21':
        model = Arch21(lig_dim, prot_dim, config['lig_n'], config['prot_n'], config['hidden_feat_size'], config['RMSNorm'], config['use_rope'],
                       config['lig_pred'], config['prot_pred'], config['layer_num'],
                       config['pred'], config['shared_exp'], config['gatter'], config['router_noise'], config['router_act'],
                       config['prot_scale'], config['n_head'], config['drop_out'])
    elif config['arch'] in {'arch22', 'arch23', 'arch24'}:
        model = _str2arch[config['arch']](lig_dim, prot_dim, config['lig_n'], config['prot_n'], config['pred_n'],
                                          config['hidden_feat_size'], config['RMSNorm'], config['use_rope'],
                                          config['lig_pred'], config['prot_pred'], config['layer_num'],
                                          config['pred'], config['shared_exp'], config['gatter'], config['router_noise'], config['router_act'],
                                          config['prot_scale'], config['n_head'], config['drop_out'])
    else:
        raise ValueError(f"Unknown architecture: {config['arch']}")
    log_fn(str(model))
    return model


def calcu_moe_banlance_loss(model: Arch1, criterion_moe, config):
    if model.is_MoE:
        moe_act_toggle, moe_act = config['model']['moe_act_toggle'], 1e10
        if moe_act_toggle is not None and model.pred_is_MoE:
            moe_act = model.predictor.expert_activation.std() / model.predictor.expert_activation.mean()
        if moe_act_toggle is None or moe_act > moe_act_toggle:
            return model.calcu_moe_loss(criterion_moe).to(config['training']['device']) * config['model']['moe_loss_scale']
    return 0

def get_loss(model: Arch1, logits, label, criterion, criterion_moe, config, is_calcu_moe_banlance_loss: bool = True):
    if (isinstance(config['model']['pred'], str) and 'BiLevel' in config['model']['pred']) or \
       (isinstance(config['model']['pred'], list) and any('BiLevel' in n for n in config['model']['pred'])):
        logits = logits.reshape(label.shape[0], -1)
        loss_microm = criterion(logits[:, -1], label - label.floor())
        loss_marco = criterion_moe(logits[:, :-1], label.floor().long())
        with torch.no_grad():
            scale = ((loss_marco.argmax(-1).view(-1)+loss_microm.view(-1))-label.view(-1)).abs().view(-1).mean().item()
        loss = loss_microm + loss_marco * max(1, scale**2)
    else:
        loss = criterion(logits, label)
    if model.is_MoE and is_calcu_moe_banlance_loss:
        moe_banlance_loss = calcu_moe_banlance_loss(model, criterion_moe, config)
        loss = loss + moe_banlance_loss
    return loss

def run_one_config(cfg: str|dict[str, bool|str|dict[str, float]], this_run_wd: Path = None, train_split: dict[str, list[str]] = None):
    if isinstance(cfg, str):
        config = Config(cfg).get_config()
    else:
        config = cfg
    cfg_name = config['name']
    taskpool = TaskPool('thread').start()
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
        if config['data']['lig_type'] in {'token', 'PepDoRA-token'}:
            lig_dim = 1
            prot_dim = train_loader.dataset.data[0][4].shape[-1]
        elif train_loader.dataset.umol_v4_pool:
            # umol_v4_pool: List[List[idx, mid, List[feat, mask]], rec_feat, pKa]
            lig_dim = train_loader.dataset.umol_v4_pool[train_loader.dataset.data[0][0]][2][0][0].shape[-1]
            prot_dim = train_loader.dataset.umol_v4_pool[train_loader.dataset.data[0][0]][3].shape[-1]
        else:
            lig_dim = train_loader.dataset.data[0][2].shape[-1]
            prot_dim = train_loader.dataset.data[0][4].shape[-1]
        # model
        device = torch.device(config['training']['device'])
        model =  get_model(config['model'], lig_dim, prot_dim, logger).to(device)
        if not (config['training']['no_compile']):
            model: nn.Module = torch.compile(model)
        # optimizer
        lr, wd = config['training']['lr'], config['training']['weight_decay']
        if config['training']['optimizer'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif config['training']['optimizer'] == 'radam':
            optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=wd, decoupled_weight_decay=True)
        elif config['training']['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
        # scheduler
        min_lr = 0.001 * lr if config['training']['scheduler'] in str2scheduleF else 0.01 * lr
        model_scheduler = get_model_scheduler(optimizer, scheduler_type=config['training']['scheduler'], initial_value=lr,
                                            total_epochs=config['training']['epochs'], scheduler_T=config['training']['scheduler_T'],
                                            min_value=min_lr, patience=10, max_value=lr, warmup_epochs=config['training']['warmup_epochs'])
        # criterion
        criterion = nn.MSELoss()
        criterion_moe = nn.CrossEntropyLoss()
        # torch.autograd.set_detect_anomaly(True)
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # meters
        meters = Meters()
        meters.add_meters(MeterType('train', ':.4f'), MeterType('valid', ':.4f'),
                          MeterType('test2013', ':.4f'), MeterType('test2016', ':.4f'),
                          MeterType('best_2016', ':.4f', 'min'),
                          MeterType('MoE_act', ':.4f'), MeterType('MoE_loss', ':.4f'))
        meters.make_progress(1, mp = logger.info if not config['debug'] else print)
        best_model_list = []
        # scheduler manager
        scheduler_manager = SchedulerManager({
            'model': model_scheduler,
        })
        _log_fn(str(scheduler_manager))
        # start training
        valid_rmse, test2016_rmse = 0, 0
        # is_freeze_disabled = False
        model.train()
        for epoch in range(config['training']['epochs']):

            # train for one epoch
            model.train()
            for b_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}', leave=False):
                _, mid, lig_feat, mask, prot_feat, label = data
                mid, lig_feat, mask, prot_feat, label = mid.to(device), lig_feat.to(device), mask.to(device), prot_feat.to(device), label.to(device)
                
                # add input noise
                if config['model']['noise_rate'] >= 0: # make sure even noise_rate is 0, the data order is same because the random is also called
                    if config['data']['lig_type'] not in {'token', 'PepDoRA-token'}:
                        lig_feat += torch.randn_like(lig_feat) * config['model']['noise_rate']
                    prot_feat += torch.randn_like(prot_feat) * config['model']['noise_rate']
                
                # add label noise
                if config['model']['label_noise_rate'] >= 0: # make sure even label_noise_rate is 0, the data order is same because the random is also called
                    label = label + torch.randn_like(label) * config['model']['label_noise_rate']
                    
                optimizer.zero_grad()
                
                # scheduler step
                scheduler_manager.step_in_epoch('model', b_idx/len(train_loader)+epoch)
                if config['debug']:
                    print(f'feat_encode: SMILE range {lig_feat.min():7.0f}~{lig_feat.max():7.0f}, PROT range {prot_feat.min():7.0f}~{prot_feat.max():7.0f}')
                    lig_feat, prot_feat = model.feat_encode(lig_feat, mask, prot_feat)
                    x = model.feat_forward(mid, lig_feat, mask, prot_feat)  # [N, 1, D]
                    logits = model.predict_from_feat(x)
                    print(f'feat_encode: LIG range {lig_feat.min():7.0f}~{lig_feat.max():7.0f}, PROT range {prot_feat.min():7.0f}~{prot_feat.max():7.0f}')
                    if torch.any(torch.isnan(lig_feat)) or torch.any(torch.isnan(prot_feat)) or torch.any(torch.isnan(logits)):
                        raise ValueError('NaN in feat_encode')
                    loss = get_loss(model, logits.reshape(-1), label.reshape(-1), criterion, criterion_moe, config)
                else:
                    logits = model(mid, lig_feat, mask, prot_feat, noise_rate=config['model']['noise_rate'])
                    loss = get_loss(model, logits.reshape(-1), label.reshape(-1), criterion, criterion_moe, config)
                
                # attn1_weights.append(model.attn.transformer.layers[0].code_hack_atten_weights.mean(dim=0).cpu())
                # attn2_weights.append(model.attn.transformer.layers[1].code_hack_atten_weights.mean(dim=0).cpu())
                
                loss.backward()
                optimizer.step()

                meters.update('train', loss.item(), label.size(0))
                if model.is_MoE:
                    meters.update('MoE_loss', model.MoE_balance_loss.item(), label.size(0))
                if model.pred_is_MoE:
                    meters.update('MoE_act', model.predictor.expert_activation.std() / model.predictor.expert_activation.mean(), label.size(0))
            
            # log loss
            epoch_rmse = np.sqrt(meters.get('train').avg)
            moe_banlance_loss = model.MoE_balance_loss if model.is_MoE and config['model']['router_noise'] else 0
            if config['debug']:
                meters.display(epoch)
                if model.pred_is_MoE:
                    if config['model']['router_noise']:
                        print(f'MoE balance_loss: {model.MoE_balance_loss.item():.4f}')
                    print(f'MoE expert_activation: {model.predictor.expert_activation.cpu().numpy().astype(int).tolist()}')
                    print(f'MoE gate_density: {[f"{i.cpu().item():.4f}" for i in model.predictor.gate_density]}')
            else:
                # attn1_weights = torch.stack(attn1_weights, dim=0).mean(dim=0)
                # attn2_weights = torch.stack(attn2_weights, dim=0).mean(dim=0)
                # sns.heatmap(attn1_weights)
                # save_show(os.path.join(logger.get_model_dir(), f'attn1_heatmap_epoch_{epoch}.png'), dpi=300, show=False)
                # plt.close()
                # sns.heatmap(attn2_weights)
                # save_show(os.path.join(logger.get_model_dir(), f'attn2_heatmap_epoch_{epoch}.png'), dpi=300, show=False)
                # plt.close()
                # log loss
                writer.add_scalar('train_rmse', epoch_rmse, epoch)
                writer.add_scalar('MoE_act', meters.get('MoE_act').avg, epoch)
                writer.add_scalar('MoE_loss', moe_banlance_loss, epoch)
                # validating
                _, valid_rmse, valid_pr = val(model, valid_loader, device, config)
                _, test2013_rmse, test2013_pr = val(model, test2013_loader, device, config)
                _, test2016_rmse, test2016_pr = val(model, test2016_loader, device, config)
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
                    model_ckp = get_model_state_dict_copy(model)
                    taskpool.add_task(None, save_state_dict_in_thread, model_ckp, logger.get_model_dir(), msg)
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
            # if not config['debug']:
            scheduler_manager.step_after_epoch('model', epoch, valid_rmse)
                    
            meters_need_reset = list(meters.meters.keys())
            meters_need_reset.remove('best_2016')
            meters.resets(*meters_need_reset)
            if model.pred_is_MoE:
                model.predictor.reset_expert_activation()
                
            # check whether exists `quit.txt` in model dir
            if logger is not None and os.path.exists(os.path.join(logger.get_model_dir(), 'quit.txt')):
                break
            
        if not config['debug']:
            # close this run logger
            writer.close()
            # final testing
            load_model_dict(model, best_model_list[-1])
            _, valid_rmse, valid_pr = val(model, valid_loader, device, config)
            _, test2013_rmse, test2013_pr = val(model, test2013_loader, device, config)
            _, test2016_rmse, test2016_pr = val(model, test2016_loader, device, config)
            _, test2019_rmse, test2019_pr = val(model, test2019_loader, device, config)
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
    taskpool.close(30)


def make_args():
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument("-c", "--config", type=str, default=None,
                            help="config file name in task/configs dir, default is %(default)s")
    args_paser.add_argument("-a", type=str, choices=['arch1', 'arch11', 'arch2', 'arch21', 'arch22', 'arch23', 'arch24',
                                                     'arch3', 'arch31', 'arch4', 'arch41', 'arch42', 'arch43', 'arch44'], default='arch1',
                            help="architecture, default is %(default)s")
    args_paser.add_argument("--lig-n", type=int, default=3,
                            help="number of lig token, default is %(default)s")
    args_paser.add_argument("--prot-n", type=int, default=8,
                            help="number of prot token, default is %(default)s")
    args_paser.add_argument("--pred-n", type=int, default=1,
                            help="number of pred token, default is %(default)s")
    args_paser.add_argument("--modal-token", action="store_true", default=False,
                            help="use modal token, default is %(default)s")    
    args_paser.add_argument("-l", type=int, default=2,
                            help="number of layer, default is %(default)s")
    args_paser.add_argument("-d", type=int, default=512,
                            help="number of hidden, default is %(default)s")
    args_paser.add_argument("--token-moe-mask", action="store_true", default=False,
                            help="use token moe mask, default is %(default)s")    
    add_pred_args(args_paser)
    add_transformer_args(args_paser)
    add_noise_args(args_paser)
    add_train_control_args(args_paser)
    args_paser.add_argument("-lig", type=str, default='PepDoRA', choices=['PepDoRA', 'PepDoRA_fix', 'ChemBERTa_10M', 'ChemBERTa_100M_MLM', 'ChemBERTa_77M_MTR', 'ChemBERTa_77M_MLM',
                                                                          'MolFormer', 'SELFormer', 'MolAI', 'GeminiMol',
                                                                          'UniMol-v3', 'UniMol-v3-570m', 'UniMol-v4', 'UniMol-v4-gen',
                                                                          'token', 'PepDoRA-token', '3Dimg', '3Dimg_vit', 'MaskMol_224',
                                                                          'rdkit_vit_224', 'rdkit_vit_512', 'ImageMol'],
                            help="lig type, default is %(default)s")
    args_paser.add_argument("--lig-emb-dim", type=int, default=384,
                            help="lig embedding dimension, default is %(default)s")
    args_paser.add_argument("--lig-token-moe", action="store_true", default=False,
                            help="use token moe for lig, default is %(default)s")
    args_paser.add_argument("-r", type=float, default=0.01,
                            help="dataset load ratio, default is %(default)s")
    args_paser.add_argument("-lo", type=str, default='front', choices=['front', 'back', 'random', 'more'],
                            help="dataset load order, default is %(default)s")
    args_paser.add_argument("--cat-v4-n", type=int, default=None,
                            help="number of unimol v4 data to concat, default is %(default)s")
    args_paser.add_argument("-prot", type=str, default='esm3-open_split', choices=['esm2-mean_each_mean', 'esm2-3B_mean_each_mean', 'esm3-open_split',
                                                                                   'esm2-3B-36_mean_each_mean', 'esm2-3B-30_mean_each_mean', 'ProSST-2048-fix_mean_each_mean',
                                                                                   'ProSST-2048_mean_each_mean', 'UniMol-v3', 'SaProt'],
                            help="prot type, default is %(default)s")
    args_paser.add_argument("--prot-std", action="store_true", default=False,
                            help="use mean, std to normalize prot, default is %(default)s")
    args_paser.add_argument("--prot-scale", action='store_true', default=False,
                            help="scale prot feature after loaded, default is %(default)s")
    args_paser.add_argument("--prot-transform", type=str, nargs='+', default=None,
                            help="transform prot feature after loaded, default is %(default)s")
    args_paser.add_argument("--lig-seq-reduce", type=str, default=None, choices=['sum', 'mean', 'zero'],
                            help="reduce lig data along seq dim, default is %(default)s")
    args = args_paser.parse_args()
    
    if args.config is None:
        args.config = Config('s1_EHIGN_base').get_config()
        
        # name
        model_data = f'{args.lig}{"std" if args.prot_std else ""}'
        model_detail = f'{args.l}h{args.d}do{args.do}ns{args.ns}lns{args.lns}bs{args.bs}'
        if args.a in {'arch1', 'arch11', 'arch3', 'arch31', 'arch4', 'arch41', 'arch42', 'arch43', 'arch44'}:
            args.config['name'] = f's1_{args.a}{",".join(args.pred)}|{args.gatter}_{args.optim}{args.lr}_{model_data}_{model_detail}'
        elif args.a == 'arch2':
            args.config['name'] = f's1_{args.a}{",".join(args.pred)}PN{args.prot_n}|{args.gatter}_{args.optim}{args.lr}_{model_data}_{model_detail}'
        elif args.a == 'arch21':
            args.config['name'] = f's1_{args.a}{",".join(args.pred)}PN{args.prot_n}LN{args.lig_n}|{args.gatter}_{args.optim}{args.lr}_{model_data}_{model_detail}'
        elif args.a in {'arch22', 'arch23', 'arch24'}:
            args.config['name'] = f's1_{args.a}{",".join(args.pred)}PN{args.prot_n}LN{args.lig_n}TN{args.pred_n}|{args.gatter}_{args.optim}{args.lr}_{model_data}_{model_detail}'
        # data
        args.config['data']['lig_type'] = args.lig
        args.config['data']['load_ratio'] = args.r
        args.config['data']['load_order'] = args.lo
        args.config['data']['cat_v4_n'] = args.cat_v4_n
        args.config['data']['prot_type'] = args.prot
        args.config['data']['use_prot_std'] = args.prot_std
        args.config['data']['lig_seq_reduce'] = args.lig_seq_reduce
        args.config['data']['prot_transform'] = args.prot_transform
        
        # arch
        args.config['model']['arch'] = args.a
        args.config['model']['lig_n'] = args.lig_n
        args.config['model']['lig_emb_dim'] = args.lig_emb_dim
        args.config['model']['lig_token_moe'] = args.lig_token_moe
        args.config['model']['prot_n'] = args.prot_n
        args.config['model']['pred_n'] = args.pred_n
        args.config['model']['modal_token'] = args.modal_token
        args.config['model']['layer_num'] = args.l
        args.config['model']['hidden_feat_size'] = args.d
        args.config['model']['token_moe_mask'] = args.token_moe_mask
        args.config['model']['prot_scale'] = args.prot_scale
        
        # unify
        assign_pred_args(args)
        assign_noise_args(args)
        assign_train_control_args(args)
        assign_transformer_args(args)
    
    return args


if __name__ == '__main__':
    # command launch
    args = make_args()    
    run_one_config(args.config)
