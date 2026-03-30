
import argparse
from typing import Union

from mbapy.dl_torch.optim import LrScheduler, str2scheduleF


def add_pred_args(args_paser: argparse.ArgumentParser):
    """
    Arguments:
        -pred: predictor, default is ['MLP']
        --shared-exp: shared expert, default is None
        -gatter: gatter type, default is 'sum', choices are ['sum', 'mean', 'weighted']
        -rn: router noise rate, default is False
        --moe-loss-scale: MoE balance loss scale, default is 0.05
    """
    args_paser.add_argument("-pred", type=str, nargs='+', default=['MLP'],
                            help="predictor, default is %(default)s")
    args_paser.add_argument("--lig-pred", type=str, nargs='+', default=['LinearDO'],
                            help="predictor, default is %(default)s")
    args_paser.add_argument("--prot-pred", type=str, nargs='+', default=['LinearDO'],
                            help="predictor, default is %(default)s")
    args_paser.add_argument("--lig-pred-moe-n-head", type=int, default=1,
                            help="predictor MoE head number, default is %(default)s")
    args_paser.add_argument("--prot-pred-moe-n-head", type=int, default=1,
                            help="predictor MoE head number, default is %(default)s")
    args_paser.add_argument("--feat-mini-mhsa", default=False, action='store_true',
                            help="whether to apply attention to each feature, default is %(default)s")
    args_paser.add_argument("--shared-exp", type=str, nargs='+', default=None,
                            help="shared expert, default is %(default)s")
    args_paser.add_argument("-gatter", type=str, default='sum', choices=['sum', 'mean', 'weighted'],
                            help="gatter type, default is %(default)s")
    args_paser.add_argument("--router-act", type=str, default='lambda', choices=['lambda', 'softmax', 'sigmoid', 'tanh'],
                            help="router activation function, default is %(default)s")
    args_paser.add_argument("-rn", default=False, action='store_true',
                            help="router noise rate, default is %(default)s")
    args_paser.add_argument("--moe-loss-scale", type=float, default=0.05,
                            help="MoE balance loss scale, default is %(default)s") 
    args_paser.add_argument("--moe-act-toggle", type=float, default=None,
                            help="MoE activation toggle, if moe activation std/mean is larger than this value, "
                                 "then use moe_loss_scale, else will set moe_loss_scale to 0 temporarily, default is %(default)s")
    return args_paser


def assign_pred_args(args: argparse.Namespace):
    """Assign predictor, shared expert, gatter type, router noise rate,
    MoE balance loss scale from args to args.config"""
    def _assign_pred_args(args: argparse.Namespace, pred_type: str, pred: Union[str, list[str]]):
        if isinstance(pred, list) and len(pred) == 1:
            args.config['model'][pred_type] = pred[0]
        else:
            args.config['model'][pred_type] = pred
    _assign_pred_args(args, 'pred', args.pred)
    _assign_pred_args(args, 'lig_pred', args.lig_pred)
    _assign_pred_args(args, 'prot_pred', args.prot_pred)
    _assign_pred_args(args, 'shared_exp', args.shared_exp)
    args.config['model']['lig_moe_n_head'] = args.lig_pred_moe_n_head
    args.config['model']['prot_moe_n_head'] = args.prot_pred_moe_n_head
    args.config['model']['feat_mini_mhsa'] = args.feat_mini_mhsa
    args.config['model']['gatter'] = args.gatter
    args.config['model']['router_noise'] = args.rn
    args.config['model']['moe_loss_scale'] = args.moe_loss_scale
    args.config['model']['router_act'] = args.router_act
    args.config['model']['moe_act_toggle'] = args.moe_act_toggle


def add_transformer_args(args_paser: argparse.ArgumentParser):
    """Add transformer arguments:
        - RMSNorm: use RMSNorm, default is False
        - no-rope: disable rope, default is False
        - norm-first: use norm first, default is False
        - softmax-partition: use softmax partition, default is False
        - moe-ffn: FALG for each transformer layer to use moe ffn, default is None
        - hydraformer: use hydraformer, default is False
        - gated-sdpa: use gated sdpa, default is False
        - low-mem-transformer: use low memory transformer, default is False
    """
    args_paser.add_argument("--RMSNorm", action="store_true", default=False,
                            help="use RMSNorm, default is %(default)s")
    args_paser.add_argument("--no-rope", action="store_true", default=False,
                            help="disable rope, default is %(default)s")    
    args_paser.add_argument("--norm-first", action="store_true", default=False,
                            help="use norm first, default is %(default)s")
    args_paser.add_argument("--softmax-partition", action="store_true", default=False,
                            help="use softmax partition, default is %(default)s")
    args_paser.add_argument("--moe-ffn", type=int, nargs='+', default = None,
                            help="FALG for each transformer layer to use moe ffn, default is %(default)s")
    args_paser.add_argument("--hydraformer", action="store_true", default=False,
                            help="use hydraformer, default is %(default)s")
    args_paser.add_argument("--use-method-id", type=str, nargs='+', choices=['replace', 'external', 'shared_hydraformer', 'separate_hydraformer', None], default=[],
                            help="use method id, default is %(default)s") 
    args_paser.add_argument("--gated-sdpa", type=int, default=0,
                            help="use gated sdpa. if 0, disabled; if -1, will start from seq-idx=0; if 1, will start from seq-idx=1, ...; default is %(default)s")  
    args_paser.add_argument("--low-mem-transformer", action="store_true", default=False,
                            help="use low memory transformer, default is %(default)s")
    return args_paser


def assign_transformer_args(args: argparse.Namespace):
    """Assign transformer arguments from args to args.config"""
    args.config['model']['RMSNorm'] = args.RMSNorm
    args.config['model']['use_rope'] = not args.no_rope
    args.config['model']['norm_first'] = args.norm_first
    args.config['model']['softmax_partition'] = args.softmax_partition
    args.config['model']['moe_ffn'] = args.moe_ffn
    args.config['model']['hydraformer'] = args.hydraformer
    args.config['model']['use_method_id'] = args.use_method_id
    args.config['model']['gated_sdpa'] = args.gated_sdpa
    args.config['model']['low_mem_transformer'] = args.low_mem_transformer
    
    # check hydraformer and use_method_id
    if 'replace' in args.use_method_id and any([x in args.use_method_id for x in ['shared_hydraformer', 'separate_hydraformer']]):
        raise ValueError("shared_hydraformer or separate_hydraformer only works with 'external' method-id type")


def add_noise_args(args_paser: argparse.ArgumentParser, default_do: float = 0.1,
                   default_ns: float = 0.05, default_lns: float = 0.05):
    """
    Arguments:
        -do: dropout rate, default is 0.1
        -ns: noise rate, default is 0.05
        -lns: label noise rate, default is 0.05
        -mask: mask rate, default is 0
        -fusion: fusion rate, default is 0
        -replace: replace rate, default is 0
        --token-replace: token replace rate, default is 0
    """
    args_paser.add_argument("-do", type=float, default=0.1,
                            help="dropout rate, default is %(default)s")
    args_paser.add_argument("-ns", type=float, default=0.05,
                            help="noise rate, default is %(default)s")
    args_paser.add_argument("-lns", type=float, default=0.05,
                            help="label noise rate, default is %(default)s")
    return args_paser


def assign_noise_args(args: argparse.Namespace):
    """Assign dropout rate, noise rate, label noise rate, mask rate,
    fusion rate, replace rate, token replace rate from args to args.config"""
    args.config['model']['drop_out'] = args.do
    args.config['model']['noise_rate'] = args.ns
    args.config['model']['label_noise_rate'] = args.lns


def add_train_control_args(args_paser: argparse.ArgumentParser):
    """
    Arguments:
        -optim: optimizer, default is 'adam', choices are ['adamw', 'sgd']
        -lr: learning rate, default is 1e-4
        -wd: weight decay, default is 1e-5
        -bs: batch size, default is 256
        -epoch: max epoch, default is 1000
        --ckp-path: checkpoint path, default is None
        --act-all-ckp: activate all parameters grad from checkpoint, default is False
        --no-compile: disable torch.compile, default is False
        --debug: debug mode, default is False
        --start-time: start time fmt is '2025-01-15 17:30:00', default is None
    """
    args_paser.add_argument("-optim", type=str, default='adamw', choices=['adamw', 'sgd', 'muon', 'radam'],
                            help="optimizer, default is %(default)s")
    args_paser.add_argument("-lr", type=str, default='1e-4',
                            help="learning rate, default is %(default)s")
    args_paser.add_argument("-wd", type=str, default='1e-6',
                            help="weight decay, default is %(default)s")
    args_paser.add_argument("--scheduler", type=str, default=None, choices=['constant', 'plateau', 'cosine']+list(str2scheduleF.keys()),
                            help="scheduler, default is %(default)s")
    args_paser.add_argument("--scheduler-T", type=int, default=50,
                            help="scheduler T_0, default is %(default)s")
    args_paser.add_argument("--warmup-epochs", type=int, default=None,
                            help="warmup epochs, default is %(default)s")
    args_paser.add_argument("-bs", type=int, default=256,
                            help="batch size, default is %(default)s")
    args_paser.add_argument("-epoch", type=int, default=1000,
                            help="max epoch, default is %(default)s")
    args_paser.add_argument("--least-epoch", type=int, default=300,
                            help="least epoch, default is %(default)s")
    args_paser.add_argument("--early-stop", type=int, default=100,
                            help="early_stop_epoch, default is %(default)s")
    args_paser.add_argument("--ckp-path", type=str, default=None,
                            help="checkpoint path, default is %(default)s")
    args_paser.add_argument("--act-all-ckp", action="store_true", default=False,
                            help="activate all parameters grad from checkpoint, default is %(default)s")
    args_paser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'],
                            help="device to use, default is %(default)s")
    args_paser.add_argument("--no-compile", action="store_true", default=False,
                            help="disable torch.compile, default is %(default)s")
    args_paser.add_argument("--no-tensorboard", action="store_true", default=False,
                            help="disable tensorboard, default is %(default)s")
    args_paser.add_argument("--debug", action="store_true", default=False,
                            help="debug mode, default is %(default)s")
    return args_paser


def assign_train_control_args(args: argparse.Namespace):
    """Assign optimizer, learning rate, weight decay, batch size,
    max epoch, early stop epoch, checkpoint path,
    activate all parameters grad from checkpoint, disable torch.compile,
    debug mode, start time from args to config from args to args.config"""
    args.config['training']['optimizer'] = args.optim
    args.config['training']['lr'] = eval(args.lr)
    args.config['training']['weight_decay'] = eval(args.wd)
    args.config['training']['scheduler'] = args.scheduler
    args.config['training']['scheduler_T'] = args.scheduler_T
    args.config['training']['warmup_epochs'] = args.warmup_epochs
    args.config['data']['batch_size'] = args.bs
    args.config['training']['epochs'] = args.epoch
    args.config['training']['least_epochs'] = args.least_epoch
    args.config['training']['early_stop_epoch'] = args.early_stop
    args.config['training']['ckp_path'] = args.ckp_path
    args.config['training']['activate_all_ckp'] = args.act_all_ckp
    args.config['training']['device'] = args.device    
    args.config['training']['no_compile'] = args.no_compile
    args.config['training']['no_tensorboard'] = args.no_tensorboard
    args.config['debug'] = args.debug
    

__all__ = [
    'add_pred_args', 'assign_pred_args',
    'add_transformer_args', 'assign_transformer_args',
    'add_noise_args', 'assign_noise_args',
    'add_train_control_args', 'assign_train_control_args',
]