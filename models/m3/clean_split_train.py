
import argparse
import os
import sys

from mbapy.base import get_fmt_time
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import shutil
import warnings
from pathlib import Path
from mbapy import get_paths_with_extension, opts_file

from config.config_dict import *
from log.train_logger_v1 import *
from models._utils.arg import *
from models.m3.train import make_args
from models.m3.train import run_one_config as _run_one_config

warnings.filterwarnings('ignore')


def run_one_config(cfg: str):
    if isinstance(cfg, str):
        config = Config(cfg).get_config()
    else:
        config = cfg
    cfg_name = config['name']
    if not config['debug']:
        this_run_wd = Path(config['training']['save_dir']) / f'{get_fmt_time("%Y%m%d-%H%M%S")}-CleanTrain-{cfg_name}'
        os.makedirs(this_run_wd, exist_ok=True)
        import fnmatch
        def ignore_static_files(_, files):
            return [f for f in files if any(fnmatch.fnmatch(f, fmt) for fmt in ['*.pt', '*.joblib'])]
        shutil.copytree('models', this_run_wd / f'src', ignore=ignore_static_files)
        
    for clean_split_path in get_paths_with_extension('data/clean_split', ['json'], name_substr='cleansplit'):
        clean_split = opts_file(clean_split_path, way='json')
        _this_run_wd = this_run_wd / f'{Path(clean_split_path).stem}'
        os.makedirs(_this_run_wd, exist_ok=True)
        config['training']['random_seed'] = [0]
        print(f'clean_split: {clean_split_path}')
        _run_one_config(config, _this_run_wd, clean_split)


if __name__ == '__main__':
    # command launch
    args = make_args()
    wait_till_start_time(args.start_time)
    run_one_config(args.config)
