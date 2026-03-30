'''
self info removed
'''
import os
import sys

if sys.path[-1] != os.getcwd():
    sys.path.append(os.getcwd())

import json
import time

from config.config_dict import Config
from log.basic_logger import BasicLogger


def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

class TrainLogger(BasicLogger):
    def __init__(self, args: dict, config_name: str, this_run_wd: str, create: bool=True):
        self.args = args

        save_tag = f"randomseed{args['training']['now_random_seed']}"
        train_save_dir = os.path.join(this_run_wd, save_tag)
        self.log_dir = os.path.join(train_save_dir, 'log', 'train')
        self.model_dir = os.path.join(train_save_dir, 'model')
        self.result_dir = os.path.join(train_save_dir, 'result')

        if create:
            create_dir([self.log_dir, self.model_dir, self.result_dir])
            print(self.log_dir)
            log_path = os.path.join(self.log_dir, 'Train.log')
            super().__init__(log_path)
            self.record_config(config_name)

    def record_config(self, config_name):
        with open(os.path.join(self.log_dir, f'{config_name}.json'), 'w') as f:
            f.write(json.dumps(self.args))

    def get_log_dir(self):
        if hasattr(self, 'log_dir'):
            return self.log_dir
        else:
            return None

    def get_model_dir(self):
        if hasattr(self, 'model_dir'):
            return self.model_dir
        else:
            return None

    def get_result_dir(self):
        if hasattr(self, 'result_dir'):
            return self.result_dir
        else:
            return None


__all__ = ['TrainLogger']


if __name__ == "__main__":
    cfg_name = 'v2_EHIGN_l3h512'
    args = Config(cfg_name).get_config()
    logger = TrainLogger(args, cfg_name, '.')
    logger.record_config(cfg_name)
    model_path = logger.get_model_dir()
    print(model_path)