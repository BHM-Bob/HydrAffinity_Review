import os
import pickle

import torch


def get_model_state_dict_copy(model: torch.nn.Module):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def save_state_dict_in_thread(state_dict, model_dir, msg):
    """
    在子线程中保存模型状态字典
    Args:
        state_dict: 从主线程传入的模型状态字典
        model_dir: 保存模型的目录
        msg: 模型文件名前缀
    """
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(state_dict, model_path)
    print("model state dict has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, name: str, best_type):
        self.name = name
        self.best_type = best_type  
        self.count = 0 
        self.sum_best = 0     
        self.reset()

    def reset(self):
        self.sum_best = 0
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        """set the best value, reset the counter to 0, add 1 to the sum_best"""
        self.best = best
        self.count = 0
        self.sum_best += 1

    def get_best(self):
        return self.best

    def counter(self):
        """add 1 to the counter, and return the counter value"""
        self.count += 1
        return self.count
    
    def __str__(self):
        return f'{self.name}: {self.best:.4f}'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg
