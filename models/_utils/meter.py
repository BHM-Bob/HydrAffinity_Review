'''
self info removed
'''
from collections import namedtuple

from mbapy.dl_torch.utils import AverageMeter, ProgressMeter

from utils import BestMeter

MeterType = namedtuple('MeterType', ['name', 'fmt', 'type'], defaults=('avg', ))


class Meters:
    def __init__(self):
        self.meters: dict[str, AverageMeter | BestMeter] = {}
        self.progress: ProgressMeter | None = None
        
    def add_meter(self, name: str, fmt: str, _type: str = 'avg'):
        assert _type in {'avg', 'min', 'max'}, f'Only support avg, min and max, but got {_type}'
        if _type == 'avg':
            self.meters[name] = AverageMeter(name, fmt)
        else:
            self.meters[name] = BestMeter(name, _type)
            
    def add_meters(self, *args: list[MeterType]):
        for meter in args:
            if meter is not None:
                self.add_meter(*meter)
            
    def get(self, name: str):
        if not name in self.meters:
            raise ValueError(f'No meter named {name}')
        return self.meters[name]
            
    def make_progress(self, n: int, mp: None):
        self.progress = ProgressMeter(n, list(self.meters.values()), mp=mp)
        
    def update(self, name: str, value: float, n: int = 1):
        if not name in self.meters:
            raise ValueError(f'No meter named {name}')
        if isinstance(self.meters[name], AverageMeter):
            self.meters[name].update(value, n)
        else:
            self.meters[name].update(value)
        
    def display(self, idx: int):
        self.progress.display(idx)
        
    def reset(self, name: str):
        if name in self.meters:
            self.meters[name].reset()
        else:
            raise ValueError(f'No meter named {name}')
        
    def resets(self, *args: str):
        for name in args:
            self.reset(name)
        

class FakeSummaryWriter:
    """do nothing when calling add_scalar"""
    def __init__(self, *args, **kwargs):
        pass
    
    def add_scalar(self, *args, **kwargs):
        pass
    
    def close(self):
        pass
