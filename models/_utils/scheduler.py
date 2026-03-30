from typing import Union, Optional

import torch
import torch.optim as optim
from mbapy.dl_torch.optim import LrScheduler, str2scheduleF


class WarmupScheduler:
    """
    Warmup scheduler that combines warmup phase with another scheduler.
    Starts from 0 and linearly increases to the target value over warmup_steps,
    then continues with the base scheduler.
    """
    def __init__(self, optimizer: optim.Optimizer, base_scheduler, warmup_epochs: int, total_epochs: int, initial_value: float):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.initial_value = initial_value
        self.now_lr = initial_value
        
    def __repr__(self):
        return f"WarmupScheduler(warmup_epochs={self.warmup_epochs}, total_epochs={self.total_epochs}, initial_value={self.initial_value}, base_scheduler={self.base_scheduler})"
        
    def step(self, epoch: float):
        """Step the scheduler and return the current value."""
        # Warmup phase
        if epoch <= self.warmup_epochs:
            # Linear warmup from 0 to initial_value
            lr = self.initial_value * (epoch / self.warmup_epochs + 1e-8)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Base scheduler phase
            if hasattr(self.base_scheduler, 'step') and callable(getattr(self.base_scheduler, 'step')):
                # Adjust epoch for base scheduler (subtract warmup steps)
                adjusted_epoch = epoch - self.warmup_epochs
                lr = self.base_scheduler.step(adjusted_epoch)
            if hasattr(self.base_scheduler, 'get_last_lr'):
                lr = self.base_scheduler.get_last_lr()[0]
        self.now_lr = lr
        return lr
    
    def get_last_lr(self):
        """Get the current value."""
        return [self.now_lr]


class ConstantScheduler:
    """
    Constant scheduler that maintains a fixed value.
    Can be used for warmup-only effect when combined with warmup.
    """
    def __init__(self, value: float):
        self.value = value
        
    def __repr__(self):
        return f"ConstantScheduler(value={self.value})"
        
    def step(self, epoch: Optional[float] = None):
        """Step the scheduler and return the constant value."""
        return self.value
    
    def get_last_lr(self):
        """Get the constant value."""
        return [self.value]


def create_scheduler(optimizer, scheduler_type, initial_value, total_epochs, 
                     scheduler_T=None, min_value=None, patience=None, max_value=None,
                     warmup_epochs: Optional[int] = None):
    """
    基础的scheduler创建函数，根据类型和参数生成对应的调度器
    
    Args:
        optimizer: 优化器实例
        scheduler_type: 调度器类型 ('plateau', 'cosine' 或 str2scheduleF中支持的类型)
        initial_value: 初始值（用于学习率或超参数）
        total_epochs: 总训练轮数
        scheduler_T: 调度器周期参数（用于某些自定义调度器）
        min_value: 最小值限制
        patience: plateau调度器的耐心值
        max_value: 最大值限制（用于某些调度器）
        warmup_epochs: warmup轮数，如果指定则启用warmup
    
    Returns:
        scheduler: 创建的调度器实例
    """
    # Handle constant scheduler (warmup-only effect)
    if scheduler_type == 'constant':
        scheduler = ConstantScheduler(initial_value)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=patience or 10,
            min_lr=min_value or 0.01 * initial_value,
            verbose=False,
            **({'max_lr': max_value} if max_value is not None else {})
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=min_value or 0.01 * initial_value
        )
    elif isinstance(scheduler_type, str) and scheduler_type.startswith('exponential'):
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_T,
        )
    elif scheduler_type in str2scheduleF:
        scheduler = LrScheduler(
            optimizer,
            initial_value,
            0,
            scheduler_T,
            total_epochs,
            scheduler_type,
            min_lr=min_value or 0.01 * initial_value
        )
    else:
        scheduler = None
    
    # Apply warmup if specified
    if warmup_epochs is not None and warmup_epochs > 0 and scheduler is not None:
        scheduler = WarmupScheduler(optimizer, scheduler, warmup_epochs, total_epochs, initial_value)
    
    return scheduler


def get_model_scheduler(optimizer, scheduler_type, initial_value, total_epochs, scheduler_T=None,
                        min_value=None, patience=None, max_value=None, warmup_epochs: Optional[int] = None):
    """
    获取模型参数的学习率调度器
    
    Args:
        optimizer: 优化器实例
        scheduler_type: 调度器类型 ('plateau', 'cosine' 或 str2scheduleF中支持的类型)
        initial_value: 初始值（用于学习率或超参数）
        total_epochs: 总训练轮数
        scheduler_T: 调度器周期参数（用于某些自定义调度器）
        min_value: 最小值限制
        patience: plateau调度器的耐心值
        max_value: 最大值限制（用于某些调度器）
        warmup_epochs: warmup轮数，如果指定则启用warmup
    
    Returns:
        scheduler: 学习率调度器实例
    """
    return create_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        initial_value=initial_value,
        total_epochs=total_epochs,
        scheduler_T=scheduler_T,
        min_value=min_value,
        patience=patience,
        max_value=max_value,
        warmup_epochs=warmup_epochs
    )


def get_hyperparameter_scheduler(value: float, scheduler_type: str, total_epochs: int,
                                 scheduler_T=None, min_value=None, patience=None, max_value=None, warmup_epochs: Optional[int] = None):
    """
    获取超参数的调度器
    
    Args:
        value: 超参数初始值
        scheduler_type: 调度器类型 ('plateau', 'cosine' 或 str2scheduleF中支持的类型)
        total_epochs: 总训练轮数
        scheduler_T: 调度器周期参数（用于某些自定义调度器）
        min_value: 最小值限制
        patience: plateau调度器的耐心值
        max_value: 最大值限制（用于某些调度器）
        warmup_steps: warmup步数，如果指定则启用warmup
    
    Returns:
        scheduler: 学习率调度器实例
    """
    if value is None:
        return None
    fake_optimizer = optim.SGD([torch.tensor(value, dtype=torch.float32, requires_grad=True)], lr=value)
        
    scheduler = create_scheduler(
        optimizer=fake_optimizer,
        scheduler_type=scheduler_type,
        initial_value=value,
        total_epochs=total_epochs,
        scheduler_T=scheduler_T,
        min_value=min_value or 0.01 * value,
        patience=patience or scheduler_T,
        max_value=max_value,
        warmup_epochs=warmup_epochs
    )
    
    return scheduler


class SchedulerManager:
    def __init__(self, schedulers: dict[str, Union[LrScheduler, optim.lr_scheduler._LRScheduler, WarmupScheduler, ConstantScheduler, None]]):
        self.schedulers = schedulers
        
    def __repr__(self):
        return f"SchedulerManager: \n" + '\n'.join([f'{name}: {scheduler}' for name, scheduler in self.schedulers.items()])
        
    def step_after_epoch(self, name: str, epoch: int, perf):
        """step for CosineAnnealingLR or ReduceLROnPlateau"""
        scheduler = self.schedulers.get(name, None)
        if scheduler is None:
            return
        # Handle PyTorch built-in schedulers or with warmup
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR) or \
                (isinstance(scheduler, WarmupScheduler) and isinstance(scheduler.base_scheduler, optim.lr_scheduler.CosineAnnealingLR)):
            scheduler.step(epoch)
        elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) or \
                (isinstance(scheduler, WarmupScheduler) and isinstance(scheduler.base_scheduler, optim.lr_scheduler.ReduceLROnPlateau)):
            scheduler.step(perf)
        elif isinstance(scheduler, optim.lr_scheduler.ExponentialLR) or \
                (isinstance(scheduler, WarmupScheduler) and isinstance(scheduler.base_scheduler, optim.lr_scheduler.ExponentialLR)):
            scheduler.step()
        else:
            return
        return scheduler.get_last_lr()[0]
    
    def step_in_epoch(self, name: str, epoch: float):
        """step for other schedulers"""
        scheduler = self.schedulers.get(name, None)
        if scheduler is None:
            return
        # Handle LrScheduler from mbapy or with warmup
        elif isinstance(scheduler, LrScheduler) or \
                (isinstance(scheduler, WarmupScheduler) and isinstance(scheduler.base_scheduler, LrScheduler)) or \
                    (isinstance(scheduler, WarmupScheduler) and epoch < scheduler.warmup_epochs):
            return scheduler.step(epoch)
        else:
            return


__all__ = [
    'get_model_scheduler',
    'get_hyperparameter_scheduler',
    'SchedulerManager',
]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for gamma in [0.995, 0.99, 0.98, 0.95]:
        scheduler = get_hyperparameter_scheduler(
            value=1,
            scheduler_type='exponential',
            total_epochs=1000,
            scheduler_T=gamma,
            min_value=0.0001,
            patience=10,
            max_value=0.01
        )
        manager = SchedulerManager({'dgl_lambda': scheduler})
        print(manager)        
        values = []
        for epoch in range(500):
            values.append(manager.step_after_epoch('dgl_lambda', epoch, None))
        plt.plot(values, label=f'gamma={gamma}')
    plt.legend()
    plt.grid()
    plt.savefig('exponential_scheduler.png', dpi=600)
