"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def build_optimizer(model, cfg):
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": cfg.training.lr
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.training.lr_backbone,
        },
    ]

    if cfg.training.optimizer == 'sgd':
        return optim.SGD(param_dicts, lr=cfg.training.lr, momentum=cfg.training.momentum,
                               weight_decay=cfg.training.weight_decay)
    elif cfg.training.optimizer == 'adam':
        return optim.Adam(param_dicts, lr=cfg.training.lr, betas=cfg.training.betas, eps=cfg.training.eps)
    elif cfg.training.optimizer == 'adamw':
        return optim.AdamW(param_dicts, lr=cfg.training.lr, betas=cfg.training.betas, eps=cfg.training.eps,
                                 weight_decay=cfg.training.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")


def build_lr_scheduler(optimizer, cfg):
    if cfg.training.scheduler == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=cfg.training.lr_drop, gamma=cfg.training.gamma)
    elif cfg.training.scheduler == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    elif cfg.training.scheduler == 'multistep':
        return lr_scheduler.MultiStepLR(optimizer, milestones=cfg.training.milestones, gamma=cfg.training.gamma)
    elif cfg.training.scheduler == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.T_max)
    elif cfg.training.scheduler == 'onecycle':
        return lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.training.max_lr, steps_per_epoch=cfg.training.steps_per_epoch, epochs=cfg.training.epochs)
    elif cfg.training.scheduler == 'lambda':
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=cfg.training.lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.training.scheduler}")