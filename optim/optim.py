"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

__all__ = ["AdamW", "SGD", "Adam", "MultiStepLR", "CosineAnnealingLR", "OneCycleLR", "LambdaLR"]


SGD = optim.SGD
Adam = optim.Adam
AdamW = optim.AdamW


MultiStepLR = lr_scheduler.MultiStepLR
CosineAnnealingLR = lr_scheduler.CosineAnnealingLR
OneCycleLR = lr_scheduler.OneCycleLR
LambdaLR = lr_scheduler.LambdaLR
