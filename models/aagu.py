# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: aagu.py
@Time    : 2025/5/16 下午7:58
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
"""
from torch import nn

from models import Encoder


class AAGU(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg.model.backbone)


    def forward(self, rgb, tir):
        fused = self.encoder(rgb, tir)


