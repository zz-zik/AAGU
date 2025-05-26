# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: inference.py
@Time    : 2025/5/26 下午10:14
@Author  : ZhouFei
@Email   : zhoufei21@s.nuit.edu.cn
@Desc    : 
@Usage   :
"""
import torch

from models import build_model


def load_model(cfg, weights_dir, device):
    """
    加载模型函数
    Args:
        cfg: 配置文件
    Returns:
        model: nn.Module，加载的模型
    """
    model = build_model(cfg, training=False)
    model.to(device)

    if weights_dir is not None:
        checkpoint = torch.load(weights_dir, map_location=device)
        model.load_state_dict(checkpoint['model'])
    model.eval()
    return model
