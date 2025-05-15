# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: swint.py
@Time    : 2025/5/15 下午11:41
@Author  : ZhouFei
@Email   : zhoufei21@s.nuit.edu.cn
@Desc    : 
@Usage   :
"""
from timm import create_model
from torch import nn


class VisualEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 使用Swin Transformer作为视觉编码器
        self.backbone = create_model(model_name=cfg.model.backbone_name, features_only=True, out_indices=[0, 1, 2, 3],
                                     pretrained=cfg.model.pretrained, img_size=cfg.model.img_size)
        self.out_dims = cfg.model.out_dims

    def forward(self, x):
        """
        x: [B,3,H,W]
        returns list of multi-scale features:
          feats[i] is [B, C_i, H/2^{i+2}, W/2^{i+2}]
        """
        feats = self.backbone(x)
        # 将每个特征图的维度从 [B, H, W, C] 转换为 [B, C, H, W]
        feats = [feat.permute(0, 3, 1, 2).contiguous() for feat in feats]
        return feats


# 测试
if __name__ == '__main__':
    import torch
    from utils import load_config

    cfg = load_config('../configs/config.yaml')
    # 测试Swin Transformer
    model_swin = VisualEncoder(cfg)
    x = torch.randn(2, 3, 512, 512)  # 示例输入：2张3通道512x512的图像
    feats_swin = model_swin(x)
    print("Swin Transformer Features:")
    for i, feat in enumerate(feats_swin):
        print(f"Layer {i} feature shape: {feat.shape}")

    # 计算FLOPs和Params
    from thop import profile

    flops_swin, params_swin = profile(model_swin, inputs=(x,))
    print(f"\nSwin Transformer Backbone FLOPs: {flops_swin / 1e9:.2f} G, Params: {params_swin / 1e6:.2f} M")
