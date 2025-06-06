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

__all__ = ['SwinT']


class SwinT(nn.Module):
    def __init__(self, name: str = 'swin_base_patch4_window7_224', img_size: tuple = (512, 640),
                 return_idx: list = [1, 2, 3], pretrained: bool = True):
        super().__init__()
        # 使用Swin Transformer作为视觉编码器
        self.backbone = create_model(model_name=name, features_only=True, out_indices=return_idx,
                                     pretrained=pretrained, img_size=img_size)
        self.out_dims = [96, 192, 384, 768]

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

    # 测试Swin Transformer
    model_swin = SwinT(name='swin_base_patch4_window7_224', img_size=(512, 640), return_idx=[1, 2, 3], pretrained=True)
    x = torch.randn(2, 3, 512, 640)  # 示例输入：2张3通道512x640的图像
    feats_swin = model_swin(x)
    print("Swin Transformer Features:")
    for i, feat in enumerate(feats_swin):
        print(f"Layer {i + 1} feature shape: {feat.shape}")

    # 计算FLOPs和Params
    from thop import profile

    flops_swin, params_swin = profile(model_swin, inputs=(x,))
    print(f"\nSwin Transformer Backbone FLOPs: {flops_swin / 1e9:.2f} G, Params: {params_swin / 1e6:.2f} M")
