# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: encoders.py
@Time    : 2025/5/16 下午8:06
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
"""
from torch import nn
from models import build_backbone
from models.encoder import CenterHead, AAGF

__all__ = ['Encoder']


class Encoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.out_dims = cfg.model.out_dims
        if cfg.model.backbone_name == 'resnet50':
            self.conv_list = nn.ModuleList([nn.Conv2d(in_dim, out_dim, kernel_size=1) for in_dim, out_dim in
                                            zip([256, 512, 1024, 2048], cfg.model.out_dims)])
        elif cfg.model.backbone_name == 'swint':
            # 添加1×1卷积层用于统一通道
            self.conv_list = nn.ModuleList([nn.Conv2d(in_dim, out_dim, kernel_size=1) for in_dim, out_dim in
                                            zip([128, 256, 512, 1024], cfg.model.out_dims)])
        elif cfg.model.backbone_name == 'hgnetv2':
            self.conv_list = nn.ModuleList([nn.Conv2d(in_dim, out_dim, kernel_size=1) for in_dim, out_dim in
                                            zip([64, 256, 512, 1024], cfg.model.out_dims)])

        # 锚点检测头
        self.ancher = CenterHead(self.out_dims[0], num_anchors=10)
        # 锚点注意力引导融合模块
        self.aagf = AAGF(channels=self.out_dims[0], roi_size=7, use_confidence=True, use_attention=False,
                         use_similarity=False)

    def forward(self, rgb, tir):
        feats_rgb = self.backbone(rgb)
        feats_tir = self.backbone(tir)

        # TODO: 舍弃第一层
        # 统一通道
        feats_rgb = [self.conv_list[i](feat) for i, feat in enumerate(feats_rgb)]
        feats_tir = [self.conv_list[i](feat) for i, feat in enumerate(feats_tir)]

        # 锚点检测头
        ancher_rgb = [self.ancher(feat) for feat in feats_rgb]
        ancher_tir = [self.ancher(feat) for feat in feats_tir]

        # 锚点注意力引导融合模块
        fused = [self.aagf(f_rgb, f_tir, a_rgb, a_tir) for f_rgb, f_tir, a_rgb, a_tir in zip(feats_rgb, feats_tir,  ancher_rgb, ancher_tir)]

        return fused


if __name__ == '__main__':
    import torch
    from utils import load_config

    cfg = load_config('../../configs/config.yaml')

    # 示例输入：2张3通道512x512的图像
    rgb = torch.randn(2, 3, 512, 640)
    tir = torch.randn(2, 3, 512, 640)
    model = Encoder(cfg)
    feats = model(rgb, tir)
    for feat in feats:
        print(feat.shape)

    from thop import profile

    flops, params = profile(model, inputs=(rgb, tir))
    print(f"encoder FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

