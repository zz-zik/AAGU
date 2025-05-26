# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: backbones.py
@Time    : 2025/5/16 下午8:06
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
"""
from torch import nn
from models.backbones import CenterHead, AAGF, TFAM
from models.backbones.backbone import build_backbone

__all__ = ['BackBones']


# class BackBones(nn.Module):
#
#     def __init__(self, cfg):
#         super().__init__()
#         self.backbone = build_backbone(cfg)
#         self.out_dims = cfg.model.out_dims
#         if cfg.model.backbone_name == 'resnet50':
#             self.conv_list = nn.ModuleList([nn.Conv2d(in_dim, out_dim, kernel_size=1) for in_dim, out_dim in
#                                             zip([512, 1024, 2048], cfg.model.out_dims)])  # [256, 512, 1024, 2048]
#         elif cfg.model.backbone_name == 'swint':
#             # 添加1×1卷积层用于统一通道
#             self.conv_list = nn.ModuleList([nn.Conv2d(in_dim, out_dim, kernel_size=1) for in_dim, out_dim in
#                                             zip([256, 512, 1024], cfg.model.out_dims)])  # [128, 256, 512, 1024]
#         elif cfg.model.backbone_name == 'hgnetv2':
#             self.conv_list = nn.ModuleList([nn.Conv2d(in_dim, out_dim, kernel_size=1) for in_dim, out_dim in
#                                             zip([256, 512, 1024], cfg.model.out_dims)])  # [128, 256, 512, 1024]
#
#         # 锚点检测头
#         self.ancher = CenterHead(self.out_dims[0], num_anchors=10)
#         # 锚点注意力引导融合模块
#         self.aagf = AAGF(channels=self.out_dims[0], roi_size=7, use_confidence=True, use_attention=False,
#                          use_similarity=False)
#
#     def forward(self, rgb, tir):
#         feats_rgb = self.backbone(rgb)
#         feats_tir = self.backbone(tir)
#
#         # 舍弃第一层
#         feats_rgb = feats_rgb[1:]  # 舍弃第一层特征
#         feats_tir = feats_tir[1:]  # 舍弃第一层特征
#
#         # 统一通道
#         feats_rgb = [self.conv_list[i](feat) for i, feat in enumerate(feats_rgb)]
#         feats_tir = [self.conv_list[i](feat) for i, feat in enumerate(feats_tir)]
#
#         # 锚点检测头
#         ancher_rgb = [self.ancher(feat) for feat in feats_rgb]
#         ancher_tir = [self.ancher(feat) for feat in feats_tir]
#
#         # 锚点注意力引导融合模块
#         fused = [self.aagf(f_rgb, f_tir, a_rgb, a_tir) for f_rgb, f_tir, a_rgb, a_tir in zip(feats_rgb, feats_tir,  ancher_rgb, ancher_tir)]
#
#         return fused

class BackBones(nn.Module):

    def __init__(self, cfg, backbone_name='hgnetv2', out_dims=[512, 1024, 2048], roi_sizes=[14, 7, 3],
                 use_confidence=True, use_attention=False, use_similarity=False):
        super().__init__()
        self.backbone_name = backbone_name
        self.out_dims = out_dims
        self.roi_sizes = roi_sizes
        self.backbone = build_backbone(cfg)

        # # 锚点检测头
        # self.ancher = nn.ModuleList([CenterHead(dim, num_anchors=10) for dim in self.out_dims])
        # # 锚点注意力引导融合模块
        # self.aagf = nn.ModuleList([
        #     AAGF(in_channel=dim, roi_size=rs, use_confidence=use_confidence, use_attention=use_attention,
        #          use_similarity=use_similarity)
        #     for dim, rs in zip(self.out_dims, self.roi_sizes)
        # ])
        img_size = [80, 40, 20]
        self.fusion = nn.ModuleList([
            TFAM(size) for size in img_size
        ])

    def forward(self, rgb, tir):
        feats_rgb = self.backbone(rgb)
        feats_tir = self.backbone(tir)

        # 舍弃第一层
        feats_rgb = feats_rgb[1:]  # 舍弃第一层特征
        feats_tir = feats_tir[1:]  # 舍弃第一层特征

        # # 锚点检测头
        # ancher_rgb = [self.ancher[i](feat) for i, feat in enumerate(feats_rgb)]
        # ancher_tir = [self.ancher[i](feat) for i, feat in enumerate(feats_tir)]
        #
        # # 锚点注意力引导融合模块
        # fused = [self.aagf[i](f_rgb, f_tir, a_rgb, a_tir) for i, f_rgb, f_tir, a_rgb, a_tir in
        #          zip(range(len(self.out_dims)), feats_rgb, feats_tir, ancher_rgb, ancher_tir)]
        fused = [self.fusion[i](f_rgb, f_tir) for i, f_rgb, f_tir in
                 zip(range(len(self.out_dims)), feats_rgb, feats_tir)]

        return fused


if __name__ == '__main__':
    import torch
    from utils import load_config

    cfg = load_config('../../configs/config.yaml')

    # 示例输入：2张3通道512x512的图像
    rgb = torch.randn(2, 3, 512, 640)
    tir = torch.randn(2, 3, 512, 640)
    model = BackBones(cfg)
    feats = model(rgb, tir)
    for feat in feats:
        print(feat.shape)

    from thop import profile

    flops, params = profile(model, inputs=(rgb, tir))
    print(f"backbones FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
