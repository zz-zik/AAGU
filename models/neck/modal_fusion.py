# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: neck.py
@Time    : 2025/5/16 下午8:06
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
"""
from torch import nn
from models.neck import TFAM, MultiScaleABAM
from models import build_backbone

__all__ = ['Fusion']


class Fusion(nn.Module):

    def __init__(self, backbone_name: str='hgnetv2', name="B5", use_lab: bool=False, img_size: tuple = (512, 640),
                 return_idx:  list=[1, 2, 3], out_channels: list=[512, 1024, 2048], pretrained: bool=True):
        super().__init__()
        self.return_idx = return_idx
        self.backbone = build_backbone(
            backbone_name=backbone_name,
            name=name,
            use_lab=use_lab,
            img_size=img_size,
            return_idx=return_idx,
            pretrained=pretrained
        )

        # # 锚点检测头
        # self.ancher = nn.ModuleList([CenterHead(dim, num_anchors=10) for dim in self.out_dims])
        # # 锚点注意力引导融合模块
        # self.aagf = nn.ModuleList([
        #     AAGF(in_channel=dim, roi_size=rs, use_confidence=use_confidence, use_attention=use_attention,
        #          use_similarity=use_similarity)
        #     for dim, rs in zip(self.out_dims, self.roi_sizes)
        # ])

        # img_size = [80, 40, 20]
        # self.fusion = nn.ModuleList([
        #     TFAM(size) for size in img_size
        # ])

        self.abam = MultiScaleABAM(in_channels_list=out_channels, num_anchors=9)

        # self.cafm = MultiScaleABAM(in_channels_list=out_channels)

    def forward(self, rgb, tir):
        feats_rgb = self.backbone(rgb)
        feats_tir = self.backbone(tir)

        # # 锚点检测头
        # ancher_rgb = [self.ancher[i](feat) for i, feat in enumerate(feats_rgb)]
        # ancher_tir = [self.ancher[i](feat) for i, feat in enumerate(feats_tir)]
        # 锚点注意力引导融合模块
        # fused = [self.aagf[i](f_rgb, f_tir, a_rgb, a_tir) for i, f_rgb, f_tir, a_rgb, a_tir in
        #          zip(range(len(self.out_dims)), feats_rgb, feats_tir, ancher_rgb, ancher_tir)]

        # fused = [self.fusion[i](f_rgb, f_tir) for i, f_rgb, f_tir in
        #          zip(range(len(self.return_idx)), feats_rgb, feats_tir)]
        fused, _ = self.abam(feats_rgb, feats_tir)

        # fused = self.cafm(feats_rgb, feats_tir)
        return fused


if __name__ == '__main__':
    import torch
    from utils import load_config

    cfg = load_config('../../configs/config.yaml')

    # 示例输入：2张3通道512x512的图像
    rgb = torch.randn(2, 3, 512, 640)
    tir = torch.randn(2, 3, 512, 640)
    model = Fusion(
        backbone_name=cfg.model.backbone_name,
        name=cfg.model.backbone.name,
        use_lab=cfg.model.backbone.use_lab,
        img_size=cfg.model.img_size,
        return_idx=cfg.model.backbone.return_idx,
        out_channels=cfg.model.backbone.out_channels,
        pretrained=cfg.model.backbone.pretrained
    )
    feats = model(rgb, tir)
    for feat in feats:
        print(feat.shape)

    from thop import profile

    flops, params = profile(model, inputs=(rgb, tir))
    print(f"neck FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
