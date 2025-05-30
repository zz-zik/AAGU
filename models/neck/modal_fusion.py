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
                 return_idx:  list=[1, 2, 3], out_channels: list=[512, 1024, 2048], num_anchors: int=9,
                 align_thres: float=0.5, pretrained: bool=True):
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

        # img_size = [80, 40, 20]
        # self.fusion = nn.ModuleList([
        #     TFAM(size) for size in img_size
        # ])

        self.abam = MultiScaleABAM(
            in_channels_list=out_channels,
            num_anchors=num_anchors,
            align_thres=align_thres,
        )

        # self.cafm = MultiScaleABAM(in_channels_list=out_channels)

    def forward(self, rgb, tir):
        feats_rgb = self.backbone(rgb)
        feats_tir = self.backbone(tir)

        # fused = [self.fusion[i](f_rgb, f_tir) for i, f_rgb, f_tir in
        #          zip(range(len(self.return_idx)), feats_rgb, feats_tir)]

        fused, align_infos = self.abam(feats_rgb, feats_tir)

        # fused = self.cafm(feats_rgb, feats_tir)
        return fused, align_infos


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
        out_channels=cfg.model.abam.out_channels,
        num_anchors=cfg.model.abam.num_anchors,
        align_thres=cfg.model.abam.align_thres,
        pretrained=cfg.model.backbone.pretrained
    )
    feats = model(rgb, tir)
    for feat in feats:
        print(feat.shape)

    from thop import profile

    flops, params = profile(model, inputs=(rgb, tir))
    print(f"neck FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
