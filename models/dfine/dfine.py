"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

__all__ = [
    "DFINE",
]

from models import Fusion
from models.dfine import HybridEncoder, DFINETransformer


class DFINE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fusion = Fusion(
            backbone_name=cfg.model.backbone_name,
            name=cfg.model.backbone.name,
            use_lab=cfg.model.backbone.use_lab,
            img_size=cfg.model.img_size,
            return_idx=cfg.model.backbone.return_idx,
            pretrained=cfg.model.backbone.pretrained
        )
        # self.backbone = build_backbone(cfg)
        self.encoder = HybridEncoder(
            in_channels=cfg.model.encoder.in_channels,
            feat_strides=cfg.model.encoder.feat_strides,
            hidden_dim=cfg.model.encoder.hidden_dim,
            use_encoder_idx=cfg.model.encoder.use_encoder_idx,
            num_encoder_layers=cfg.model.encoder.num_encoder_layers,
            nhead=cfg.model.encoder.nhead,
            dim_feedforward=cfg.model.encoder.dim_feedforward,
            dropout=cfg.model.encoder.dropout,
            enc_act=cfg.model.encoder.enc_act,
            expansion=cfg.model.encoder.expansion,
            depth_mult=cfg.model.encoder.depth_mult,
            act=cfg.model.encoder.act,
        )
        self.decoder = DFINETransformer(
            num_classes=cfg.model.num_classes,
            hidden_dim=cfg.model.decoder.hidden_dim,
            num_queries=cfg.model.decoder.num_queries,
            feat_channels=cfg.model.decoder.feat_channels,
            feat_strides=cfg.model.decoder.feat_strides,
            num_levels=cfg.model.decoder.num_levels,
            num_points=cfg.model.decoder.num_points,
            num_layers=cfg.model.decoder.num_layers,
            num_denoising=cfg.model.decoder.num_denoising,
            label_noise_ratio=cfg.model.decoder.label_noise_ratio,
            box_noise_scale=cfg.model.decoder.box_noise_scale,
            eval_idx=cfg.model.decoder.eval_idx,
            cross_attn_method=cfg.model.decoder.cross_attn_method,
            query_select_method=cfg.model.decoder.query_select_method,
            reg_max=cfg.model.decoder.reg_max,
            reg_scale=cfg.model.decoder.reg_scale,
            layer_scale=cfg.model.decoder.layer_scale,
        )

    def forward(self, rgb, tir, targets=None):
        x = self.fusion(rgb, tir)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(
            self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


# 測試
if __name__ == "__main__":
    import torch
    from utils import load_config

    cfg = load_config('../../configs/config.yaml')

    # 示例输入：2张3通道512x512的图像
    rgb = torch.randn(2, 3, 512, 640)
    tir = torch.randn(2, 3, 512, 640)
    targets = [
    {
        "labels": torch.tensor([0, 0]),              # 第一张图的类别标签
        "boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]]),
    },
    {
        "labels": torch.tensor([0]),                 # 第二张图的类别标签
        "boxes": torch.tensor([[0.2, 0.2, 0.4, 0.4]]),
    }
    ]

    model = DFINE(cfg)
    output = model(rgb, tir, targets)
    # print(output)

    from thop import profile
    flops, params = profile(model, inputs=(rgb, tir))
    print(f"fusion FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
