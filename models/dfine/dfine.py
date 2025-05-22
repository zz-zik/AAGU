"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

__all__ = [
    "DFINE",
]

from models import BackBones
from models.backbones.backbone import build_backbone
from models.dfine import HybridEncoder, DFINETransformer


class DFINE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = BackBones(cfg, cfg.model.backbone_name, cfg.model.out_dims, cfg.model.aagf.roi_sizes,
                                  cfg.model.aagf.use_confidence, cfg.model.aagf.use_attention, cfg.model.aagf.use_similarity)
        # self.backbone = build_backbone(cfg)
        self.encoder = HybridEncoder(in_channels=cfg.model.out_dims, feat_strides=[8, 16, 32])
        self.decoder = DFINETransformer(num_classes=cfg.model.num_classes, feat_channels=[256, 256, 256])

    def forward(self, rgb, tir, targets=None):
        x = self.backbone(rgb, tir)
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
    print(output)

