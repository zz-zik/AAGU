# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: resnet.py
@Time    : 2025/5/15 下午11:32
@Author  : ZhouFei
@Email   : zhoufei21@s.nuit.edu.cn
@Desc    : 
@Usage   :
"""
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

__all__ = ['ResNet50']


class ResNet50(nn.Module):
    def __init__(self, return_idx: list = [1, 2, 3], pretrained: bool=True):
        super().__init__()
        self.return_idx = return_idx
        # 使用ResNet50作为视觉编码器
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None
        self.backbone = resnet50(resnet50(weights=weights))
        # 获取ResNet50的各层输出维度
        self.out_dims = [64, 256, 512, 1024, 2048]

        # 定义各特征提取层
        self.layer0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

    def forward(self, x):
        """
        x: [B,3,H,W]
        returns list of multi-scale features:
          feats[i] is [B, C_i, H/2^{i}, W/2^{i}]
        """
        feats = []
        x = self.layer0(x)
        x = self.layer1(x)
        if 0 in self.return_idx:
            feats.append(x)  # Layer1 output - 1/4 resolution
        x = self.layer2(x)
        if 1 in self.return_idx:
            feats.append(x)  # Layer2 output - 1/8 resolution
        x = self.layer3(x)
        if 2 in self.return_idx:
            feats.append(x)  # Layer3 output - 1/16 resolution
        x = self.layer4(x)
        if 3 in self.return_idx:
            feats.append(x)  # Layer4 output - 1/32 resolution
        return feats


# 测试
if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 512, 640)  # 示例输入：2张3通道512x512的图像

    # 测试ResNet50
    model_resnet = ResNet50(return_idx=[1, 2, 3], pretrained=True)
    feats_resnet = model_resnet(x)
    print("\nResNet50 Features:")
    for i, feat in enumerate(feats_resnet):
        print(f"Layer {i} feature shape: {feat.shape}")

    # 计算FLOPs和Params
    from thop import profile

    flops_resnet, params_resnet = profile(model_resnet, inputs=(x,))
    print(f"ResNet50 Backbone FLOPs: {flops_resnet / 1e9:.2f} G, Params: {params_resnet / 1e6:.2f} M")
