# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: aagf.py
@Time    : 2025/5/16 下午2:03
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 锚点注意力引导融合模块
@Usage   : 该方法通过先分别在RGB和TIR的特征图上生成目标的中心锚点，然后模型根据锚点在哪个特征图上，选择哪个特征图的这部分特征，分别在RGB和TIR选择含有目标的特征后进行注意力融合。
feat_rgb ──┐
            ├─ Global Attention Fusion ─── fused_global ──┐
feat_tir ──┘                                              │
                                                          ├─ Final Output
                                                Local Anchor-based Fusion (ROI)
"""
"""
Anchor Attention Guided Fusion Module (AAGFM)
基于TFAM的锚点注意力引导融合模块
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.neck import TFAM


class AnchorDetector(nn.Module):
    """目标中心锚点检测器"""

    def __init__(self, in_channel, num_anchors=16, anchor_size=7):
        super().__init__()
        self.num_anchors = num_anchors
        self.anchor_size = anchor_size

        # 特征压缩和锚点预测
        self.feature_compress = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, 3, padding=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 4, 1, 1),
            nn.Sigmoid()
        )

        # 自适应池化用于生成固定数量的锚点
        self.adaptive_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, feat):
        """
        生成目标中心锚点
        Args:
            feat: 输入特征图 [B, C, H, W]
        Returns:
            anchor_map: 锚点热力图 [B, 1, H, W]
            anchor_coords: 锚点坐标 [B, num_anchors, 2]
        """
        B, C, H, W = feat.shape

        # 生成锚点热力图
        anchor_map = self.feature_compress(feat)  # [B, 1, H, W]

        # 提取top-k个锚点坐标
        anchor_scores = anchor_map.view(B, -1)  # [B, H*W]
        _, top_indices = torch.topk(anchor_scores, self.num_anchors, dim=1)  # [B, num_anchors]

        # 转换为坐标
        anchor_coords = torch.stack([
            top_indices % W,  # x坐标
            top_indices // W  # y坐标
        ], dim=-1).float()  # [B, num_anchors, 2]

        return anchor_map, anchor_coords


class LocalAttentionFusion(nn.Module):
    """局部锚点注意力融合"""

    def __init__(self, in_channel, roi_size=7):
        super().__init__()
        self.roi_size = roi_size
        self.in_channel = in_channel

        # 局部特征融合网络
        self.local_fusion = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        # 权重生成网络
        self.weight_generator = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 4, 2, 1),
            nn.Softmax(dim=1)
        )

    def extract_roi_features(self, feat, coords, roi_size):
        """从特征图中提取ROI特征"""
        B, C, H, W = feat.shape
        num_anchors = coords.shape[1]

        # 创建ROI特征容器
        roi_features = torch.zeros(B, num_anchors, C, roi_size, roi_size,
                                   device=feat.device, dtype=feat.dtype)

        for b in range(B):
            for i in range(num_anchors):
                x, y = coords[b, i].long()

                # 计算ROI边界
                x_start = max(0, x - roi_size // 2)
                x_end = min(W, x + roi_size // 2 + 1)
                y_start = max(0, y - roi_size // 2)
                y_end = min(H, y + roi_size // 2 + 1)

                # 提取ROI并调整大小
                roi = feat[b:b + 1, :, y_start:y_end, x_start:x_end]
                roi_resized = F.interpolate(roi, size=(roi_size, roi_size),
                                            mode='bilinear', align_corners=False)
                roi_features[b, i] = roi_resized[0]

        return roi_features

    def forward(self, feat_rgb, feat_tir, anchor_coords):
        """
        局部锚点注意力融合
        Args:
            feat_rgb: RGB特征图 [B, C, H, W]
            feat_tir: TIR特征图 [B, C, H, W]
            anchor_coords: 锚点坐标 [B, num_anchors, 2]
        Returns:
            fused_local: 融合后的局部特征
        """
        B, C, H, W = feat_rgb.shape
        num_anchors = anchor_coords.shape[1]

        # 提取ROI特征
        roi_rgb = self.extract_roi_features(feat_rgb, anchor_coords, self.roi_size)
        roi_tir = self.extract_roi_features(feat_tir, anchor_coords, self.roi_size)

        # 融合局部特征
        fused_rois = []
        for i in range(num_anchors):
            # 拼接RGB和TIR的ROI特征
            concat_roi = torch.cat([roi_rgb[:, i], roi_tir[:, i]], dim=1)  # [B, 2C, roi_size, roi_size]

            # 局部融合
            fused_roi = self.local_fusion(concat_roi)  # [B, C, roi_size, roi_size]

            # 生成融合权重
            weights = self.weight_generator(fused_roi)  # [B, 2, roi_size, roi_size]

            # 加权融合
            weighted_fusion = (weights[:, 0:1] * roi_rgb[:, i] +
                               weights[:, 1:2] * roi_tir[:, i])

            fused_rois.append(weighted_fusion)

        return torch.stack(fused_rois, dim=1)  # [B, num_anchors, C, roi_size, roi_size]


class AAGF(nn.Module):
    """锚点注意力引导融合模块"""

    def __init__(self, in_channel, num_anchors=16, roi_size=7, iterations=2):
        super().__init__()
        self.num_anchors = num_anchors
        self.roi_size = roi_size
        self.iterations = iterations
        self.in_channel = in_channel

        # 全局注意力融合 (基于TFAM)
        self.global_fusion = TFAM(in_channel)

        # 锚点检测器
        self.anchor_detector_rgb = AnchorDetector(in_channel, num_anchors, roi_size)
        self.anchor_detector_tir = AnchorDetector(in_channel, num_anchors, roi_size)

        # 局部注意力融合
        self.local_fusion = LocalAttentionFusion(in_channel, roi_size)

        # 特征重建网络
        self.feature_reconstruct = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, 1),
            nn.BatchNorm2d(in_channel)
        )

        # 迭代权重更新网络
        self.weight_update = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 4, 1, 1),
            nn.Sigmoid()
        )

    def integrate_local_to_global(self, global_feat, local_features, anchor_coords):
        """将局部融合特征整合回全局特征图"""
        B, C, H, W = global_feat.shape
        num_anchors = anchor_coords.shape[1]

        updated_feat = global_feat.clone()

        for b in range(B):
            for i in range(num_anchors):
                x, y = anchor_coords[b, i].long()

                # 计算ROI边界
                x_start = max(0, x - self.roi_size // 2)
                x_end = min(W, x + self.roi_size // 2 + 1)
                y_start = max(0, y - self.roi_size // 2)
                y_end = min(H, y + self.roi_size // 2 + 1)

                # 调整局部特征大小以匹配ROI区域
                local_feat = local_features[b, i]  # [C, roi_size, roi_size]
                roi_h, roi_w = y_end - y_start, x_end - x_start

                if roi_h != self.roi_size or roi_w != self.roi_size:
                    local_feat = F.interpolate(
                        local_feat.unsqueeze(0),
                        size=(roi_h, roi_w),
                        mode='bilinear',
                        align_corners=False
                    )[0]

                # 更新全局特征图的对应区域
                updated_feat[b, :, y_start:y_end, x_start:x_end] = local_feat

        return updated_feat

    def forward(self, feat_rgb, feat_tir):
        """
        前向传播
        Args:
            feat_rgb: RGB特征图 [B, C, H, W]
            feat_tir: TIR特征图 [B, C, H, W]
        Returns:
            final_output: 最终融合输出 [B, C, H, W]
        """
        # 1. 全局注意力融合
        fused_global = self.global_fusion(feat_rgb, feat_tir)

        # 2. 迭代锚点注意力融合
        current_rgb = feat_rgb
        current_tir = feat_tir
        current_global = fused_global

        for iteration in range(self.iterations):
            # 2.1 生成锚点
            anchor_map_rgb, anchor_coords_rgb = self.anchor_detector_rgb(current_rgb)
            anchor_map_tir, anchor_coords_tir = self.anchor_detector_tir(current_tir)

            # 2.2 选择更强的锚点特征
            # 比较RGB和TIR的锚点强度，选择更强的进行融合
            rgb_strength = torch.mean(anchor_map_rgb.view(anchor_map_rgb.shape[0], -1), dim=1)
            tir_strength = torch.mean(anchor_map_tir.view(anchor_map_tir.shape[0], -1), dim=1)

            # 使用更强特征图的锚点坐标
            use_rgb_anchors = rgb_strength > tir_strength
            anchor_coords = torch.where(
                use_rgb_anchors.unsqueeze(-1).unsqueeze(-1),
                anchor_coords_rgb,
                anchor_coords_tir
            )

            # 2.3 局部锚点注意力融合
            fused_local = self.local_fusion(current_rgb, current_tir, anchor_coords)

            # 2.4 将局部特征整合回全局特征图
            updated_rgb = self.integrate_local_to_global(current_rgb, fused_local, anchor_coords)
            updated_tir = self.integrate_local_to_global(current_tir, fused_local, anchor_coords)

            # 2.5 更新迭代特征
            if iteration < self.iterations - 1:  # 不是最后一次迭代
                current_rgb = updated_rgb
                current_tir = updated_tir
                # 重新计算全局融合
                current_global = self.global_fusion(current_rgb, current_tir)

        # 3. 最终特征融合
        final_concat = torch.cat([current_global, updated_rgb], dim=1)
        final_output = self.feature_reconstruct(final_concat)

        # 4. 残差连接
        final_output = final_output + current_global

        return final_output


if __name__ == '__main__':
    from thop import profile
    import time

    # 测试模块
    print("Testing AAGFM...")
    time_stamp = time.time()

    # 创建模型
    model = AAGF(in_channel=256, num_anchors=16, roi_size=7, iterations=2)

    # 创建测试数据
    feat_rgb = torch.randn(2, 256, 128, 160)
    feat_tir = torch.randn(2, 256, 128, 160)

    # 前向传播
    output = model(feat_rgb, feat_tir)
    print(f"Input shape: RGB {feat_rgb.shape}, TIR {feat_tir.shape}")
    print(f"Output shape: {output.shape}")

    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(feat_rgb, feat_tir))
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
    print(f"Time elapsed: {time.time() - time_stamp:.4f}s")

    # 测试各个组件
    print("\nTesting components:")

    # 测试锚点检测器
    anchor_detector = AnchorDetector(256, 16, 7)
    anchor_map, anchor_coords = anchor_detector(feat_rgb)
    print(f"Anchor map shape: {anchor_map.shape}")
    print(f"Anchor coords shape: {anchor_coords.shape}")

    # 测试局部融合
    local_fusion = LocalAttentionFusion(256, 7)
    local_features = local_fusion(feat_rgb, feat_tir, anchor_coords)
    print(f"Local features shape: {local_features.shape}")

