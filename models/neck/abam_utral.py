# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: lightweight_abam_cafm_fusion.py
@Time    : 2025/6/4
@Author  : ZhouFei (Optimized)
@Email   : zhoufei.net@gmail.com
@Desc    : 锚框注意力引导融合模块
@Usage   :
Enhanced ABAM模块 - 专注于RGB与TIR特征对齐和融合
主要改进：
1. 多尺度可变形对齐
2. 边界感知注意力
3. 互补性增强融合
4. 渐进式特征对齐
5. 丰富的对齐信息输出
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


class DeformableAlign(nn.Module):
    """轻量级可变形对齐模块"""

    def __init__(self, in_channels, reduction=4):
        super(DeformableAlign, self).__init__()
        self.in_channels = in_channels
        hidden_dim = in_channels // reduction

        # 简化的偏移预测 - 使用深度可分离卷积
        self.offset_pred = nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1, groups=in_channels * 2),
            # 点卷积
            nn.Conv2d(in_channels * 2, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 18, 1)  # 9个锚点 * 2个坐标
        )

        # 轻量级特征调制
        self.modulation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, rgb_feat, tir_feat):
        """
        Args:
            rgb_feat: RGB特征 [B, C, H, W]
            tir_feat: TIR特征 [B, C, H, W]
        Returns:
            aligned_rgb, aligned_tir, alignment_quality
        """
        B, C, H, W = rgb_feat.shape

        # 预测偏移
        concat_feat = torch.cat([rgb_feat, tir_feat], dim=1)
        offsets = self.offset_pred(concat_feat)  # [B, 18, H, W]
        offsets = offsets.view(B, 9, 2, H, W)  # [B, 9, 2, H, W]

        # 全局调制权重
        modulation_weight = self.modulation(concat_feat)  # [B, C, 1, 1]

        # 简化的网格采样对齐
        rgb_aligned = self._simple_grid_sample(rgb_feat, offsets[:, :4])  # 使用前4个锚点
        tir_aligned = self._simple_grid_sample(tir_feat, offsets[:, 4:8])  # 使用中间4个锚点

        # 应用调制
        rgb_aligned = rgb_aligned * modulation_weight
        tir_aligned = tir_aligned * modulation_weight

        # 简单的对齐质量评估
        alignment_quality = torch.mean(torch.abs(rgb_aligned - tir_aligned), dim=1, keepdim=True)
        alignment_quality = torch.sigmoid(-alignment_quality + 1)  # 转换为质量分数

        return rgb_aligned, tir_aligned, alignment_quality

    def _simple_grid_sample(self, feat, offsets):
        """简化的网格采样"""
        B, C, H, W = feat.shape
        N = offsets.shape[1]  # 锚点数量

        # 选择主要偏移（取均值）
        main_offset = torch.mean(offsets, dim=1)  # [B, 2, H, W]

        # 生成网格
        grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device),
            torch.linspace(-1, 1, W, device=feat.device),
            indexing='ij'
        )

        grid = torch.stack([grid_w, grid_h], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        # 应用偏移
        offset_norm = main_offset.permute(0, 2, 3, 1) * 0.1  # 缩放偏移
        sample_grid = grid + offset_norm

        # 网格采样
        aligned_feat = F.grid_sample(feat, sample_grid, mode='bilinear',
                                     padding_mode='border', align_corners=True)

        return aligned_feat


class EfficientBoundaryAttention(nn.Module):
    """高效边界注意力模块"""

    def __init__(self, in_channels, reduction=8):
        super(EfficientBoundaryAttention, self).__init__()
        hidden_dim = max(in_channels // reduction, 16)

        # 轻量级边界检测
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, groups=min(in_channels, 8)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )

        # 简化的注意力生成
        self.attention_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),  # 降低分辨率
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        """
        Args:
            feat: 输入特征 [B, C, H, W]
        Returns:
            enhanced_feat, boundary_map
        """
        # 边界检测
        boundary_map = self.boundary_detector(feat)

        # 生成注意力权重
        attention = self.attention_gen(feat)
        attention = F.interpolate(attention, size=feat.shape[-2:],
                                  mode='bilinear', align_corners=False)

        # 边界增强
        enhanced_feat = feat * (1 + attention * boundary_map)

        return enhanced_feat, boundary_map


class ComplementaryFusion(nn.Module):
    """轻量级互补融合模块"""

    def __init__(self, in_channels, reduction=4):
        super(ComplementaryFusion, self).__init__()
        hidden_dim = in_channels // reduction

        # 互补性分析 - 使用全局信息
        self.complementary_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 1),
            nn.Softmax(dim=1)
        )

        # 轻量级特征增强
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, tir_feat):
        """
        Args:
            rgb_feat: RGB特征 [B, C, H, W]
            tir_feat: TIR特征 [B, C, H, W]
        Returns:
            fused_feat, fusion_weights
        """
        B, C, H, W = rgb_feat.shape

        # 分析互补性
        concat_feat = torch.cat([rgb_feat, tir_feat], dim=1)
        weights = self.complementary_analyzer(concat_feat)  # [B, 2, 1, 1]
        weights = weights.expand(-1, -1, H, W)  # [B, 2, H, W]

        # 特征增强
        rgb_enhanced = self.feature_enhancer(rgb_feat)
        tir_enhanced = self.feature_enhancer(tir_feat)

        # 加权融合
        fused_feat = rgb_enhanced * weights[:, 0:1] + tir_enhanced * weights[:, 1:2]

        return fused_feat, weights


class ABAM(nn.Module):
    """轻量化ABAM模块"""

    def __init__(self, in_channels, num_anchors=9, align_thres=0.6, reduction=4):
        super(ABAM, self).__init__()
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.align_thres = align_thres

        # 1. 轻量级可变形对齐
        self.deformable_alignment = DeformableAlign(in_channels, reduction)

        # 2. 高效边界注意力
        self.boundary_attention = EfficientBoundaryAttention(in_channels, reduction * 2)

        # 3. 轻量级互补融合
        self.complementary_fusion = ComplementaryFusion(in_channels, reduction)

        # 4. 简化的锚框回归
        hidden_dim = in_channels // reduction
        self.anchor_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),  # 降低分辨率
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_anchors * 2, 1)  # 只预测x,y偏移
        )

        # 5. 对齐置信度
        self.confidence_pred = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )

        # 6. 最终输出调整
        self.output_adjust = nn.Conv2d(in_channels, in_channels, 1)

        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, rgb_feat, tir_feat):
        """
        Args:
            rgb_feat: RGB特征图 [B, C, H, W]
            tir_feat: TIR特征图 [B, C, H, W]
        Returns:
            enhanced_feat: 增强融合特征
            alignment_info: 对齐信息
        """
        B, C, H, W = rgb_feat.shape

        # Stage 1: 可变形对齐
        rgb_aligned, tir_aligned, align_quality = self.deformable_alignment(rgb_feat, tir_feat)

        # Stage 2: 边界增强（只对融合后的特征进行）
        fused_temp = (rgb_aligned + tir_aligned) * 0.5
        boundary_enhanced, boundary_map = self.boundary_attention(fused_temp)

        # Stage 3: 互补融合
        complementary_fused, fusion_weights = self.complementary_fusion(rgb_aligned, tir_aligned)

        # Stage 4: 特征融合
        enhanced_feat = self.fusion_weight * boundary_enhanced + (1 - self.fusion_weight) * complementary_fused
        enhanced_feat = self.output_adjust(enhanced_feat)

        # Stage 5: 锚框和置信度预测
        anchor_offsets = self.anchor_regressor(enhanced_feat)  # [B, num_anchors*2, 4, 4]
        anchor_offsets = F.interpolate(anchor_offsets, size=(H, W), mode='bilinear', align_corners=False)
        anchor_offsets = anchor_offsets.view(B, self.num_anchors, 2, H, W)

        confidence = self.confidence_pred(enhanced_feat)  # [B, 1, H, W]

        # 构建轻量级对齐信息
        alignment_info = {
            # 核心信息
            'rgb_weight': fusion_weights[:, 0:1],
            'tir_weight': fusion_weights[:, 1:2],
            'alignment_quality': align_quality,
            'boundary_map': boundary_map,
            'confidence': confidence,

            # 兼容原接口
            'offsets': anchor_offsets,
            'confidences': confidence.unsqueeze(2).expand(-1, self.num_anchors, 2, -1, -1),
            'alignment_mask': (confidence > self.align_thres).float(),
            'alignment_ratio': torch.mean((confidence > self.align_thres).float()).item(),

            # 统计信息
            'overall_alignment_score': torch.mean(align_quality).item(),
            'fusion_balance': torch.std(fusion_weights).item(),
        }

        return enhanced_feat, alignment_info


class AdaptiveABAM(nn.Module):
    """自适应ABAM - 根据输入动态调整计算量"""

    def __init__(self, in_channels, num_anchors=9, align_thres=0.6):
        super(AdaptiveABAM, self).__init__()

        # 基础模块
        self.abam = ABAM(in_channels, num_anchors, align_thres, reduction=8)

        # 复杂度评估器
        self.complexity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

        # 可选的增强模块（仅在需要时使用）
        self.enhanced_processor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.complexity_threshold = 0.7

    def forward(self, rgb_feat, tir_feat):
        """自适应处理"""
        # 评估复杂度
        concat_feat = torch.cat([rgb_feat, tir_feat], dim=1)
        complexity_score = self.complexity_estimator(concat_feat)

        # 基础处理
        enhanced_feat, alignment_info = self.abam(rgb_feat, tir_feat)

        # 根据复杂度决定是否进行额外处理
        if complexity_score.mean() > self.complexity_threshold:
            enhanced_feat = self.enhanced_processor(enhanced_feat) + enhanced_feat
            alignment_info['enhanced_processing'] = True
        else:
            alignment_info['enhanced_processing'] = False

        alignment_info['complexity_score'] = complexity_score.mean().item()

        return enhanced_feat, alignment_info


class MultiScaleABAM(nn.Module):
    """多尺度ABAM"""

    def __init__(self, in_channels_list=[512, 1024, 2048], num_anchors=9, align_thres=0.6):
        super(MultiScaleABAM, self).__init__()

        # 为每个尺度创建模块
        self.abam_modules = nn.ModuleList([
            ABAM(in_channels, num_anchors, align_thres, reduction=8)
            for in_channels in in_channels_list
        ])

        # 轻量级跨尺度连接
        self.cross_scale_adapters = nn.ModuleList()
        for i in range(len(in_channels_list) - 1):
            adapter = nn.Conv2d(in_channels_list[i], in_channels_list[i + 1], 1)
            self.cross_scale_adapters.append(adapter)

    def forward(self, rgb_feats, tir_feats):
        """
        Args:
            rgb_feats: RGB特征金字塔列表
            tir_feats: TIR特征金字塔列表
        Returns:
            enhanced_feats: 增强融合特征列表
            alignment_infos: 各尺度对齐信息
        """
        enhanced_feats = []
        alignment_infos = []

        prev_enhanced = None

        for i, abam_module in enumerate(self.abam_modules):
            current_rgb = rgb_feats[i]
            current_tir = tir_feats[i]

            # 跨尺度信息传播（简化版）
            if prev_enhanced is not None and i > 0:
                prev_adapted = self.cross_scale_adapters[i - 1](prev_enhanced)
                prev_upsampled = F.interpolate(
                    prev_adapted, size=current_rgb.shape[-2:],
                    mode='bilinear', align_corners=False
                )
                # 简单的加权融合
                current_rgb = current_rgb + 0.1 * prev_upsampled

            # ABAM处理
            enhanced_feat, alignment_info = abam_module(current_rgb, current_tir)

            enhanced_feats.append(enhanced_feat)
            alignment_infos.append(alignment_info)

            prev_enhanced = enhanced_feat

        return enhanced_feats, alignment_infos


# 参数量对比和测试
def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    """计算模型大小（MB）"""
    return count_parameters(model) * 4 / 1024 / 1024  # float32


if __name__ == "__main__":
    print("=== ABAM模块测试 ===")

    # 创建测试数据
    B, C, H, W = 2, 512, 64, 80
    rgb_feat = torch.randn(B, C, H, W)
    tir_feat = torch.randn(B, C, H, W)

    print(f"输入特征形状: RGB {rgb_feat.shape}, TIR {tir_feat.shape}")

    # 测试各个版本
    models = {
        "ABAM": ABAM(C),
        "AdaptiveABAM": AdaptiveABAM(C),
    }

    print(f"\n{'模型':<25} {'参数量':<12} {'大小(MB)':<10} {'推理时间(ms)':<12}")
    print("-" * 65)

    import time

    for name, model in models.items():
        model.eval()

        # 参数统计
        params = count_parameters(model)
        size_mb = model_size_mb(model)

        # 推理时间测试
        with torch.no_grad():
            start_time = time.time()
            for _ in range(10):
                enhanced_feat, alignment_info = model(rgb_feat, tir_feat)
            avg_time = (time.time() - start_time) / 10 * 1000  # ms

        print(f"{name:<25} {params:<12,} {size_mb:<10.2f} {avg_time:<12.1f}")

        # 输出信息
        print(f"  输出形状: {enhanced_feat.shape}")
        print(f"  对齐分数: {alignment_info['overall_alignment_score']:.4f}")
        if 'complexity_score' in alignment_info:
            print(f"  复杂度分数: {alignment_info['complexity_score']:.4f}")
        print()

    # 测试多尺度版本
    print("=== 多尺度ABAM测试 ===")
    multi_model = MultiScaleABAM()

    rgb_feats = [
        torch.randn(2, 512, 64, 80),
        torch.randn(2, 1024, 32, 40),
        torch.randn(2, 2048, 16, 20),
    ]

    tir_feats = [
        torch.randn(2, 512, 64, 80),
        torch.randn(2, 1024, 32, 40),
        torch.randn(2, 2048, 16, 20),
    ]

    with torch.no_grad():
        enhanced_feats, alignment_infos = multi_model(rgb_feats, tir_feats)

    multi_params = count_parameters(multi_model)
    multi_size = model_size_mb(multi_model)

    print(f"多尺度模型参数量: {multi_params:,}")
    print(f"多尺度模型大小: {multi_size:.2f} MB")

    for i, (feat, info) in enumerate(zip(enhanced_feats, alignment_infos)):
        print(f"尺度{i + 1}: 特征{feat.shape}, 对齐分数{info['overall_alignment_score']:.4f}")