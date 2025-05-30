# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: abam.py
@Time    : 2025/5/29 下午12:38
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 锚框注意力引导融合模块
@Usage   : 
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """轻量级通道注意力模块"""

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 使用更小的隐藏层
        hidden_channels = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class SpatialAttention(nn.Module):
    """轻量级空间注意力模块"""

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


class TFAM(nn.Module):
    """轻量级时序融合注意力模块（TFAM）"""

    def __init__(self, in_channels):
        super(TFAM, self).__init__()
        self.in_channels = in_channels

        # 使用深度可分离卷积减少参数
        self.rgb_transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),  # 深度卷积
            nn.Conv2d(in_channels, in_channels, 1, bias=False),  # 点卷积
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.tir_transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 轻量级注意力
        self.channel_attention = ChannelAttention(in_channels, reduction=16)
        self.spatial_attention = SpatialAttention()

        # 简化的融合权重学习
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, rgb_feat, tir_feat):
        # 特征变换
        rgb_transformed = self.rgb_transform(rgb_feat)
        tir_transformed = self.tir_transform(tir_feat)

        # 简单的加权融合
        fusion_weights = self.weight_predictor(rgb_transformed + tir_transformed)
        rgb_weight = fusion_weights[:, 0:1, :, :]
        tir_weight = fusion_weights[:, 1:2, :, :]

        # 加权融合
        fused_feat = rgb_transformed * rgb_weight + tir_transformed * tir_weight

        # 应用注意力机制
        fused_feat = self.channel_attention(fused_feat)
        fused_feat = self.spatial_attention(fused_feat)

        return fused_feat, rgb_weight, tir_weight


class AnchorBoxAlignment(nn.Module):
    """轻量级锚框对齐模块"""

    def __init__(self, in_channels, num_anchors=9, align_thres=0.5):
        super(AnchorBoxAlignment, self).__init__()
        self.num_anchors = num_anchors

        # 共享的特征提取器
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        # 轻量级偏移量预测
        self.offset_predictor = nn.Conv2d(in_channels // 2, num_anchors * 2, 1)

        # 轻量级置信度预测
        self.confidence_predictor = nn.Sequential(
            nn.Conv2d(in_channels // 2, num_anchors * 2, 1),
            nn.Sigmoid()
        )

        # 对齐阈值
        self.align_thres = align_thres

    def forward(self, rgb_feat, tir_feat):
        concat_feat = torch.cat([rgb_feat, tir_feat], dim=1)
        B, _, H, W = concat_feat.shape

        # 共享特征提取
        shared_feat = self.shared_conv(concat_feat)

        # 预测偏移量和置信度
        offsets = self.offset_predictor(shared_feat)  # [B, num_anchors*2, H, W]
        confidences = self.confidence_predictor(shared_feat)  # [B, num_anchors*2, H, W]

        # 重塑维度
        offsets = offsets.view(B, self.num_anchors, 2, H, W)  # [B, num_anchors, 2, H, W]
        confidences = confidences.view(B, self.num_anchors, 2, H, W)  # [B, num_anchors, 2, H, W]

        # 计算偏移量的幅度
        offset_magnitude = torch.sqrt(
            offsets[:, :, 0, :, :] ** 2 + offsets[:, :, 1, :, :] ** 2)  # [B, num_anchors, H, W]

        # 创建对齐掩码（偏移量小于阈值认为对齐良好）
        alignment_mask = (offset_magnitude < self.align_thres).float()  # [B, num_anchors, H, W]

        return offsets, confidences, alignment_mask


class ABAM(nn.Module):
    """轻量级锚框注意力引导融合模块"""

    def __init__(self, in_channels, num_anchors=9, align_thres=0.5):  # 减少锚框数量
        super(ABAM, self).__init__()
        self.in_channels = in_channels
        self.num_anchors = num_anchors

        # TFAM模块
        self.tfam = TFAM(in_channels)

        # 锚框对齐模块
        self.alignment_module = AnchorBoxAlignment(in_channels, num_anchors, align_thres)

        # 简化的自适应融合
        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

        # 对齐阈值
        self.align_thres = align_thres

    def forward(self, rgb_feat, tir_feat):
        """
        Args:
            rgb_feat: RGB特征图 [B, C, H, W]
            tir_feat: TIR特征图 [B, C, H, W]

        Returns:
            fused_feat: 融合后的特征图 [B, C, H, W]
            alignment_info: 对齐信息字典
        """
        B, C, H, W = rgb_feat.shape

        # 步骤1: 通过TFAM模块进行初始特征融合
        initial_fused, rgb_weight, tir_weight = self.tfam(rgb_feat, tir_feat)

        # 步骤2: 锚框对齐分析
        offsets, confidences, alignment_mask = self.alignment_module(rgb_feat, tir_feat)

        # 步骤3: 简化的自适应特征选择
        rgb_confidence = torch.mean(confidences[:, :, 0, :, :], dim=1, keepdim=True)  # [B, 1, H, W]
        tir_confidence = torch.mean(confidences[:, :, 1, :, :], dim=1, keepdim=True)  # [B, 1, H, W]
        alignment_score = torch.mean(alignment_mask.float(), dim=1, keepdim=True)  # [B, 1, H, W]

        # 基于对齐分数的特征选择
        confidence_mask = (rgb_confidence > tir_confidence).float()

        # 对齐良好时直接融合，对齐不良时根据置信度选择
        final_fused = torch.where(
            alignment_score > self.align_thres,
            initial_fused,  # 对齐良好，使用融合特征
            torch.where(
                confidence_mask > self.align_thres,
                rgb_feat,  # RGB置信度高
                tir_feat  # TIR置信度高
            )
        )

        # 步骤4: 自适应融合refinement
        refined_feat = self.adaptive_fusion(final_fused)

        # 残差连接
        output_feat = refined_feat + self.residual_weight * initial_fused

        # 构建对齐信息
        align_info = {
            'offsets': offsets,
            'confidences': confidences,
            'alignment_mask': alignment_mask,
            'rgb_weight': rgb_weight,
            'tir_weight': tir_weight,
            'alignment_ratio': torch.mean(alignment_mask.float()).item()
        }

        return output_feat, align_info


# class MultiScaleABAM(nn.Module):
#     """轻量级多尺度ABAM网络"""
#
#     def __init__(self, in_channels_list=[512, 1024, 2048], num_anchors=9):
#         super(MultiScaleABAM, self).__init__()
#         self.abam_modules = nn.ModuleList([
#             ABAM(in_channels, num_anchors) for in_channels in in_channels_list
#         ])
#
#         # 简化的特征金字塔融合
#         self.fpn_fusion = nn.ModuleList([
#             nn.Conv2d(in_channels, 256, 1, bias=False) for in_channels in in_channels_list
#         ])
#
#     def forward(self, rgb_feats, tir_feats):
#         """
#         Args:
#             rgb_feats: RGB特征金字塔列表 [feat1, feat2, feat3]
#             tir_feats: TIR特征金字塔列表 [feat1, feat2, feat3]
#
#         Returns:
#             fused_feats: 融合后的特征金字塔列表
#             alignment_infos: 各尺度的对齐信息列表
#         """
#         fused_feats = []
#         alignment_infos = []
#
#         for i, (abam_module, fpn_conv) in enumerate(zip(self.abam_modules, self.fpn_fusion)):
#             # ABAM融合
#             fused_feat, alignment_info = abam_module(rgb_feats[i], tir_feats[i])
#
#             # FPN标准化
#             fused_feat = fpn_conv(fused_feat)
#
#             fused_feats.append(fused_feat)
#             alignment_infos.append(alignment_info)
#
#         return fused_feats, alignment_infos

class MultiScaleABAM(nn.Module):
    """轻量级多尺度ABAM网络"""

    def __init__(self, in_channels_list=[512, 1024, 2048], num_anchors=9, align_thres=0.5):
        super(MultiScaleABAM, self).__init__()
        self.abam_modules = nn.ModuleList([
            ABAM(in_channels, num_anchors, align_thres) for in_channels in in_channels_list
        ])

    def forward(self, rgb_feats, tir_feats):
        """
        Args:
            rgb_feats: RGB特征金字塔列表 [feat1, feat2, feat3]
            tir_feats: TIR特征金字塔列表 [feat1, feat2, feat3]

        Returns:
            fused_feats: 融合后的特征金字塔列表
            alignment_infos: 各尺度的对齐信息列表
        """
        fused_feats = []
        align_infos = []

        for i, abam_module in enumerate(self.abam_modules):
            # ABAM融合
            fused_feat, align_info = abam_module(rgb_feats[i], tir_feats[i])

            fused_feats.append(fused_feat)
            align_infos.append(align_info)

        return fused_feats, align_infos


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = ABAM(in_channels=512, num_anchors=9)

    # 创建测试数据
    rgb_feat = torch.randn(1, 512, 64, 80)
    tir_feat = torch.randn(1, 512, 64, 80)

    # 前向传播
    with torch.no_grad():
        fused_feat, alignment_info = model(rgb_feat, tir_feat)

    print("=== ABAM模块测试结果 ===")
    print(f"输入RGB特征形状: {rgb_feat.shape}")
    print(f"输入TIR特征形状: {tir_feat.shape}")
    print(f"输出融合特征形状: {fused_feat.shape}")
    print(f"对齐比例: {alignment_info['alignment_ratio']:.3f}")
    print(f"偏移量形状: {alignment_info['offsets'].shape}")
    print(f"置信度形状: {alignment_info['confidences'].shape}")

    # 测试多尺度版本
    print("\n=== 多尺度ABAM测试 ===")
    multi_scale_model = MultiScaleABAM()

    from thop import profile

    flops, params = profile(model, inputs=(rgb_feat, tir_feat))
    print(f"neck FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

    # 创建多尺度测试数据
    rgb_feats = [
        torch.randn(1, 512, 64, 80),  # 尺度1
        torch.randn(1, 1024, 32, 40),  # 尺度2
        torch.randn(1, 2048, 16, 20),  # 尺度3
    ]

    tir_feats = [
        torch.randn(1, 512, 64, 80),  # 尺度1
        torch.randn(1, 1024, 32, 40),  # 尺度2
        torch.randn(1, 2048, 16, 20),  # 尺度3
    ]

    with torch.no_grad():
        fused_feats, alignment_infos = multi_scale_model(rgb_feats, tir_feats)

    for i, (fused_feat, alignment_info) in enumerate(zip(fused_feats, alignment_infos)):
        print(f"尺度{i + 1} - 输出形状: {fused_feat.shape}, 对齐比例: {alignment_info['alignment_ratio']:.3f}")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n=== 模型信息 ===")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

    flops, params = profile(multi_scale_model, inputs=(rgb_feats, tir_feats))
    print(f"neck FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
