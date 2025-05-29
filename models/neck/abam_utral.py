# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: abam_cafm_fusion.py
@Time    : 2025/5/29 下午12:38
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 锚框注意力引导融合模块 + CAFM融合 (Fixed Version)
@Usage   :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# CAFM模块相关组件
class GDM(nn.Module):
    def __init__(self, channels):
        super(GDM, self).__init__()
        self.convk1d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0,
                      groups=channels, bias=False)
        )
        self.convk3d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0),
                      groups=channels, bias=False)
        )
        self.convk5d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 5), stride=1, padding=(0, 2),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(5, 1), stride=1, padding=(2, 0),
                      groups=channels, bias=False)
        )
        self.convk7d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 7), stride=1, padding=(0, 3),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(7, 1), stride=1, padding=(3, 0),
                      groups=channels, bias=False)
        )

    def forward(self, x):
        c1 = self.convk1d1(x)
        c2 = self.convk3d1(x)
        c3 = self.convk5d1(x)
        c4 = self.convk7d1(x)
        return c1, c2, c3, c4


class GVM(nn.Module):
    def __init__(self, channels):
        super(GVM, self).__init__()
        self.channels = channels
        self.convk1d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0,
                      groups=channels, bias=False)
        )
        self.convk3d3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 3), dilation=(1, 3),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(3, 0), dilation=(3, 1),
                      groups=channels, bias=False)
        )
        self.convk3d5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 5), dilation=(1, 5),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(5, 0), dilation=(5, 1),
                      groups=channels, bias=False)
        )
        self.convk3d7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 7), dilation=(1, 7),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(7, 0), dilation=(7, 1),
                      groups=channels, bias=False)
        )
        self.conv1 = nn.Conv2d(channels * 4, channels, kernel_size=1, stride=1, padding=0, bias=False)

        # 固定输出通道数为原始通道数的1/3，确保整除
        self.output_channels = channels // 3
        self.convk1 = nn.Conv2d(channels, self.output_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, a0, a1, a2, a3, a):
        c1 = self.convk1d1(x)
        c5 = self.convk3d3(x)
        c6 = self.convk3d5(x)
        c7 = self.convk3d7(x)
        y1 = self.relu(c1 * a0 + c1 * a1 + c1 * a2 + c1 * a3)
        y2 = self.relu(c5 * a0 + c5 * a1 + c5 * a2 + c5 * a3)
        y3 = self.relu(c6 * a0 + c6 * a1 + c6 * a2 + c6 * a3)
        y4 = self.relu(c7 * a0 + c7 * a1 + c7 * a2 + c7 * a3)
        res = self.conv1(torch.cat((y1, y2, y3, y4), dim=1))
        a = a.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, res.shape[1], res.shape[2], res.shape[3])
        res = self.relu(res * a)
        return self.convk1(res)


class CAM(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CAM, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, out_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(fea))))
        out = max_out
        ress = self.sigmoid(out)
        return ress


class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv2d(out)
        out = self.sigmoid(out)
        return out


class CAFM(nn.Module):
    def __init__(self, channels):
        super(CAFM, self).__init__()
        # 确保输入通道数能被3整除
        if (channels * 3) % 3 != 0:
            raise ValueError(f"channels * 3 ({channels * 3}) must be divisible by 3")

        concat_channels = channels * 3
        self.ca = CAM(concat_channels, 1)
        self.sa = SAM()
        self.gdm = GDM(concat_channels)
        self.gvm = GVM(concat_channels)

        # GVM的输出通道数
        self.output_channels = self.gvm.output_channels

    def forward(self, x1, x2):
        out = torch.cat((x1, x2, x1 + x2), dim=1)
        a = self.ca(out).view(-1)
        out0, out1, out2, out3 = self.gdm(out)
        out = self.gvm(out, out0, out1, out2, out3, a)
        out = self.sa(out) * out
        return out


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

    def __init__(self, in_channels, num_anchors=9):
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
        self.alignment_threshold = 0.5

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
        alignment_mask = (offset_magnitude < self.alignment_threshold).float()  # [B, num_anchors, H, W]

        return offsets, confidences, alignment_mask


class ABAMWithCAFM(nn.Module):
    """整合CAFM的锚框注意力引导融合模块"""

    def __init__(self, in_channels, num_anchors=3, use_cafm=True):
        super(ABAMWithCAFM, self).__init__()
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.use_cafm = use_cafm

        # TFAM模块
        self.tfam = TFAM(in_channels)

        # 锚框对齐模块
        self.alignment_module = AnchorBoxAlignment(in_channels, num_anchors)

        # CAFM融合模块
        if self.use_cafm:
            self.cafm = CAFM(in_channels)
            # 获取CAFM的实际输出通道数并调整到原始通道数
            cafm_out_channels = self.cafm.output_channels
            self.cafm_adjust = nn.Sequential(
                nn.Conv2d(cafm_out_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

        # 简化的自适应融合
        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

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

        # 步骤3: CAFM增强融合（如果启用）
        if self.use_cafm:
            cafm_fused = self.cafm(rgb_feat, tir_feat)
            cafm_fused = self.cafm_adjust(cafm_fused)
        else:
            cafm_fused = initial_fused

        # 步骤4: 简化的自适应特征选择
        rgb_confidence = torch.mean(confidences[:, :, 0, :, :], dim=1, keepdim=True)  # [B, 1, H, W]
        tir_confidence = torch.mean(confidences[:, :, 1, :, :], dim=1, keepdim=True)  # [B, 1, H, W]
        alignment_score = torch.mean(alignment_mask.float(), dim=1, keepdim=True)  # [B, 1, H, W]

        # 基于对齐分数的特征选择
        confidence_mask = (rgb_confidence > tir_confidence).float()

        # 对齐良好时使用CAFM融合，对齐不良时根据置信度选择
        final_fused = torch.where(
            alignment_score > 0.5,
            cafm_fused,  # 对齐良好，使用CAFM融合特征
            torch.where(
                confidence_mask > 0.5,
                rgb_feat,  # RGB置信度高
                tir_feat  # TIR置信度高
            )
        )

        # 步骤5: 自适应融合refinement
        refined_feat = self.adaptive_fusion(final_fused)

        # 残差连接
        output_feat = refined_feat + self.residual_weight * initial_fused

        # 构建对齐信息
        alignment_info = {
            'offsets': offsets,
            'confidences': confidences,
            'alignment_mask': alignment_mask,
            'rgb_weight': rgb_weight,
            'tir_weight': tir_weight,
            'alignment_ratio': torch.mean(alignment_mask.float()).item(),
            'cafm_enabled': self.use_cafm
        }

        return output_feat, alignment_info


class MultiScaleABAM(nn.Module):
    """多尺度融合网络，支持ABAM和CAFM两种融合策略"""

    def __init__(self, in_channels_list=[512, 1024, 2048], num_anchors=3, fusion_type='abam_cafm'):
        """
        Args:
            in_channels_list: 各尺度输入通道数列表
            num_anchors: 锚框数量
            fusion_type: 融合类型 ['abam', 'cafm', 'abam_cafm']
        """
        super(MultiScaleABAM, self).__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'cafm':
            # 纯CAFM融合
            self.fusion_modules = nn.ModuleList([
                CAFM(in_channels) for in_channels in in_channels_list
            ])
            # 为CAFM添加调整层
            self.cafm_adjusts = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d((in_channels * 3) // 3, in_channels, 1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                ) for in_channels in in_channels_list
            ])
        elif fusion_type == 'abam_cafm':
            # ABAM+CAFM融合
            self.fusion_modules = nn.ModuleList([
                ABAMWithCAFM(in_channels, num_anchors, use_cafm=True)
                for in_channels in in_channels_list
            ])
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, rgb_feats, tir_feats):
        """
        Args:
            rgb_feats: RGB特征金字塔列表 [feat1, feat2, feat3]
            tir_feats: TIR特征金字塔列表 [feat1, feat2, feat3]

        Returns:
            fused_feats: 融合后的特征金字塔列表
            alignment_infos: 各尺度的对齐信息列表（仅ABAM相关方法返回）
        """
        fused_feats = []
        alignment_infos = []

        for i, fusion_module in enumerate(self.fusion_modules):
            if self.fusion_type == 'cafm':
                # CAFM只返回融合特征，需要调整通道数
                fused_feat = fusion_module(rgb_feats[i], tir_feats[i])
                fused_feat = self.cafm_adjusts[i](fused_feat)
                fused_feats.append(fused_feat)
                alignment_infos.append(None)
            else:
                # ABAM相关方法返回融合特征和对齐信息
                fused_feat, alignment_info = fusion_module(rgb_feats[i], tir_feats[i])
                fused_feats.append(fused_feat)
                alignment_infos.append(alignment_info)

        return fused_feats, alignment_infos


# 测试代码
if __name__ == "__main__":
    print("=== 测试ABAMWithCAFM模块 ===")

    # 创建模型
    model = ABAMWithCAFM(in_channels=512, num_anchors=9, use_cafm=True)

    # 创建测试数据
    rgb_feat = torch.randn(1, 512, 64, 80)
    tir_feat = torch.randn(1, 512, 64, 80)

    # 前向传播
    with torch.no_grad():
        fused_feat, alignment_info = model(rgb_feat, tir_feat)

    print(f"输入RGB特征形状: {rgb_feat.shape}")
    print(f"输入TIR特征形状: {tir_feat.shape}")
    print(f"输出融合特征形状: {fused_feat.shape}")
    print(f"对齐比例: {alignment_info['alignment_ratio']:.3f}")
    print(f"CAFM启用状态: {alignment_info['cafm_enabled']}")

    # 测试不同融合策略的多尺度版本
    print("\n=== 测试多尺度融合 ===")

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

    # 测试不同融合策略
    fusion_types = ['cafm', 'abam_cafm']

    for fusion_type in fusion_types:
        print(f"\n--- 测试{fusion_type.upper()}融合策略 ---")
        multi_scale_model = MultiScaleABAM(fusion_type=fusion_type)

        with torch.no_grad():
            fused_feats, alignment_infos = multi_scale_model(rgb_feats, tir_feats)

        for i, fused_feat in enumerate(fused_feats):
            if alignment_infos[i] is not None:
                print(f"尺度{i + 1} - 输出形状: {fused_feat.shape}, "
                      f"对齐比例: {alignment_infos[i]['alignment_ratio']:.3f}")
            else:
                print(f"尺度{i + 1} - 输出形状: {fused_feat.shape}")

        # 计算FLOPs和参数量
        try:
            from thop import profile

            flops, params = profile(multi_scale_model, inputs=(rgb_feats, tir_feats))
            print(f"{fusion_type.upper()} - FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
        except ImportError:
            print("thop未安装，跳过FLOPs计算")

    # 计算单个ABAM+CAFM模块的参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n=== ABAMWithCAFM模型信息 ===")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

    # 对比测试：启用和不启用CAFM的效果
    print(f"\n=== 对比测试：CAFM启用/禁用 ===")

    model_without_cafm = ABAMWithCAFM(in_channels=512, num_anchors=9, use_cafm=False)

    with torch.no_grad():
        fused_feat_with_cafm, _ = model(rgb_feat, tir_feat)
        fused_feat_without_cafm, _ = model_without_cafm(rgb_feat, tir_feat)

    print(f"启用CAFM输出形状: {fused_feat_with_cafm.shape}")
    print(f"禁用CAFM输出形状: {fused_feat_without_cafm.shape}")
    print(f"特征差异L2范数: {torch.norm(fused_feat_with_cafm - fused_feat_without_cafm).item():.6f}")