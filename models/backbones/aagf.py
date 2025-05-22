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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

__all__ = ['AAGF']


class LocalFeatureFusion(nn.Module):
    """对两个模态的局部特征进行注意力加权融合"""

    def __init__(self, channels, roi_size=7):
        super().__init__()
        self.att_conv = nn.Conv2d(channels * 2, 2, kernel_size=1)
        self.roi_size = roi_size

    def forward(self, feat_rgb, feat_tir):
        # feat_rgb/tir: (N, C, roi_size, roi_size)
        concat_feat = torch.cat([feat_rgb, feat_tir], dim=1)  # (N, 2C, r, r)
        attention_weights = F.softmax(self.att_conv(concat_feat), dim=1)  # (N, 2, r, r)

        # 加权融合
        fused_feat = feat_rgb * attention_weights[:, 0:1] + feat_tir * attention_weights[:, 1:2]
        return fused_feat


class GlobalFeatureFusion(nn.Module):
    """
    全局特征融合
    """
    def __init__(self, channels):
        super().__init__()
        self.fuser = LocalFeatureFusion(channels, roi_size=1)  # ROI 大小为 1x1

    def forward(self, feat_rgb, feat_tir):
        """
        Args:
            feat_rgb: (B, C, H, W)
            feat_tir: (B, C, H, W)
        Returns:
            fused_feat: (B, C, H, W)
        """
        B, C, H, W = feat_rgb.shape

        # 使用 reshape 避免 view 报错
        feat_rgb_flat = feat_rgb.reshape(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        feat_tir_flat = feat_tir.reshape(B, C, H * W).transpose(1, 2)

        # 变成 (B*H*W, C, 1, 1)
        feat_rgb_flat = feat_rgb_flat.reshape(B * H * W, C, 1, 1)
        feat_tir_flat = feat_tir_flat.reshape(B * H * W, C, 1, 1)

        # 全局注意力融合
        fused_flat = self.fuser(feat_rgb_flat, feat_tir_flat)  # (B*H*W, C, 1, 1)

        # 恢复成原始形状
        fused = fused_flat.reshape(B, H * W, C).transpose(1, 2)  # (B, C, H*W)
        fused = fused.reshape(B, C, H, W)  # (B, C, H, W)

        return fused


class AAGF(nn.Module):
    def __init__(self, channels, roi_size=7,
                 use_confidence=False,
                 use_similarity=False,
                 use_attention=True):
        super().__init__()
        self.channels = channels
        self.roi_size = roi_size
        self.use_confidence = use_confidence
        self.use_similarity = use_similarity
        self.use_attention = use_attention

        if use_attention:
            self.att_conv = nn.Conv2d(channels * 2, 2, kernel_size=1)

        self.global_fuser = GlobalFeatureFusion(channels=self.channels)

        # 使用 RoIAlign 替代手工裁剪
        self.roi_align = RoIAlign(output_size=(roi_size, roi_size), spatial_scale=1.0, sampling_ratio=-1)

    def extract_roi_features(self, features, anchors):
        """
        使用 RoIAlign 提取 ROI 特征
        Args:
            features: (B, C, H, W)
            anchors: (B, N, 2)  [x, y]
        Returns:
            rois: (B*N, C, roi_size, roi_size)
        """
        B, C, H, W = features.shape
        device = features.device
        anchors = anchors.to(device)

        # 转换为 [x_center, y_center] -> [x1, y1, x2, y2]
        anchors = anchors.float()
        half_size = self.roi_size / 2
        boxes = torch.stack([
            anchors[..., 0] - half_size,
            anchors[..., 1] - half_size,
            anchors[..., 0] + half_size,
            anchors[..., 1] + half_size,
        ], dim=-1)  # shape: (B, N, 4)

        # 添加 batch index
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, anchors.shape[1], -1)
        rois_input = torch.cat([batch_indices, boxes], dim=-1).view(-1, 5)  # shape: (B*N, 5)

        # 使用 RoIAlign 提取特征
        rois = self.roi_align(features, rois_input)
        return rois  # shape: (B*N, C, r, r)

    def select_by_confidence(self, rgb_rois, tir_rois, conf_rgb, conf_tir):
        """置信度比较
        Args:
            rgb_rois: (N, C, r, r)
            tir_rois: (N, C, r, r)
            conf_rgb: (B, N) 或 (N,)
            conf_tir: (B, N) 或 (N,)
        Returns:
            selected_rois: (N, C, r, r)
        """
        B, N = conf_rgb.shape
        conf_rgb = conf_rgb.view(-1)
        conf_tir = conf_tir.view(-1)

        mask = conf_rgb > conf_tir
        mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (N, 1, 1, 1)
        return torch.where(mask, rgb_rois, tir_rois)

    def select_by_similarity(self, rgb_rois, tir_rois):
        """
        根据特征相似度选择更清晰的一侧
        """
        norm_rgb = F.normalize(rgb_rois.view(rgb_rois.size(0), -1), dim=1)
        norm_tir = F.normalize(tir_rois.view(tir_rois.size(0), -1), dim=1)
        sim = torch.sum(norm_rgb * norm_tir, dim=1)  # (N,)
        mask = sim.unsqueeze(1).unsqueeze(2).unsqueeze(3) > 0.5
        return torch.where(mask, rgb_rois, tir_rois)

    def attention_fuse(self, rgb_rois, tir_rois):
        """
        根据注意力权重进行融合
        """
        concat_feat = torch.cat([rgb_rois, tir_rois], dim=1)  # (N, 2C, r, r)
        attention_weights = F.softmax(self.att_conv(concat_feat), dim=1)  # (N, 2, r, r)
        fused_rois = rgb_rois * attention_weights[:, 0:1] + tir_rois * attention_weights[:, 1:2]
        return fused_rois

    def fuse_rois_to_map(self, base_feat, rois, anchors):
        """
        使用掩码加权融合 ROI 到全局特征图
        Args:
            base_feat: 基础特征图，可以是全局融合后的特征 (B, C, H, W)
            rois: (N, C, r, r)
            anchors: (B, N, 2)
        Returns:
            fused_feat: (B, C, H, W)
        """
        B, C, H, W = base_feat.shape
        fused_map = base_feat.clone()

        # 构造 ROI 坐标
        anchors = anchors.to(base_feat.device).float()
        half_size = self.roi_size / 2
        x0 = (anchors[..., 0] - half_size).long().clamp(0, W - self.roi_size)
        y0 = (anchors[..., 1] - half_size).long().clamp(0, H - self.roi_size)

        idx = 0
        for b in range(B):
            for i in range(anchors.shape[1]):
                x_start, y_start = x0[b, i], y0[b, i]
                x_end, y_end = x_start + self.roi_size, y_start + self.roi_size
                fused_map[b, :, y_start:y_end, x_start:x_end] = rois[idx].detach()  # detach 避免保留图
                idx += 1

        return fused_map

    def forward(self, feat_rgb, feat_tir, anchors_rgb_with_conf, anchors_tir_with_conf):
        """
        Args:
            feat_rgb: RGB 特征图，(B, C, H, W)
            feat_tir: TIR 特征图，(B, C, H, W)
            anchors_rgb_with_conf: Tensor，形状 (B, N, 3)，最后一位是 confidence
            anchors_tir_with_conf: Tensor，形状 (B, N, 3)
        Returns:
            fused_feat: 融合后的特征图，(B, C, H, W)
        """
        # Step 0: 分离坐标和置信度
        anchors_rgb = anchors_rgb_with_conf[..., :2]  # (B, N, 2)
        conf_rgb = anchors_rgb_with_conf[..., 2]  # (B, N)

        anchors_tir = anchors_tir_with_conf[..., :2]
        conf_tir = anchors_tir_with_conf[..., 2]

        # Step 1: 全局融合（所有区域）
        fused_global = self.global_fuser(feat_rgb, feat_tir)  # (B, C, H, W)

        # Step 2: 提取 ROI 并局部融合
        rgb_rois = self.extract_roi_features(feat_rgb, anchors_rgb)
        tir_rois = self.extract_roi_features(feat_tir, anchors_tir)

        if rgb_rois.size(0) == 0 or tir_rois.size(0) == 0:
            return fused_global  # 如果没有 ROI，直接返回全局融合结果

        # Step 3: 使用置信度、相似度或注意力进行局部融合
        if self.use_confidence:
            fused_rois = self.select_by_confidence(rgb_rois, tir_rois, conf_rgb, conf_tir)
        elif self.use_similarity:
            fused_rois = self.select_by_similarity(rgb_rois, tir_rois)
        elif self.use_attention:
            fused_rois = self.attention_fuse(rgb_rois, tir_rois)
        else:
            fused_rois = (rgb_rois + tir_rois) / 2  # 默认平均融合

        # Step 4: 映射回原图（基于全局融合结果进行覆盖）
        fused_feat = self.fuse_rois_to_map(fused_global, fused_rois, anchors_rgb)

        return fused_feat


if __name__ == '__main__':
    x_rgb = torch.randn(2, 256, 128, 160)
    x_tir = torch.randn(2, 256, 128, 160)

    # 构造带置信度的 anchor 输入 (B, N, 3)
    anchors_rgb_with_conf = torch.tensor([
        [[100, 100, 0.9], [200, 200, 0.7], [120, 120, 0.6]],
        [[150, 150, 0.8], [160, 160, 0.75], [170, 170, 0.65]]
    ], dtype=torch.float32)

    anchors_tir_with_conf = torch.tensor([
        [[95, 95, 0.6], [205, 205, 0.8], [115, 115, 0.5]],
        [[145, 145, 0.7], [165, 165, 0.6], [175, 175, 0.85]]
    ], dtype=torch.float32)

    print("Anchors shape:", anchors_rgb_with_conf.shape)  # (2, 3, 3)

    model = AAGF(
        channels=256,
        roi_size=7,
        use_confidence=True,
        use_attention=True,
        use_similarity=True
    )

    output = model(x_rgb, x_tir, anchors_rgb_with_conf, anchors_tir_with_conf)
    print(output.shape)  # torch.Size([2, 256, 128, 160])

    from thop import profile

    flops, params = profile(model, inputs=(x_rgb, x_tir, anchors_rgb_with_conf, anchors_tir_with_conf))
    print(f"encoder FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
