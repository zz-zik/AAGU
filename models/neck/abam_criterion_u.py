# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: abam_criterion_u.py
@Time    : 2025/6/4
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 针对增强ABAM模型的RGB-TIR融合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """计算边界框面积"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """计算两组边界框的IoU"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    return inter / union


def create_target_mask(targets: List[Dict], feature_sizes: List[Tuple[int, int]],
                                device: torch.device, sigma=2.0) -> List[torch.Tensor]:
    """
    创建增强的目标区域掩码，包含边界关注和尺度感知

    Args:
        targets: 目标信息列表
        feature_sizes: 特征图尺寸列表
        device: 目标设备
        sigma: 高斯权重的标准差

    Returns:
        target_masks: 增强的目标掩码列表，包含边界权重
    """
    batch_size = len(targets)
    target_masks = []

    for scale_idx, (h, w) in enumerate(feature_sizes):
        batch_masks = []

        for batch_idx in range(batch_size):
            target = targets[batch_idx]
            boxes = target.get('boxes')
            labels = target.get('labels')

            # 创建基础掩码
            mask = torch.zeros(h, w, device=device, dtype=torch.float32)

            if boxes is not None and labels is not None and boxes.shape[0] > 0:
                boxes = boxes.to(device)
                labels = labels.to(device)

                # 将归一化坐标转换为特征图坐标
                boxes_scaled = boxes.clone()
                boxes_scaled[:, [0, 2]] *= w
                boxes_scaled[:, [1, 3]] *= h

                for box, label in zip(boxes_scaled, labels):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    # 创建以目标中心为核心的高斯权重
                    y_coords, x_coords = torch.meshgrid(
                        torch.arange(h, device=device, dtype=torch.float32),
                        torch.arange(w, device=device, dtype=torch.float32),
                        indexing='ij'
                    )

                    # 计算到目标中心的距离
                    dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
                    gaussian_weight = torch.exp(-dist_sq / (2 * sigma ** 2))

                    # 边界区域增强权重
                    boundary_weight = torch.ones_like(gaussian_weight)

                    # 目标框内部区域
                    x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
                    x2_int, y2_int = min(w - 1, int(x2)), min(h - 1, int(y2))

                    if x2_int > x1_int and y2_int > y1_int:
                        # 边界区域（外扩1-2像素）获得更高权重
                        boundary_margin = max(1, min(2, min(x2_int - x1_int, y2_int - y1_int) // 4))

                        x1_ext = max(0, x1_int - boundary_margin)
                        y1_ext = max(0, y1_int - boundary_margin)
                        x2_ext = min(w - 1, x2_int + boundary_margin)
                        y2_ext = min(h - 1, y2_int + boundary_margin)

                        # 边界区域权重提升
                        boundary_weight[y1_ext:y2_ext + 1, x1_ext:x2_ext + 1] = 1.5
                        # 目标内部核心区域
                        boundary_weight[y1_int:y2_int + 1, x1_int:x2_int + 1] = 2.0

                    # 根据类别和尺度调整权重
                    class_weight = 1.0
                    if label.item() in [0, 1]:  # 人、车等关键目标
                        class_weight = 1.5
                    elif label.item() in [2, 3]:  # 建筑物等大目标
                        class_weight = 1.2

                    # 小目标在高分辨率层获得更高权重
                    box_area_normalized = (x2 - x1) * (y2 - y1) / (h * w)
                    if box_area_normalized < 0.01 and scale_idx == 0:  # 小目标在高分辨率层
                        class_weight *= 1.3

                    # 合并权重
                    combined_weight = gaussian_weight * boundary_weight * class_weight
                    mask = torch.max(mask, combined_weight)

            batch_masks.append(mask)

        target_masks.append(torch.stack(batch_masks))

    return target_masks


class ABAMAlignmentLoss(nn.Module):

    def __init__(self,
                 # 核心融合损失权重
                 deformable_alignment_weight=3.0,  # 可变形对齐损失
                 boundary_enhancement_weight=2.5,  # 边界增强损失
                 complementary_fusion_weight=3.5,  # 互补融合损失

                 # IoU直接优化权重
                 fusion_quality_weight=4.0,  # 融合质量损失
                 alignment_confidence_weight=2.0,  # 对齐置信度损失

                 # 跨模态一致性权重
                 modal_consistency_weight=1.5,  # 模态一致性损失
                 feature_coherence_weight=1.0,  # 特征连贯性损失

                 # 辅助损失权重
                 spatial_smoothness_weight=0.5,  # 空间平滑性损失

                 # 目标引导权重
                 target_guided_weight=2.5,  # 目标引导权重
                 boundary_precision_weight=1.5,  # 边界精度权重

                 # 超参数
                 alignment_threshold=0.6,
                 use_target_guidance=True,
                 temperature=3.0):

        super(ABAMAlignmentLoss, self).__init__()

        # 保存权重参数
        self.deformable_alignment_weight = deformable_alignment_weight
        self.boundary_enhancement_weight = boundary_enhancement_weight
        self.complementary_fusion_weight = complementary_fusion_weight
        self.fusion_quality_weight = fusion_quality_weight
        self.alignment_confidence_weight = alignment_confidence_weight
        self.modal_consistency_weight = modal_consistency_weight
        self.feature_coherence_weight = feature_coherence_weight
        self.spatial_smoothness_weight = spatial_smoothness_weight
        self.target_guided_weight = target_guided_weight
        self.boundary_precision_weight = boundary_precision_weight

        self.alignment_threshold = alignment_threshold
        self.use_target_guidance = use_target_guidance
        self.temperature = temperature

        # 损失函数
        self.huber_loss = nn.SmoothL1Loss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def compute_deformable_alignment_loss(self, alignment_info, target_mask=None):
        """计算可变形对齐损失"""
        alignment_quality = alignment_info['alignment_quality']  # [B, 1, H, W]

        if target_mask is not None:
            # 在目标区域期望更高的对齐质量
            target_mask_expanded = target_mask.unsqueeze(1)  # [B, 1, H, W]

            # 动态设置目标质量
            high_quality_target = torch.ones_like(alignment_quality) * 0.9
            medium_quality_target = torch.ones_like(alignment_quality) * 0.7

            target_quality = torch.where(
                target_mask_expanded > 0.5,
                high_quality_target,
                medium_quality_target
            )

            # 加权损失 - 目标区域权重更高
            quality_loss = self.mse_loss(alignment_quality, target_quality)
            weighted_loss = quality_loss * (1.0 + 2.0 * target_mask_expanded)  # 目标区域权重x3

            return torch.mean(weighted_loss)
        else:
            # 整体对齐质量目标
            target_quality = torch.ones_like(alignment_quality) * 0.8
            return F.mse_loss(alignment_quality, target_quality)

    def compute_boundary_enhancement_loss(self, alignment_info, target_mask=None):
        """计算边界增强损失"""
        boundary_map = alignment_info['boundary_map']  # [B, 1, H, W]

        # 边界清晰度损失
        boundary_clarity_loss = -torch.mean(
            boundary_map * torch.log(boundary_map + 1e-8) +
            (1 - boundary_map) * torch.log(1 - boundary_map + 1e-8)
        )

        # 边界强度损失
        boundary_strength_loss = F.mse_loss(
            boundary_map.mean(),
            torch.tensor(0.4, device=boundary_map.device)  # 期望40%的区域有边界
        )

        # 目标边界增强
        if target_mask is not None:
            # 计算目标边界
            target_edges = self._compute_edge_map(target_mask)

            # 在目标边界处期望更强的边界响应
            boundary_precision_loss = F.mse_loss(
                boundary_map.squeeze(1) * target_edges,
                target_edges * 0.8  # 目标边界处期望0.8的响应
            )

            return boundary_clarity_loss + boundary_strength_loss + boundary_precision_loss

        return boundary_clarity_loss + boundary_strength_loss

    def compute_complementary_fusion_loss(self, alignment_info, target_mask=None):
        """计算互补融合损失"""
        rgb_weight = alignment_info['rgb_weight']  # [B, 1, H, W]
        tir_weight = alignment_info['tir_weight']  # [B, 1, H, W]

        # 权重归一化损失
        weight_sum = rgb_weight + tir_weight
        normalization_loss = F.mse_loss(weight_sum, torch.ones_like(weight_sum))

        # 模态平衡损失
        rgb_global_avg = torch.mean(rgb_weight)
        tir_global_avg = torch.mean(tir_weight)

        # 期望全局平衡在0.3-0.7之间
        balance_loss = F.relu(torch.abs(rgb_global_avg - 0.5) - 0.2) + \
                       F.relu(torch.abs(tir_global_avg - 0.5) - 0.2)

        # 融合多样性损失
        def compute_spatial_diversity(weight_map):
            # 计算权重的空间标准差
            spatial_std = torch.std(weight_map.view(weight_map.shape[0], -1), dim=1)
            return torch.mean(spatial_std)

        diversity_loss = -(compute_spatial_diversity(rgb_weight) + compute_spatial_diversity(tir_weight))

        # 目标区域的优化融合
        if target_mask is not None:
            target_mask_expanded = target_mask.unsqueeze(1)

            # 在目标区域，期望更加动态的融合权重
            target_fusion_variance = torch.var(rgb_weight * target_mask_expanded) + \
                                     torch.var(tir_weight * target_mask_expanded)

            # 鼓励目标区域有适度的权重变化
            target_fusion_loss = F.mse_loss(
                target_fusion_variance,
                torch.tensor(0.05, device=rgb_weight.device)
            )

            return normalization_loss + balance_loss + 0.5 * diversity_loss + target_fusion_loss

        return normalization_loss + balance_loss + 0.5 * diversity_loss

    def compute_fusion_quality_loss(self, alignment_info, target_mask=None):
        """计算融合质量损失"""
        alignment_quality = alignment_info['alignment_quality']  # [B, 1, H, W]
        boundary_map = alignment_info['boundary_map']  # [B, 1, H, W]
        rgb_weight = alignment_info['rgb_weight']  # [B, 1, H, W]
        tir_weight = alignment_info['tir_weight']  # [B, 1, H, W]

        # 高质量融合应该在边界区域有更好的对齐
        boundary_alignment_quality = alignment_quality * boundary_map
        quality_loss = F.mse_loss(
            boundary_alignment_quality.mean(),
            torch.tensor(0.7, device=alignment_quality.device)
        )

        # 融合一致性损失
        weight_quality_consistency = F.mse_loss(
            rgb_weight * alignment_quality,
            rgb_weight * 0.8
        ) + F.mse_loss(
            tir_weight * alignment_quality,
            tir_weight * 0.8
        )

        # 目标区域融合优化
        if target_mask is not None:
            target_mask_expanded = target_mask.unsqueeze(1)

            # 目标区域的融合质量应该更高
            target_quality = alignment_quality * target_mask_expanded
            target_quality_loss = F.mse_loss(
                target_quality,
                target_mask_expanded * 0.85  # 目标区域期望85%的质量
            )

            # 目标区域的权重分布应该更加合理
            target_weight_optimization = torch.mean(
                torch.abs(rgb_weight * target_mask_expanded - 0.5 * target_mask_expanded)
            ) + torch.mean(
                torch.abs(tir_weight * target_mask_expanded - 0.5 * target_mask_expanded)
            )

            return quality_loss + weight_quality_consistency + target_quality_loss + 0.5 * target_weight_optimization

        return quality_loss + weight_quality_consistency

    def compute_alignment_confidence_loss(self, alignment_info, target_mask=None):
        """计算对齐置信度损失"""
        confidence = alignment_info['confidence']  # [B, 1, H, W]

        # 置信度分布损失
        confidence_target = torch.ones_like(confidence) * self.alignment_threshold
        confidence_loss = F.mse_loss(confidence, confidence_target)

        # 置信度与对齐质量一致性
        alignment_quality = alignment_info['alignment_quality']
        consistency_loss = F.mse_loss(confidence, alignment_quality)

        # 目标区域置信度增强
        if target_mask is not None:
            target_mask_expanded = target_mask.unsqueeze(1)

            target_confidence_loss = F.mse_loss(
                confidence * target_mask_expanded,
                target_mask_expanded * 0.8
            )

            return confidence_loss + consistency_loss + target_confidence_loss

        return confidence_loss + consistency_loss

    def compute_modal_consistency_loss(self, alignment_info):
        """计算模态一致性损失"""
        rgb_weight = alignment_info['rgb_weight']
        tir_weight = alignment_info['tir_weight']

        # 权重互补性
        complementary_loss = F.mse_loss(rgb_weight + tir_weight, torch.ones_like(rgb_weight))

        # 空间连贯性
        def spatial_smoothness_loss(weight_map):
            diff_h = torch.abs(weight_map[:, :, 1:, :] - weight_map[:, :, :-1, :])
            diff_w = torch.abs(weight_map[:, :, :, 1:] - weight_map[:, :, :, :-1])
            return torch.mean(diff_h) + torch.mean(diff_w)

        smoothness_loss = spatial_smoothness_loss(rgb_weight) + spatial_smoothness_loss(tir_weight)

        # 全局模态平衡
        global_balance_loss = torch.abs(torch.mean(rgb_weight) - torch.mean(tir_weight))

        return complementary_loss + 0.2 * smoothness_loss + 0.3 * global_balance_loss

    def compute_feature_coherence_loss(self, alignment_info):
        """计算特征连贯性损失"""
        alignment_quality = alignment_info['alignment_quality']
        boundary_map = alignment_info['boundary_map']

        # 对齐质量与边界的相关性
        boundary_alignment_coherence = F.mse_loss(
            alignment_quality * boundary_map,
            boundary_map * 0.75  # 边界处期望较高的对齐质量
        )

        # 特征一致性
        rgb_weight = alignment_info['rgb_weight']
        tir_weight = alignment_info['tir_weight']

        # 高质量对齐区域的权重应该更加平衡
        high_quality_mask = (alignment_quality > 0.7).float()
        balanced_weight_in_hq = F.mse_loss(
            (rgb_weight * high_quality_mask).mean(),
            torch.tensor(0.5, device=rgb_weight.device)
        )

        return boundary_alignment_coherence + balanced_weight_in_hq

    def _compute_edge_map(self, mask):
        """计算边缘图"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=mask.dtype, device=mask.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=mask.dtype, device=mask.device).unsqueeze(0).unsqueeze(0)

        if len(mask.shape) == 3:  # [B, H, W]
            mask = mask.unsqueeze(1)  # [B, 1, H, W]

        edge_x = F.conv2d(mask, sobel_x, padding=1)
        edge_y = F.conv2d(mask, sobel_y, padding=1)

        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edges.squeeze(1)  # [B, H, W]

    def extract_feature_sizes(self, align_infos):
        """从对齐信息中提取特征图尺寸"""
        feature_sizes = []
        for align_info in align_infos:
            for key, tensor in align_info.items():
                if isinstance(tensor, torch.Tensor) and len(tensor.shape) >= 2:
                    H, W = tensor.shape[-2:]
                    feature_sizes.append((H, W))
                    break
        return feature_sizes

    def safe_get_device(self, align_infos):
        """安全获取设备信息"""
        for align_info in align_infos:
            for key, tensor in align_info.items():
                if isinstance(tensor, torch.Tensor):
                    return tensor.device
        return torch.device('cpu')

    def forward(self, align_infos, targets=None):
        """
        Args:
            align_infos: ABAM模块输出的对齐信息列表
            targets: 目标信息列表（可选）

        Returns:
            total_loss: 总损失
            loss_dict: 详细损失信息
        """
        total_loss = 0.0
        loss_dict = {}

        device = self.safe_get_device(align_infos)
        feature_sizes = self.extract_feature_sizes(align_infos)
        num_scales = len(align_infos)

        # 创建目标掩码
        target_masks = None
        if self.use_target_guidance and targets is not None:
            processed_targets = []
            for target in targets:
                processed_target = {}

                boxes = target.get('boxes')
                if boxes is None or (isinstance(boxes, torch.Tensor) and boxes.shape[0] == 0):
                    processed_target['boxes'] = torch.empty((0, 4), dtype=torch.float32, device=device)
                else:
                    processed_target['boxes'] = boxes.to(device)

                labels = target.get('labels')
                if labels is None or (isinstance(labels, torch.Tensor) and labels.shape[0] == 0):
                    processed_target['labels'] = torch.empty((0,), dtype=torch.int64, device=device)
                else:
                    processed_target['labels'] = labels.to(device)

                processed_targets.append(processed_target)

            target_masks = create_target_mask(processed_targets, feature_sizes, device)

        losses = {
            'deformable_alignment_loss': 0.0,
            'boundary_enhancement_loss': 0.0,
            'complementary_fusion_loss': 0.0,
            'fusion_quality_loss': 0.0,
            'alignment_confidence_loss': 0.0,
            'modal_consistency_loss': 0.0,
            'feature_coherence_loss': 0.0,
        }

        # 逐尺度计算损失
        for scale_idx, align_info in enumerate(align_infos):
            current_target_mask = target_masks[scale_idx] if target_masks is not None else None

            # 检查必要的key是否存在
            required_keys = ['alignment_quality', 'boundary_map', 'rgb_weight', 'tir_weight', 'confidence']
            missing_keys = [key for key in required_keys if key not in align_info]

            if missing_keys:
                print(f"Warning: Missing keys in alignment_info: {missing_keys}")
                continue

            # 可变形对齐损失
            deform_loss = self.compute_deformable_alignment_loss(align_info, current_target_mask)
            losses['deformable_alignment_loss'] += deform_loss

            # 边界增强损失
            boundary_loss = self.compute_boundary_enhancement_loss(align_info, current_target_mask)
            losses['boundary_enhancement_loss'] += boundary_loss

            # 互补融合损失
            fusion_loss = self.compute_complementary_fusion_loss(align_info, current_target_mask)
            losses['complementary_fusion_loss'] += fusion_loss

            # 融合质量损失
            quality_loss = self.compute_fusion_quality_loss(align_info, current_target_mask)
            losses['fusion_quality_loss'] += quality_loss

            # 对齐置信度损失
            confidence_loss = self.compute_alignment_confidence_loss(align_info, current_target_mask)
            losses['alignment_confidence_loss'] += confidence_loss

            # 模态一致性损失
            consistency_loss = self.compute_modal_consistency_loss(align_info)
            losses['modal_consistency_loss'] += consistency_loss

            # 特征连贯性损失
            coherence_loss = self.compute_feature_coherence_loss(align_info)
            losses['feature_coherence_loss'] += coherence_loss

        # 平均多尺度损失
        for key in losses:
            if num_scales > 0:
                losses[key] /= num_scales

        # 加权组合总损失
        total_loss = (
                self.deformable_alignment_weight * losses['deformable_alignment_loss'] +
                self.boundary_enhancement_weight * losses['boundary_enhancement_loss'] +
                self.complementary_fusion_weight * losses['complementary_fusion_loss'] +  # 最重要
                self.fusion_quality_weight * losses['fusion_quality_loss'] +  # IoU直接优化
                self.alignment_confidence_weight * losses['alignment_confidence_loss'] +
                self.modal_consistency_weight * losses['modal_consistency_loss'] +
                self.feature_coherence_weight * losses['feature_coherence_loss']
        )

        # 安全性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 构建详细损失字典
        def safe_item(tensor):
            if isinstance(tensor, torch.Tensor):
                if torch.isnan(tensor) or torch.isinf(tensor):
                    return 0.0
                return tensor.item()
            return float(tensor)

        loss_dict = {
            'total_loss': safe_item(total_loss),
            'deformable_alignment_loss': safe_item(losses['deformable_alignment_loss']),
            'boundary_enhancement_loss': safe_item(losses['boundary_enhancement_loss']),
            'complementary_fusion_loss': safe_item(losses['complementary_fusion_loss']),
            'fusion_quality_loss': safe_item(losses['fusion_quality_loss']),
            'alignment_confidence_loss': safe_item(losses['alignment_confidence_loss']),
            'modal_consistency_loss': safe_item(losses['modal_consistency_loss']),
            'feature_coherence_loss': safe_item(losses['feature_coherence_loss']),
        }

        return total_loss, loss_dict


# 示例
if __name__ == "__main__":
    import torch
    from models.neck.abam_utral import ABAM, MultiScaleABAM

    print("=== ABAMFusionLoss 完整测试 ===")

    criterion = ABAMAlignmentLoss(
            deformable_alignment_weight=3.0,
            boundary_enhancement_weight=2.5,
            complementary_fusion_weight=4.0,
            fusion_quality_weight=4.5,
            alignment_confidence_weight=2.0,
            modal_consistency_weight=1.5,
            feature_coherence_weight=1.0,
            alignment_threshold=0.65,
            use_target_guidance=True,
            temperature=3.0
    )

    model_single = ABAM(in_channels=512)
    rgb_feat_single = torch.randn(2, 512, 64, 80)
    tir_feat_single = torch.randn(2, 512, 64, 80)

    with torch.no_grad():
        feat_single, align_info_single = model_single(rgb_feat_single, tir_feat_single)

    targets = [
        {
            "boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]]),
            "labels": torch.tensor([0, 1])
        },
        {
            "boxes": torch.tensor([[0.2, 0.2, 0.6, 0.6]]),
            "labels": torch.tensor([2])
        }
    ]

    loss_single, loss_dict_single = criterion([align_info_single], targets)

    print("\n=== 单尺度损失结果 ===")
    for k, v in loss_dict_single.items():
        print(f"{k}: {v:.4f}")

    multi_model = MultiScaleABAM()
    rgb_feats_multi = [
        torch.randn(2, 512, 64, 80),
        torch.randn(2, 1024, 32, 40),
        torch.randn(2, 2048, 16, 20),
    ]
    tir_feats_multi = [
        torch.randn(2, 512, 64, 80),
        torch.randn(2, 1024, 32, 40),
        torch.randn(2, 2048, 16, 20),
    ]

    with torch.no_grad():
        feats_multi, align_infos_multi = multi_model(rgb_feats_multi, tir_feats_multi)

    loss_multi, loss_dict_multi = criterion(align_infos_multi, targets)

    print("\n=== 多尺度损失结果 ===")
    for k, v in loss_dict_multi.items():
        print(f"{k}: {v:.4f}")
