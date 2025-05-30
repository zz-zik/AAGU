# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: enhanced_abam_criterion.py
@Time    : 2025/5/30
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 增强版锚框注意力引导融合损失 - 加入目标信息引导 (修复设备问题)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


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
                       device: torch.device, image_size: Tuple[int, int] = (640, 512)) -> List[torch.Tensor]:
    """
    根据目标信息创建目标区域掩码

    Args:
        targets: 目标信息列表，每个元素包含boxes和labels
        feature_sizes: 特征图尺寸列表 [(H1,W1), (H2,W2), ...]
        device: 目标设备
        image_size: 原始图像尺寸 (H, W)

    Returns:
        target_masks: 各尺度的目标掩码列表
    """
    batch_size = len(targets)
    target_masks = []

    for h, w in feature_sizes:
        # 创建该尺度的目标掩码
        batch_masks = []

        for batch_idx in range(batch_size):
            target = targets[batch_idx]

            # ====== 处理没有目标的情况 ======
            boxes = target.get('boxes')
            labels = target.get('labels')

            # 创建该样本的掩码，直接在目标设备上创建
            mask = torch.zeros(h, w, device=device, dtype=torch.float32)

            # 如果有目标框，则处理
            if boxes is not None and labels is not None and boxes.shape[0] > 0:
                # 确保boxes和labels在正确的设备上
                boxes = boxes.to(device)
                labels = labels.to(device)

                # 将归一化坐标转换为特征图坐标
                boxes_scaled = boxes.clone()
                boxes_scaled[:, [0, 2]] *= w  # x坐标缩放
                boxes_scaled[:, [1, 3]] *= h  # y坐标缩放

                for box, label in zip(boxes_scaled, labels):
                    x1, y1, x2, y2 = box.int()
                    x1, y1 = max(0, x1.item()), max(0, y1.item())
                    x2, y2 = min(w - 1, x2.item()), min(h - 1, y2.item())

                    if x2 > x1 and y2 > y1:
                        # 根据类别设置不同权重
                        weight = 1.0
                        if label.item() in [1, 2, 3]:  # 重要类别
                            weight = 1.5
                        elif label.item() in [4, 5]:  # 次要类别
                            weight = 0.8

                        mask[y1:y2 + 1, x1:x2 + 1] = weight

            batch_masks.append(mask)

        target_masks.append(torch.stack(batch_masks))  # [B, H, W]

    return target_masks


class ABAMAlignmentLoss(nn.Module):
    """增强版ABAM对齐损失函数 - 加入目标信息引导"""

    def __init__(self,
                 offset_weight=1.0,
                 confidence_weight=0.5,
                 alignment_weight=2.0,
                 balance_weight=0.3,
                 smoothness_weight=0.2,
                 target_guided_weight=1.5,  # 新增：目标引导权重
                 class_aware_weight=0.5,  # 新增：类别感知权重
                 target_alignment_ratio=0.8,
                 modality_bias=0.7,
                 use_target_guidance=True):
        """
        Args:
            target_guided_weight: 目标引导损失权重
            class_aware_weight: 类别感知损失权重
            use_target_guidance: 是否使用目标引导
        """
        super(ABAMAlignmentLoss, self).__init__()

        self.offset_weight = offset_weight
        self.confidence_weight = confidence_weight
        self.alignment_weight = alignment_weight
        self.balance_weight = balance_weight
        self.smoothness_weight = smoothness_weight
        self.target_guided_weight = target_guided_weight
        self.class_aware_weight = class_aware_weight
        self.target_alignment_ratio = target_alignment_ratio
        self.modality_bias = modality_bias
        self.use_target_guidance = use_target_guidance

        # 根据模态偏向确定目标权重比例
        self.target_tir_weight = modality_bias
        self.target_rgb_weight = 1.0 - modality_bias

        # Huber损失用于偏移量回归
        self.huber_loss = nn.SmoothL1Loss(reduction='none')

    def extract_feature_sizes(self, align_infos):
        """从align_infos中提取特征图尺寸"""
        feature_sizes = []
        for align_info in align_infos:
            # 从任一张量中获取空间尺寸，假设使用alignment_mask
            alignment_mask = align_info['alignment_mask']  # [B, num_anchors, H, W]
            H, W = alignment_mask.shape[-2:]
            feature_sizes.append((H, W))
        return feature_sizes

    def safe_get_device(self, align_infos):
        """安全获取设备信息"""
        for align_info in align_infos:
            for key, tensor in align_info.items():
                if isinstance(tensor, torch.Tensor):
                    return tensor.device
        return torch.device('cpu')

    def preprocess_targets(self, targets: List[Dict], device: torch.device) -> List[Dict]:
        """预处理targets，确保所有张量都在正确的设备上"""
        processed_targets = []

        for target in targets:
            processed_target = {}

            # 处理boxes
            boxes = target.get('boxes')
            if boxes is None or (isinstance(boxes, torch.Tensor) and boxes.shape[0] == 0):
                processed_target['boxes'] = torch.empty((0, 4), dtype=torch.float32, device=device)
            else:
                processed_target['boxes'] = boxes.to(device)

            # 处理labels
            labels = target.get('labels')
            if labels is None or (isinstance(labels, torch.Tensor) and labels.shape[0] == 0):
                processed_target['labels'] = torch.empty((0,), dtype=torch.int64, device=device)
            else:
                processed_target['labels'] = labels.to(device)

            # 复制其他字段
            for key, value in target.items():
                if key not in ['boxes', 'labels']:
                    processed_target[key] = value

            processed_targets.append(processed_target)

        return processed_targets

    def forward(self, align_infos, targets=None):
        """
        Args:
            align_infos: ABAM模块输出的对齐信息列表
            targets: 目标信息列表，包含boxes和labels (可选)

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失详情
        """
        total_loss = 0.0
        loss_dict = {}

        num_scales = len(align_infos)

        # 获取设备信息
        device = self.safe_get_device(align_infos)

        # 从align_infos中提取特征图尺寸
        feature_sizes = self.extract_feature_sizes(align_infos)

        # 创建目标掩码（如果提供了目标信息）
        target_masks = None
        processed_targets = None
        if self.use_target_guidance and targets is not None:
            # 预处理targets，处理空目标的情况，并确保设备一致性
            processed_targets = self.preprocess_targets(targets, device)
            target_masks = create_target_mask(processed_targets, feature_sizes, device)

        # 各项损失累加器
        offset_loss_total = 0.0
        confidence_loss_total = 0.0
        alignment_loss_total = 0.0
        balance_loss_total = 0.0
        smoothness_loss_total = 0.0
        target_guided_loss_total = 0.0
        class_aware_loss_total = 0.0

        for scale_idx, align_info in enumerate(align_infos):
            offsets = align_info['offsets']  # [B, num_anchors, 2, H, W]
            confidences = align_info['confidences']  # [B, num_anchors, 2, H, W]
            alignment_mask = align_info['alignment_mask']  # [B, num_anchors, H, W]
            rgb_weight = align_info['rgb_weight']  # [B, 1, H, W]
            tir_weight = align_info['tir_weight']  # [B, 1, H, W]

            # Convert alignment_mask to float if it's boolean
            if alignment_mask.dtype == torch.bool:
                alignment_mask = alignment_mask.float()

            # 获取当前尺度的目标掩码
            current_target_mask = None
            if target_masks is not None:
                current_target_mask = target_masks[scale_idx]  # [B, H, W]

            # 1. 目标引导的偏移量损失
            offset_magnitude = torch.sqrt(offsets[:, :, 0, :, :] ** 2 + offsets[:, :, 1, :, :] ** 2)

            if current_target_mask is not None:
                # 在目标区域应用更强的约束
                target_mask_expanded = current_target_mask.unsqueeze(1)  # [B, 1, H, W]
                target_weight = torch.where(target_mask_expanded > 0,
                                            target_mask_expanded * 2.0,  # 目标区域权重加倍
                                            torch.ones_like(target_mask_expanded) * 0.5)  # 背景区域降权

                weighted_offset_loss = offset_magnitude * target_weight
                offset_loss = torch.mean(weighted_offset_loss)
            else:
                # 原始偏移量损失计算
                if self.modality_bias > 0.5:
                    rgb_offset_penalty = offset_magnitude * (2 * self.modality_bias)
                    offset_loss = torch.mean(rgb_offset_penalty)
                elif self.modality_bias < 0.5:
                    tir_offset_penalty = offset_magnitude * (2 * (1 - self.modality_bias))
                    offset_loss = torch.mean(tir_offset_penalty)
                else:
                    offset_loss = torch.mean(offset_magnitude)

            offset_loss_total += offset_loss

            # 2. 目标引导的置信度损失
            rgb_conf = confidences[:, :, 0, :, :]  # [B, num_anchors, H, W]
            tir_conf = confidences[:, :, 1, :, :]  # [B, num_anchors, H, W]

            if current_target_mask is not None:
                # 在目标区域期望更高的置信度
                target_mask_expanded = current_target_mask.unsqueeze(1)  # [B, 1, H, W]

                # 目标区域的置信度应该更高
                target_conf_boost = 0.2 * target_mask_expanded
                target_tir_conf = torch.clamp(
                    torch.ones_like(tir_conf) * self.target_tir_weight + target_conf_boost,
                    0, 1)
                target_rgb_conf = torch.clamp(
                    torch.ones_like(rgb_conf) * self.target_rgb_weight + target_conf_boost,
                    0, 1)

                tir_conf_loss = F.mse_loss(tir_conf, target_tir_conf)
                rgb_conf_loss = F.mse_loss(rgb_conf, target_rgb_conf)
                confidence_loss = tir_conf_loss + rgb_conf_loss
            else:
                # 原始置信度损失计算
                if self.modality_bias > 0.5:
                    target_tir_conf = torch.ones_like(tir_conf) * self.target_tir_weight
                    target_rgb_conf = torch.ones_like(rgb_conf) * self.target_rgb_weight
                    tir_conf_loss = F.mse_loss(tir_conf, target_tir_conf)
                    rgb_conf_loss = F.mse_loss(rgb_conf, target_rgb_conf)
                    confidence_loss = tir_conf_loss + rgb_conf_loss
                elif self.modality_bias < 0.5:
                    target_rgb_conf = torch.ones_like(rgb_conf) * self.target_rgb_weight
                    target_tir_conf = torch.ones_like(tir_conf) * self.target_tir_weight
                    rgb_conf_loss = F.mse_loss(rgb_conf, target_rgb_conf)
                    tir_conf_loss = F.mse_loss(tir_conf, target_tir_conf)
                    confidence_loss = rgb_conf_loss + tir_conf_loss
                else:
                    confidence_diff = torch.abs(rgb_conf - tir_conf)
                    confidence_loss = torch.mean(confidence_diff)

            confidence_loss_total += confidence_loss

            # 3. 目标引导的对齐损失
            if current_target_mask is not None:
                # 目标区域应该有更高的对齐率
                target_alignment_mask = alignment_mask * current_target_mask.unsqueeze(1)
                target_alignment_ratio = torch.sum(target_alignment_mask) / (
                        torch.sum(current_target_mask.unsqueeze(1)) + 1e-7)

                # 目标区域对齐比例损失
                target_ratio_tensor = torch.tensor(self.target_alignment_ratio + 0.1,  # 目标区域期望更高对齐率
                                                   dtype=target_alignment_ratio.dtype,
                                                   device=target_alignment_ratio.device)
                target_guided_loss = F.mse_loss(target_alignment_ratio, target_ratio_tensor)
                target_guided_loss_total += target_guided_loss

            # 原始对齐损失
            current_alignment_ratio = torch.mean(alignment_mask)
            target_ratio_tensor = torch.tensor(self.target_alignment_ratio,
                                               dtype=current_alignment_ratio.dtype,
                                               device=current_alignment_ratio.device)
            alignment_ratio_loss = F.mse_loss(current_alignment_ratio, target_ratio_tensor)

            # 对齐掩码熵损失
            eps = 1e-7
            p = torch.clamp(alignment_mask, eps, 1 - eps)
            log_p = torch.log(p)
            log_1_minus_p = torch.log(1 - p)

            if torch.isnan(log_p).any() or torch.isnan(log_1_minus_p).any():
                alignment_entropy = torch.tensor(0.0, device=alignment_mask.device)
            else:
                alignment_entropy = -torch.mean(p * log_p + (1 - p) * log_1_minus_p)
                if torch.isnan(alignment_entropy):
                    alignment_entropy = torch.tensor(0.0, device=alignment_mask.device)

            alignment_loss = alignment_ratio_loss + 0.1 * alignment_entropy
            if torch.isnan(alignment_loss):
                alignment_loss = alignment_ratio_loss

            alignment_loss_total += alignment_loss

            # 4. 类别感知的权重平衡损失
            if current_target_mask is not None and processed_targets is not None:
                # 根据目标类别调整权重期望
                class_aware_loss = 0.0
                batch_size = current_target_mask.shape[0]

                for batch_idx in range(batch_size):
                    if batch_idx < len(processed_targets):
                        labels = processed_targets[batch_idx]['labels']

                        # 如果没有标签，跳过该batch
                        if labels.shape[0] == 0:
                            continue

                        unique_labels = torch.unique(labels)

                        for label in unique_labels:
                            # TODO:不同类别可能需要不同的模态偏向
                            if label.item() in [0, 1]:  # 人、车等热目标，TIR更重要
                                class_modality_bias = 0.8
                            elif label.item() in [2, 3, 4]:  # 建筑物等，RGB更重要
                                class_modality_bias = 0.3
                            else:  # 其他类别保持默认
                                class_modality_bias = self.modality_bias

                            class_target_tir = class_modality_bias
                            class_target_rgb = 1.0 - class_modality_bias

                            # 在该类别对应的区域计算权重损失
                            label_mask = (current_target_mask[batch_idx] > 0)
                            if torch.sum(label_mask) > 0:
                                tir_weight_in_region = tir_weight[batch_idx, 0][label_mask]
                                rgb_weight_in_region = rgb_weight[batch_idx, 0][label_mask]

                                class_tir_loss = F.mse_loss(tir_weight_in_region,
                                                            torch.full_like(tir_weight_in_region, class_target_tir))
                                class_rgb_loss = F.mse_loss(rgb_weight_in_region,
                                                            torch.full_like(rgb_weight_in_region, class_target_rgb))
                                class_aware_loss += class_tir_loss + class_rgb_loss

                class_aware_loss_total += class_aware_loss

            # 5. 原始模态权重平衡损失
            target_tir_weight_tensor = torch.ones_like(tir_weight) * self.target_tir_weight
            target_rgb_weight_tensor = torch.ones_like(rgb_weight) * self.target_rgb_weight

            tir_weight_loss = F.mse_loss(tir_weight, target_tir_weight_tensor)
            rgb_weight_loss = F.mse_loss(rgb_weight, target_rgb_weight_tensor)

            bias_strength = abs(self.modality_bias - 0.5) * 2

            if self.modality_bias > 0.5:
                weight_balance_loss = tir_weight_loss * (1 + bias_strength) + rgb_weight_loss
            elif self.modality_bias < 0.5:
                weight_balance_loss = rgb_weight_loss * (1 + bias_strength) + tir_weight_loss
            else:
                weight_balance_loss = tir_weight_loss + rgb_weight_loss

            balance_loss_total += weight_balance_loss

            # 6. 空间平滑损失
            alignment_mask_mean = torch.mean(alignment_mask, dim=1, keepdim=True)  # [B, 1, H, W]

            grad_h = torch.abs(alignment_mask_mean[:, :, :-1, :] - alignment_mask_mean[:, :, 1:, :])
            grad_v = torch.abs(alignment_mask_mean[:, :, :, :-1] - alignment_mask_mean[:, :, :, 1:])

            smoothness_loss = torch.mean(grad_h) + torch.mean(grad_v)
            smoothness_loss_total += smoothness_loss

        # 平均多尺度损失
        offset_loss_avg = offset_loss_total / num_scales
        confidence_loss_avg = confidence_loss_total / num_scales
        alignment_loss_avg = alignment_loss_total / num_scales
        balance_loss_avg = balance_loss_total / num_scales
        smoothness_loss_avg = smoothness_loss_total / num_scales
        target_guided_loss_avg = target_guided_loss_total / num_scales if target_guided_loss_total > 0 else torch.tensor(
            0.0, device=device)
        class_aware_loss_avg = class_aware_loss_total / num_scales if class_aware_loss_total > 0 else torch.tensor(0.0,
                                                                                                                   device=device)

        # 总损失加权组合
        total_loss = (self.offset_weight * offset_loss_avg +
                      self.confidence_weight * confidence_loss_avg +
                      self.alignment_weight * alignment_loss_avg +
                      self.balance_weight * balance_loss_avg +
                      self.smoothness_weight * smoothness_loss_avg)

        # 如果使用目标引导，添加额外损失项
        if self.use_target_guidance and target_guided_loss_avg > 0:
            total_loss += self.target_guided_weight * target_guided_loss_avg

        if class_aware_loss_avg > 0:
            total_loss += self.class_aware_weight * class_aware_loss_avg

        # 安全处理NaN
        if torch.isnan(total_loss):
            total_loss = torch.tensor(0.0, device=device)

        # 损失详情
        def safe_item(tensor):
            if torch.isnan(tensor) or torch.isinf(tensor):
                return float('nan')
            return tensor.item()

        loss_dict = {
            'total_loss': safe_item(total_loss),
            'offset_loss': safe_item(offset_loss_avg),
            'confidence_loss': safe_item(confidence_loss_avg),
            'alignment_loss': safe_item(alignment_loss_avg),
            'balance_loss': safe_item(balance_loss_avg),
            'smoothness_loss': safe_item(smoothness_loss_avg),
            'target_guided_loss': safe_item(target_guided_loss_avg),
            'class_aware_loss': safe_item(class_aware_loss_avg),
            'modality_bias': self.modality_bias,
            'target_tir_weight': self.target_tir_weight,
            'target_rgb_weight': self.target_rgb_weight
        }

        return total_loss, loss_dict


# 使用示例
if __name__ == "__main__":
    """使用示例"""

    # 创建模拟数据
    batch_size = 2
    num_anchors = 9

    # 模拟目标信息（包含空目标的情况）
    targets = [
        {
            "boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]]),  # 归一化坐标
            "labels": torch.tensor([0, 1])  # 人和车
        },
        {
            # ====== 模拟没有目标的情况 ======
            "boxes": None,  # 或者 torch.empty((0, 4), dtype=torch.float32)
            "labels": None  # 或者 torch.empty((0,), dtype=torch.int64)
        }
    ]

    # 额外测试：显式的空目标
    targets_with_empty = [
        {
            "boxes": torch.tensor([[0.2, 0.2, 0.4, 0.4]]),
            "labels": torch.tensor([2])
        },
        {
            "boxes": torch.empty((0, 4), dtype=torch.float32),  # 显式空张量
            "labels": torch.empty((0,), dtype=torch.int64)
        }
    ]

    # 特征图尺寸（现在不需要单独传入）
    feature_sizes = [(64, 80), (32, 40), (16, 20)]

    # 模拟对齐信息
    align_infos = []
    for scale in feature_sizes:
        H, W = scale

        alignment_mask_prob = torch.sigmoid(torch.randn(batch_size, num_anchors, H, W))
        alignment_mask_prob = torch.clamp(alignment_mask_prob, 0.1, 0.9)
        alignment_mask = (alignment_mask_prob > 0.5)

        rgb_weight = torch.sigmoid(torch.randn(batch_size, 1, H, W))
        tir_weight = 1.0 - rgb_weight

        align_info = {
            'offsets': torch.randn(batch_size, num_anchors, 2, H, W),
            'confidences': torch.sigmoid(torch.randn(batch_size, num_anchors, 2, H, W)),
            'alignment_mask': alignment_mask,
            'rgb_weight': rgb_weight,
            'tir_weight': tir_weight,
        }
        align_infos.append(align_info)

    # 测试优化后的损失函数（只需两个参数）
    print("=== 优化后的ABAM损失 (处理空目标情况) ===")
    enhanced_loss = ABAMAlignmentLoss(
        modality_bias=0.7,  # 偏向TIR
        use_target_guidance=True,
        target_guided_weight=1.5,
        class_aware_weight=0.5
    )

    # 测试包含空目标的情况
    total_loss, loss_dict = enhanced_loss(align_infos, targets)
    print(f"Total Loss (含空目标): {total_loss:.4f}")
    for key, value in loss_dict.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\n=== 测试显式空目标 ===")
    total_loss_empty, loss_dict_empty = enhanced_loss(align_infos, targets_with_empty)
    print(f"Total Loss (显式空目标): {total_loss_empty:.4f}")
    for key, value in loss_dict_empty.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\n=== 对比：不使用目标引导 ===")
    basic_loss = ABAMAlignmentLoss(
        modality_bias=0.7,
        use_target_guidance=False
    )

    # 同样只需要两个参数，targets可以为None
    total_loss_basic, loss_dict_basic = basic_loss(align_infos)
    print(f"Total Loss: {total_loss_basic:.4f}")
    for key, value in loss_dict_basic.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print(f"\n=== 改进效果 ===")
    print(f"目标引导损失改进: {loss_dict.get('target_guided_loss', 0):.4f}")
    print(f"类别感知损失改进: {loss_dict.get('class_aware_loss', 0):.4f}")
    print(f"总损失变化: {total_loss.item() - total_loss_basic.item():.4f}")
