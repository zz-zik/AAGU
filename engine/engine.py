# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: train_engine.py
@Time    : 2025/5/21 下午4:01
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 训练引擎
@Usage   : pycocotools
"""
import math
import sys

import numpy as np
import torch
from tqdm import tqdm
from models.dfine.box_ops import box_cxcywh_to_xyxy


def scale_boxes(boxes, orig_shape, resized_shape):
    """
    boxes in format: [x1, y1, x2, y2], absolute values
    orig_shape: [height, width]
    resized_shape: [height, width]
    """
    scale_x = orig_shape[1] / resized_shape[1]
    scale_y = orig_shape[0] / resized_shape[0]
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y
    return boxes


def compute_iou(box1, box2):
    """计算两个box的IoU，输入格式为xyxy"""
    # box1: [N, 4], box2: [M, 4]
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]

    # 计算交集
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


def compute_ap_single_class(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """计算单个类别的AP"""
    if len(pred_boxes) == 0:
        return 0.0 if len(gt_boxes) > 0 else 1.0

    if len(gt_boxes) == 0:
        return 0.0

    # 按置信度排序
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    # 计算IoU
    ious = compute_iou(pred_boxes, gt_boxes)  # [N_pred, N_gt]

    # 匹配预测框和真实框
    tp = torch.zeros(len(pred_boxes))
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

    for i, iou_row in enumerate(ious):
        max_iou, max_idx = torch.max(iou_row, 0)
        if max_iou >= iou_threshold and not gt_matched[max_idx]:
            tp[i] = 1
            gt_matched[max_idx] = True

    # 计算累积精度和召回率
    tp_cumsum = torch.cumsum(tp, 0)
    fp_cumsum = torch.cumsum(1 - tp, 0)

    recall = tp_cumsum / len(gt_boxes)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # 计算AP (使用11点插值)
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        p_interp = torch.max(precision[recall >= t]) if torch.any(recall >= t) else torch.tensor(0.0)
        ap += p_interp / 11

    return ap.item()


def compute_batch_detection_metrics(gt_all, preds_all):
    """计算批量检测指标"""
    all_ap_50 = []
    all_ap_50_95 = []
    all_iou_50 = []

    for gt, pred in zip(gt_all, preds_all):
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]

        if len(gt_boxes) == 0:
            continue

        if len(pred_boxes) == 0:
            all_ap_50.append(0.0)
            all_ap_50_95.append(0.0)
            all_iou_50.append(0.0)
            continue

        # 按类别分别计算AP
        unique_labels = torch.unique(gt_labels)
        class_ap_50 = []
        class_ap_50_95 = []
        class_iou_50 = []

        for label in unique_labels:
            # 该类别的真实框
            gt_mask = gt_labels == label
            gt_class_boxes = gt_boxes[gt_mask]

            # 该类别的预测框
            pred_mask = pred_labels == label
            if not pred_mask.any():
                class_ap_50.append(0.0)
                class_ap_50_95.append(0.0)
                class_iou_50.append(0.0)
                continue

            pred_class_boxes = pred_boxes[pred_mask]
            pred_class_scores = pred_scores[pred_mask]

            # 计算该类别的AP
            ap_50 = compute_ap_single_class(pred_class_boxes, pred_class_scores, gt_class_boxes, 0.5)
            class_ap_50.append(ap_50)

            # 计算AP@0.5:0.95
            ap_list = []
            for iou_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                ap = compute_ap_single_class(pred_class_boxes, pred_class_scores, gt_class_boxes, iou_thresh)
                ap_list.append(ap)
            class_ap_50_95.append(np.mean(ap_list))

            # 计算最大IoU
            if len(pred_class_boxes) > 0 and len(gt_class_boxes) > 0:
                ious = compute_iou(pred_class_boxes, gt_class_boxes)
                max_iou = torch.max(ious).item()
                class_iou_50.append(max_iou)
            else:
                class_iou_50.append(0.0)

        # 取所有类别的平均值
        all_ap_50.append(np.mean(class_ap_50) if class_ap_50 else 0.0)
        all_ap_50_95.append(np.mean(class_ap_50_95) if class_ap_50_95 else 0.0)
        all_iou_50.append(np.mean(class_iou_50) if class_iou_50 else 0.0)

    return {
        'ap': np.mean(all_ap_50) if all_ap_50 else 0.0,
        'iou_50': np.mean(all_iou_50) if all_iou_50 else 0.0,
        'iou_50_95': np.mean(all_ap_50_95) if all_ap_50_95 else 0.0
    }

def train(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        dataloader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        max_norm: float = 0.1,
        **kwargs
):
    """训练一个epoch - 参考D-FINE实现"""
    model.train()
    criterion.train()

    total_loss = 0.0
    total_samples = 0
    losses = []

    # 从kwargs获取可选参数
    scaler = kwargs.get("scaler", None)
    ema = kwargs.get("ema", None)
    lr_warmup_scheduler = kwargs.get("lr_warmup_scheduler", None)

    with tqdm(dataloader, desc=f'Epoch {epoch} [Training]') as pbar:
        for batch_idx, (rgb_images, tir_images, targets) in enumerate(pbar):
            global_step = epoch * len(dataloader) + batch_idx
            metas = dict(epoch=epoch, step=batch_idx, global_step=global_step, epoch_step=len(dataloader))

            rgb_images = rgb_images.to(device)
            tir_images = tir_images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                       for t in targets]

            # 混合精度训练
            if scaler is not None:
                with torch.autocast(device_type=str(device), cache_enabled=True):
                    outputs = model(rgb_images, tir_images, targets=targets)

                # 检查NaN
                if hasattr(outputs, 'pred_boxes') and torch.isnan(outputs["pred_boxes"]).any():
                    print("NaN detected in pred_boxes!")
                    continue

                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, targets, **metas)

                loss = sum(loss_dict.values())
                scaler.scale(loss).backward()

                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            else:
                outputs = model(rgb_images, tir_images, targets=targets)
                loss_dict = criterion(outputs, targets, **metas)

                loss = sum(loss_dict.values())
                optimizer.zero_grad()
                loss.backward()

                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()

            # EMA更新
            if ema is not None:
                ema.update(model)

            # 学习率预热
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

            # 检查loss是否有限
            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, stopping training")
                print(loss_dict)
                sys.exit(1)

            total_loss += loss.item() * rgb_images.size(0)
            total_samples += rgb_images.size(0)
            losses.append(loss.item())

            pbar.set_postfix({
                'loss': total_loss / total_samples,
                'lr': optimizer.param_groups[0]["lr"]
            })

    epoch_loss = total_loss / total_samples
    return {
        'loss': epoch_loss,
        'lr': optimizer.param_groups[0]["lr"]
    }


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        postprocessor,  # DetNMSPostProcessor
        dataloader,
        device: torch.device,
        epoch: int,
        **kwargs
):
    """评估模型 - 支持cxcywh格式的框"""
    model.eval()
    criterion.eval()
    if postprocessor is not None:
        postprocessor.eval()

    total_loss = 0.0
    total_samples = 0

    # 存储所有预测和真实标签用于整体评估
    gt_all = []
    preds_all = []

    with torch.no_grad(), tqdm(dataloader, desc=f'Epoch {epoch} [Validation]') as pbar:
        for batch_idx, (rgb_images, tir_images, targets) in enumerate(pbar):
            rgb_images = rgb_images.to(device)
            tir_images = tir_images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                       for target in targets]

            outputs = model(rgb_images, tir_images)

            # 计算损失
            metas = dict(epoch=epoch, step=batch_idx)
            loss_dict = criterion(outputs, targets, **metas)
            loss = sum(loss_dict.values())
            total_loss += loss.item() * rgb_images.size(0)
            total_samples += rgb_images.size(0)

            # 后处理输出
            if postprocessor is not None:
                # 获取原始图像尺寸
                if "size" in targets[0]:
                    orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                else:
                    h, w = rgb_images.shape[-2:]
                    orig_target_sizes = torch.tensor([[h, w]] * len(targets), device=device)

                # 使用DetNMSPostProcessor进行后处理
                results = postprocessor(outputs, orig_target_sizes)

                # 格式化结果用于评估
                for idx, (target, result) in enumerate(zip(targets, results)):
                    # 真实标签 - 确保boxes格式正确
                    gt_boxes = target["boxes"]
                    if gt_boxes.shape[0] > 0:
                        # 将cxcywh转换为xyxy
                        gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
                        # 使用target的size将归一化坐标转换为实际像素坐标
                        gt_boxes_xyxy[:, 0::2] *= target["size"][1]  # 宽度缩放
                        gt_boxes_xyxy[:, 1::2] *= target["size"][0]  # 高度缩放
                        gt_all.append({
                            "boxes": gt_boxes_xyxy,  # 转换为非归一化的xyxy格式
                            "labels": target["labels"],
                        })

                        # 预测结果已经是非归一化的xyxy格式
                        preds_all.append({
                            "boxes": result["boxes"],
                            "labels": result["labels"],
                            "scores": result["scores"]
                        })

            pbar.set_postfix({'loss': total_loss / total_samples})

    val_loss = total_loss / total_samples

    # 计算整体指标
    if len(gt_all) > 0 and len(preds_all) > 0:
        avg_metrics = compute_batch_detection_metrics(gt_all, preds_all)
    else:
        avg_metrics = {'ap': 0.0, 'iou_50': 0.0, 'iou_50_95': 0.0}

    metrics = {
        'loss': val_loss,
        **avg_metrics
    }

    return metrics


def compute_batch_detection_metrics(gt_all, preds_all):
    """计算批量检测指标"""
    all_ap_50 = []
    all_ap_50_95 = []
    all_iou_50 = []

    for gt, pred in zip(gt_all, preds_all):
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]

        if len(gt_boxes) == 0:
            continue

        if len(pred_boxes) == 0:
            all_ap_50.append(0.0)
            all_ap_50_95.append(0.0)
            all_iou_50.append(0.0)
            continue

        # 按类别分别计算AP
        unique_labels = torch.unique(gt_labels)
        class_ap_50 = []
        class_ap_50_95 = []
        class_iou_50 = []

        for label in unique_labels:
            # 该类别的真实框
            gt_mask = gt_labels == label
            gt_class_boxes = gt_boxes[gt_mask]

            # 该类别的预测框
            pred_mask = pred_labels == label
            if not pred_mask.any():
                class_ap_50.append(0.0)
                class_ap_50_95.append(0.0)
                class_iou_50.append(0.0)
                continue

            pred_class_boxes = pred_boxes[pred_mask]
            pred_class_scores = pred_scores[pred_mask]

            # 计算该类别的AP
            ap_50 = compute_ap_single_class(pred_class_boxes, pred_class_scores, gt_class_boxes, 0.5)
            class_ap_50.append(ap_50)

            # 计算AP@0.5:0.95
            ap_list = []
            for iou_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                ap = compute_ap_single_class(pred_class_boxes, pred_class_scores, gt_class_boxes, iou_thresh)
                ap_list.append(ap)
            class_ap_50_95.append(np.mean(ap_list))

            # 计算最大IoU
            if len(pred_class_boxes) > 0 and len(gt_class_boxes) > 0:
                ious = compute_iou(pred_class_boxes, gt_class_boxes)
                max_iou = torch.max(ious).item()
                class_iou_50.append(max_iou)
            else:
                class_iou_50.append(0.0)

        # 取所有类别的平均值
        all_ap_50.append(np.mean(class_ap_50) if class_ap_50 else 0.0)
        all_ap_50_95.append(np.mean(class_ap_50_95) if class_ap_50_95 else 0.0)
        all_iou_50.append(np.mean(class_iou_50) if class_iou_50 else 0.0)

    return {
        'ap': np.mean(all_ap_50) if all_ap_50 else 0.0,
        'iou_50': np.mean(all_iou_50) if all_iou_50 else 0.0,
        'iou_50_95': np.mean(all_ap_50_95) if all_ap_50_95 else 0.0
    }
