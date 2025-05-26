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
import csv
import os

import cv2
import math
import sys

import numpy as np
import torch
from tqdm import tqdm

from dataloader import mscoco_category2label, DeNormalize
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


def ap_class(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
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


def det_metrics(gt_all, preds_all):
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
        class_ap_50_95 = []
        class_iou_50 = []

        for label in unique_labels:
            # 该类别的真实框
            gt_mask = gt_labels == label
            gt_class_boxes = gt_boxes[gt_mask]

            # 该类别的预测框
            pred_mask = pred_labels == label
            if not pred_mask.any():
                class_ap_50_95.append(0.0)
                class_iou_50.append(0.0)
                continue

            pred_class_boxes = pred_boxes[pred_mask]
            pred_class_scores = pred_scores[pred_mask]

            # 计算AP@0.5:0.95
            ap_list = []
            for iou_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                ap = ap_class(pred_class_boxes, pred_class_scores, gt_class_boxes, iou_thresh)
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
        all_ap_50_95.append(np.mean(class_ap_50_95) if class_ap_50_95 else 0.0)
        all_iou_50.append(np.mean(class_iou_50) if class_iou_50 else 0.0)

    return {
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
    """评估模型"""
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
                if "orig_size" in targets[0]:
                    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                else:
                    h, w = rgb_images.shape[-2:]
                    orig_target_sizes = torch.tensor([[w, h]] * len(targets), device=device)

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
                        gt_boxes_xyxy[:, 0::2] *= target["orig_size"][0]  # 宽度缩放
                        gt_boxes_xyxy[:, 1::2] *= target["orig_size"][1]  # 高度缩放
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
        avg_metrics = det_metrics(gt_all, preds_all)
    else:
        avg_metrics = {'iou_50': 0.0, 'iou_50_95': 0.0}

    metrics = {
        'loss': val_loss,
        **avg_metrics
    }

    return metrics


@torch.no_grad()
def _test(
        model: torch.nn.Module,
        postprocessor,
        dataloader,
        device: torch.device,
        **kwargs
):
    """测试模型"""
    model.eval()
    if postprocessor is not None:
        postprocessor.eval()

    # 存储所有真实标签
    preds = []

    output_dir = kwargs.get('output_dir', '')
    show = kwargs.get('show', False)

    with torch.no_grad(), tqdm(dataloader, desc=f'[Test]') as pbar:
        for batch_idx, (rgb_images, tir_images, targets) in enumerate(pbar):
            rgb_images = rgb_images.to(device)
            tir_images = tir_images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                       for target in targets]

            outputs = model(rgb_images, tir_images)

            # 后处理输出
            if postprocessor is not None:
                # 获取原始图像尺寸
                if "orig_size" in targets[0]:
                    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                else:
                    h, w = rgb_images.shape[-2:]
                    orig_target_sizes = torch.tensor([[w, h]] * len(targets), device=device)

                # 使用DetNMSPostProcessor进行后处理
                results = postprocessor(outputs, orig_target_sizes)

                # 格式化结果用于评估
                for idx, (target, result) in enumerate(zip(targets, results)):
                    # boxes格式正确
                    image_name = target['image_name']
                    labels = (
                        torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()]).to(
                            result["labels"].device).reshape(result["labels"].shape)
                    ) if postprocessor.remap_mscoco_category else result["labels"]
                    # 预测结果已经是非归一化的xyxy格式
                    preds.append({
                        "boxes": result["boxes"],
                        "labels": labels,
                        "scores": result["scores"],
                        "image_name": image_name
                    })
                    if show:
                        # 转换为图像
                        denorm_rgb = DeNormalize(mean=[0.341, 0.355, 0.258], std=[0.131, 0.135, 0.118])
                        denorm_tir = DeNormalize(mean=[0.615, 0.116, 0.374], std=[0.236, 0.156, 0.188])

                        save_image(
                            image_tensor=rgb_images[idx],
                            result=result,
                            output_dir=output_dir,
                            image_name=image_name,
                            denorm=denorm_rgb,
                            modal_name='RGB'
                        )
                        save_image(
                            image_tensor=tir_images[idx],
                            result=result,
                            output_dir=output_dir,
                            image_name=image_name,
                            denorm=denorm_tir,
                            modal_name='TIR'
                        )

    # 保存结果到CSV文件
    save_to_csv(preds, output_dir)

    return preds


def tensor_to_image(img_tensor):
    """将 PyTorch Tensor 转换为 NumPy 图像（HWC 格式）"""
    img = img_tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = (img * 255).astype(np.uint8)  # [0,1] -> [0,255]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR for OpenCV


def draw_boxes(image, result):
    """在图像上绘制检测到的边界框和标签"""
    boxes = result['boxes'].cpu().numpy()
    labels = result['labels'].cpu().numpy()
    scores = result['scores'].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        # 绘制边界框
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框

        # 绘制标签背景
        label_text = f"{label}:{score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width + 5, y1), (0, 0, 255), -1)  # 红色背景

        # 绘制标签文本
        cv2.putText(image, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def save_image(image_tensor, result, output_dir, image_name, denorm, modal_name='RGB'):
    """
    处理单张图像（RGB 或 TIR）并保存绘制边界框后的图像

    Args:
        image_tensor (torch.Tensor): 输入图像张量 (C, H, W)
        result (dict): 检测结果，包含 boxes、labels、scores
        output_dir (str): 保存根目录
        image_name (str): 图像文件名
        denorm (callable): 归一化还原函数
        modal_name (str): 模态名称如 'RGB' 或 'TIR'，用于构建子目录
    """
    # 反归一化处理
    image = denorm(image_tensor.clone().cpu())

    # 张量转 NumPy 图像
    image_np = tensor_to_image(image)

    result = {k: v.cpu() for k, v in result.items()}
    # 绘制边界框
    image_with_boxes = draw_boxes(image_np.copy(), result)

    # 构建路径并保存
    output_subdir = os.path.join(output_dir, modal_name)
    os.makedirs(output_subdir, exist_ok=True)
    filename = os.path.join(output_subdir, f"{image_name}.png")
    cv2.imwrite(filename, image_with_boxes)


def save_to_csv(preds, output_dir):
    """
    将检测结果保存为CSV文件
    preds: List of dicts with keys ['boxes', 'labels', 'scores', 'image_name']
    output_dir: 输出目录
    """
    result_file = os.path.join(output_dir, "results.csv")

    # 写入CSV文件
    with open(result_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['id', 'image_id', 'category_id', 'bbox', 'score'])

        for idx, pred in enumerate(preds):
            image_name = pred['image_name']
            boxes = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()

            # 格式化 bbox 为 [x, y, width, height]
            formatted_boxes = []
            for box in boxes:
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                width = box[2] - box[0]
                height = box[3] - box[1]
                formatted_boxes.append([x_center, y_center, width, height])

            # 遍历每个检测结果，写入一行
            for box, label, score in zip(formatted_boxes, labels, scores):
                bbox_str = ','.join(map(str, box))
                writer.writerow([
                    idx,              # 检测结果的 ID
                    image_name,       # 图像名
                    str(label),       # 类别 ID
                    bbox_str,         # Bounding Box
                    str(score)        # 置信度分数
                ])

    print(f"✅检测结果已保存至 {result_file}")

