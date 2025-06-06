# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: test_engine.py
@Time    : 2025/5/24 下午6:43
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 测试引擎
@Usage   : 
"""
import csv
import os
import cv2
import numpy as np
import torch
import torchvision
from torch import Tensor
from tqdm import tqdm

from dataloader import mscoco_category2label, DeNormalize


@torch.no_grad()
def tester(
        model: torch.nn.Module,
        postprocessor,
        dataloader,
        device: torch.device,
        threshold: float = 0.5,
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

            outputs, _ = model(rgb_images, tir_images)

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

                # 格式化储存结果用于评估
                for idx, (target, result) in enumerate(zip(targets, results)):
                    # 过滤掉低于阈值的预测结果
                    keep = result["scores"] > threshold
                    pred_boxes = result["boxes"][keep]
                    pred_labels = result["labels"][keep]
                    pred_scores = result["scores"][keep]

                    # 应用 NMS 去除重复框
                    nms_keep = apply_nms(pred_boxes, pred_scores, pred_labels, iou_threshold=0.7)
                    # nms_keep = nms_keep[:keep_topk]  # 可选：限制最大保留数量

                    filtered_result = {
                        "boxes": pred_boxes[nms_keep],
                        "labels": pred_labels[nms_keep],
                        "scores": pred_scores[nms_keep],
                    }

                    image_name = os.path.splitext(target['image_name'])[0]
                    h, w = rgb_images.shape[-2:]
                    preds.append({
                        "boxes": filtered_result["boxes"],
                        "labels": filtered_result['labels'],
                        "scores": filtered_result["scores"],
                        "image_name": image_name,
                        "image_size": (h, w)  # 添加图像尺寸字段
                    })

                    if show:
                        # 转换为图像
                        denorm_rgb = DeNormalize(mean=[0.341, 0.355, 0.258], std=[0.131, 0.135, 0.118])
                        denorm_tir = DeNormalize(mean=[0.615, 0.116, 0.374], std=[0.236, 0.156, 0.188])

                        save_image(
                            image_tensor=rgb_images[idx],
                            result=filtered_result,
                            output_dir=output_dir,
                            image_name=image_name,
                            denorm=denorm_rgb,
                            modal_name='RGB'
                        )
                        save_image(
                            image_tensor=tir_images[idx],
                            result=filtered_result,
                            output_dir=output_dir,
                            image_name=image_name,
                            denorm=denorm_tir,
                            modal_name='TIR'
                        )
    # 保存结果到CSV文件
    save_to_csv(preds, output_dir)

    # 保存为yolo格式的标注
    save_to_yolo_format(preds, output_dir)

    return preds


def apply_nms(boxes: Tensor, scores: Tensor, labels: Tensor, iou_threshold: float = 0.7):
    """
    对预测的边界框应用非极大值抑制(NMS)，去除重叠的冗余框。

    Args:
        boxes (Tensor): 边界框张量，形状为 [N, 4]
        scores (Tensor): 每个框的置信度分数，形状为 [N]
        labels (Tensor): 每个框的类别标签，形状为 [N]
        iou_threshold (float): IoU 阈值，用于判断两个框是否重叠过多

    Returns:
        Tensor: 经过 NMS 后保留的边界框索引
    """
    # 按类别分别执行 batched_nms
    keep_indices = torchvision.ops.batched_nms(
        boxes=boxes,
        scores=scores,
        idxs=labels,
        iou_threshold=iou_threshold
    )
    return keep_indices


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
    将检测结果保存为CSV文件，每张图像一行数据
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

            # 去掉文件后缀
            # image_id = os.path.splitext(image_name)[0]

            if len(boxes) == 0:
                # 没有检测结果的情况
                writer.writerow([
                    idx,
                    image_name,
                    '0',
                    '[0, 0, 0, 0]',
                    '0'
                ])
                continue

            # 格式化 bbox 为 [x_center, y_center, width, height]
            formatted_boxes = []
            for box in boxes:
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                width = box[2] - box[0]
                height = box[3] - box[1]
                formatted_boxes.append([x_center, y_center, width, height])

            # 类别 ID 和 置信度分数转为字符串
            category_str = ','.join(map(str, labels.astype(int).astype(str)))
            score_str = ','.join([f"{s:.6f}" for s in scores])
            box_str = '[' + '],['.join([','.join([f"{v:.6f}" for v in b]) for b in formatted_boxes]) + ']'

            # 写入一行
            writer.writerow([
                idx,
                image_name,
                category_str,
                box_str,
                score_str
            ])

    print(f"✅ 检测结果已保存至 {result_file}")


def save_to_yolo_format(preds, output_dir):
    """
    将检测结果保存为 YOLO 格式（每个图像一个 .txt 文件）

    Args:
        preds: List of dicts with keys ['boxes', 'labels', 'scores', 'image_name', 'image_size']
        output_dir: 输出根目录
    """
    yolo_output_dir = os.path.join(output_dir, "labels")
    os.makedirs(yolo_output_dir, exist_ok=True)

    for pred in preds:
        image_name = pred["image_name"]
        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        img_h, img_w = pred["image_size"]  # 直接从 preds 中取出图像尺寸

        base_name = os.path.splitext(image_name)[0]
        label_file = os.path.join(yolo_output_dir, f"{base_name}.txt")

        with open(label_file, mode="w", encoding="utf-8") as f:
            if len(boxes) == 0:
                continue

            for box, label in zip(boxes, labels):
                x_center = (box[0] + box[2]) / 2 / img_w
                y_center = (box[1] + box[3]) / 2 / img_h
                width = (box[2] - box[0]) / img_w
                height = (box[3] - box[1]) / img_h

                f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"✅ YOLO 格式标注文件已保存至 {yolo_output_dir}")
