# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: apply_random_jitter_to_val.py
@Time    : 2025/6/3 下午4:00
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 应用 RandomJitter 到 val 数据集并保存为新数据集
"""

import os
import torch
from PIL import Image
import numpy as np
from dataloader.transforms import RandomJitter


def load_yolo_boxes(label_path):
    """加载 YOLO 格式的标签文件 (class_id, cx, cy, w, h)"""
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, cx, cy, w, h = map(float, line.strip().split())
            boxes.append([cx, cy, w, h])
    return torch.tensor(boxes), int(class_id)


def save_yolo_boxes(label_path, boxes_tensor, class_id):
    """保存更新后的 YOLO 格式标签文件"""
    boxes = boxes_tensor.cpu().numpy()
    with open(label_path, 'w') as f:
        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            f.write(f"{int(class_id)} {cx} {cy} {w} {h}\n")


def main():
    input_dir = "/sxs/zhoufei/AAGU/dataset/OdinMJ2/val"
    output_dir = "/sxs/zhoufei/AAGU/dataset/OdinMJ2/aug_val"
    max_drift = 0.03
    prob = 1.0

    jitter_transform = RandomJitter(max_drift=max_drift, prob=prob)

    os.makedirs(os.path.join(output_dir, "RGB"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "TIR"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    rgb_dir = os.path.join(input_dir, "RGB")
    tir_dir = os.path.join(input_dir, "TIR")
    label_dir = os.path.join(input_dir, "labels")

    for filename in os.listdir(rgb_dir):
        base_name, ext = os.path.splitext(filename)
        if ext.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        print(f"Processing {filename}...")

        # 图像路径
        rgb_path = os.path.join(rgb_dir, filename)
        tir_path = os.path.join(tir_dir, filename)
        label_path = os.path.join(label_dir, base_name + ".txt")

        # 输出路径
        out_rgb_path = os.path.join(output_dir, "RGB", f"{base_name}.png")
        out_tir_path = os.path.join(output_dir, "TIR", f"{base_name}.png")
        out_label_path = os.path.join(output_dir, "labels", f"{base_name}.txt")

        # 判断 label 是否为空或不存在
        if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
            # 直接复制 RGB 和 TIR 图像
            import shutil
            shutil.copy(rgb_path, out_rgb_path)
            shutil.copy(tir_path, out_tir_path)

            # 创建空标签文件或复制原空文件
            if os.path.exists(label_path):
                shutil.copy(label_path, out_label_path)
            else:
                open(out_label_path, 'w').close()

            continue

        # 否则正常加载并增强
        # 加载图像
        rgb_img = Image.open(rgb_path).convert("RGB")
        tir_img = Image.open(tir_path).convert("RGB")

        # 转换为 Tensor
        rgb_tensor = torch.from_numpy(np.array(rgb_img).transpose((2, 0, 1))).float() / 255.0
        tir_tensor = torch.from_numpy(np.array(tir_img).transpose((2, 0, 1))).float() / 255.0

        # 加载边界框
        boxes, class_id = load_yolo_boxes(label_path)
        target = {"boxes": boxes}

        # 应用 RandomJitter 变换
        augmented_rgb_tensor, augmented_tir_tensor, augmented_target = jitter_transform(
            rgb_tensor, tir_tensor, target
        )

        # 保存增强后的图像
        augmented_rgb = (augmented_rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        augmented_tir = (augmented_tir_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        Image.fromarray(augmented_rgb).save(out_rgb_path)
        Image.fromarray(augmented_tir).save(out_tir_path)

        # 保存增强后的标签
        if "boxes" in augmented_target and len(augmented_target["boxes"]) > 0:
            save_yolo_boxes(out_label_path, augmented_target["boxes"], class_id)
        else:
            open(out_label_path, 'w').close()


if __name__ == "__main__":
    main()
