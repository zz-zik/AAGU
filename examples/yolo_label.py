# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: yolo_label.py
@Time    : 2025/5/20 上午10:46
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 读取指定yolo格式文件夹下的内容，判定txt文件的class是否越界，只有一个类别0
@Usage   :
"""
import os

import numpy as np


def validate_yolo_labels(label_dir):
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    lines = f.readlines()
                for line_num, line in enumerate(lines):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = parts[0]
                    if class_id != "0":
                        print(f"[警告] 文件 {file_path} 第 {line_num + 1} 行 class id 不合法: {class_id}")


def validate_yolo_boxes(label_dir):
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    lines = f.readlines()
                for line_num, line in enumerate(lines):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        class_id = parts[0]
                        coords = np.array(parts[1:], dtype=np.float32)

                        assert len(coords) == 4, f"[错误] 文件 {file_path} 第 {line_num + 1} 行 坐标数量不合法: {len(coords)}"

                        x_center, y_center, width, height = coords

                        # 检查是否为 NaN 或 Inf
                        if np.isnan(coords).any() or np.isinf(coords).any():
                            print(f"[警告] 文件 {file_path} 第 {line_num + 1} 行 包含 NaN 或 Inf 值: {coords}")
                            continue

                        # 检查范围是否合法（YOLO 格式要求在 [0, 1] 区间）
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                            print(f"[警告] 文件 {file_path} 第 {line_num + 1} 行 坐标超出 YOLO 合法范围 [0,1]: {coords}")

                        # 检查宽高是否非负
                        if width < 0 or height < 0:
                            print(f"[警告] 文件 {file_path} 第 {line_num + 1} 行 宽或高为负数: {coords}")

                    except ValueError as e:
                        print(f"[错误] 文件 {file_path} 第 {line_num + 1} 行 数据无法解析为浮点数: {line}")


if __name__ == "__main__":
    label_dir = r'/sxs/zhoufei/AAGU/dataset/OdinMJ3/false_tag/labels'
    validate_yolo_boxes(label_dir)
