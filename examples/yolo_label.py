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


if __name__ == "__main__":
        label_dir = r'./dataset/OdinMJ/val/labels'
        validate_yolo_labels(label_dir)
