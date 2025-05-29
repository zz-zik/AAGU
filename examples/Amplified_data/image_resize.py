# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: image_resize.py
@Time    : 2025/5/28 下午3:29
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 缩放图像和标注的尺寸到指定尺寸，并将标注转为yolo标注
@Usage   :
data
  |--img
      |--0.jpg
      |--1.jpg
  |--ann
      |--0.jpg.json
      |--1.jpg.json

output
  |--img
      |--0.jpg
      |--1.jpg
  |--labels
      |--0.txt
"""

import os
import json
import cv2
import numpy as np


# 定义图像缩放函数
def resize_image(image_path, output_path, target_size):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, target_size)
    cv2.imwrite(output_path, img_resized)


# 定义标注转换函数
def convert_annotation(annotation_path, output_path, target_size):
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)

    img_width = annotation['size']['width']
    img_height = annotation['size']['height']
    target_width, target_height = target_size

    with open(output_path, 'w') as f_out:
        for obj in annotation['objects']:
            class_title = obj['classTitle']
            # 根据类别名称设置类别ID，例如：'pedestrian' 对应 0
            class_id = 0 if class_title == 'pedestrian' else 0

            exterior = obj['points']['exterior']
            x_min, y_min = exterior[0]
            x_max, y_max = exterior[1]

            # 计算YOLO格式的坐标
            x_center = (x_min + x_max) / 2.0 / img_width
            y_center = (y_min + y_max) / 2.0 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # 写入YOLO格式的标注
            f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# 定义主函数
def main():
    # 设置输入和输出目录
    input_img_dir = '/sxs/DataSets/LADD/img'
    input_ann_dir = '/sxs/DataSets/LADD/ann'
    output_img_dir = '/sxs/DataSets/LADDs/rgb'
    output_label_dir = '/sxs/DataSets/LADDs/labels'

    # 创建输出目录
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 设置目标尺寸
    target_size = (640, 512)  # (width, height)

    # 遍历图像和标注文件
    for img_file in os.listdir(input_img_dir):
        img_path = os.path.join(input_img_dir, img_file)
        ann_file = os.path.splitext(img_file)[0] + '.jpg.json'
        ann_path = os.path.join(input_ann_dir, ann_file)

        if not os.path.exists(ann_path):
            print(f"警告：未找到标注文件 {ann_path}，跳过")
            continue

        # 处理图像
        output_img_path = os.path.join(output_img_dir, img_file)
        resize_image(img_path, output_img_path, target_size)

        # 处理标注
        label_file = os.path.splitext(img_file)[0] + '.txt'
        output_label_path = os.path.join(output_label_dir, label_file)
        convert_annotation(ann_path, output_label_path, target_size)

    print("处理完成！")


# 执行脚本
if __name__ == "__main__":
    main()