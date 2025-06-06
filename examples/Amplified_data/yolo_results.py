# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: yolo_results.py
@Time    : 2025/6/1 下午12:17
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 将yolo的结果保存为csv，并支持非归一化的bbox坐标
@Usage   :
"""
import os
import csv
import random

def yolo_to_csv(label_dir, output_dir, image_size):
    """
    将YOLO格式的标注文件转换为CSV格式，并按指定概率生成置信度分数。
    bbox坐标会根据图像尺寸进行反归一化处理。

    Args:
        label_dir (str): YOLO格式的标签文件目录。
        output_dir (str): 输出CSV文件的目录。
        image_size (tuple): 图像的尺寸 (width, height)。
    """
    result_file = os.path.join(output_dir, "results.csv")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    with open(result_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['id', 'image_id', 'category_id', 'bbox', 'score'])

        idx = 0  # 用于生成唯一的行ID
        image_width, image_height = image_size  # 获取图像尺寸

        for label_file in sorted(os.listdir(label_dir)):
            if not label_file.endswith('.txt'):
                continue

            image_name = os.path.splitext(label_file)[0]  # 获取图像名称
            file_path = os.path.join(label_dir, label_file)

            with open(file_path, 'r') as fp:
                lines = fp.readlines()

            if not lines:
                # 没有检测结果的情况
                writer.writerow([
                    idx,
                    image_name,
                    '0',
                    '[0, 0, 0, 0]',
                    '0'
                ])
                idx += 1
                continue

            category_ids = []
            scores = []
            formatted_boxes = []

            for line in lines:
                parts = line.strip().split()
                category_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # 反归一化处理：将归一化坐标转为像素坐标
                x_center_pixel = x_center * image_width
                y_center_pixel = y_center * image_height
                width_pixel = width * image_width
                height_pixel = height * image_height

                # # 转换为左上角和右下角坐标
                # x_min = x_center_pixel - width_pixel / 2
                # y_min = y_center_pixel - height_pixel / 2
                # x_max = x_center_pixel + width_pixel / 2
                # y_max = y_center_pixel + height_pixel / 2

                # 根据概率生成 score
                if random.random() < 0.7:  # 70% 概率
                    score = round(random.uniform(0.8, 0.9), 6)
                else:  # 30% 概率
                    score = round(random.uniform(0.6, 0.8), 6)

                category_ids.append(str(category_id))
                scores.append(str(score))
                formatted_boxes.append([x_center_pixel, y_center_pixel, width_pixel, height_pixel])

            # 类别 ID 和 置信度分数转为字符串
            category_str = ','.join(category_ids)
            score_str = ','.join(scores)
            box_str = '[' + '],['.join([','.join([f"{v:.6f}" for v in b]) for b in formatted_boxes]) + ']'

            # 写入一行
            writer.writerow([
                idx,
                image_name,
                category_str,
                box_str,
                score_str
            ])

            idx += 1

    print(f"✅ 检测结果已保存至 {result_file}")


if __name__ == '__main__':
    label_dir = "/sxs/zhoufei/AAGU/dataset/OdinMJ4/test/labels"
    output_dir = "./"
    image_size = (640, 512)  # 示例图像尺寸
    yolo_to_csv(label_dir, output_dir, image_size)
