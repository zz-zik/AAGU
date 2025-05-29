# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: point_boxes.py
@Time    : 2025/5/28 上午10:36
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 将无人机视角下图像人头点标注转换为yolo框标注格式
@Usage   :
data
  |--train
        |--rgb
            |--1.jpg
            |--2.jpg
            |--num.jpg
        |--tir
            |--1.jpg
            |--num.jpg
        |--labels
            |--num.xml
  |__test
将yolo标注保存为train/test的子文件夹下的annotations文件夹中，yolo中的txt文件名和rgb一致
"""

import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from pathlib import Path


def read_annotation(xml_path):
    """读取xml文件，返回标注点坐标"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall('object')
    points = []
    for obj in objects:
        if obj is not None:
            point = obj.find('point')
            if point is not None:
                x = float(point.find('x').text)
                y = float(point.find('y').text)
                points.append([x, y])
            else:
                print('No point in object')
    return points


def get_image_size(image_path):
    """获取图像尺寸"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    height, width = img.shape[:2]
    return width, height


def points_to_yolo_boxes(points, img_width, img_height, box_size_ratio=0.08,
                         adaptive_sizing=True, min_box_size=10, max_box_size=30):
    """
    将点标注转换为YOLO格式的边界框

    Args:
        points: 点坐标列表 [[x1, y1], [x2, y2], ...]
        img_width: 图像宽度
        img_height: 图像高度
        box_size_ratio: 基础框大小比例（相对于图像较小边）
        adaptive_sizing: 是否启用自适应尺寸调整
        min_box_size: 最小框尺寸（像素）
        max_box_size: 最大框尺寸（像素）

    Returns:
        yolo_boxes: YOLO格式边界框列表 [[class_id, x_center, y_center, width, height], ...]
    """
    yolo_boxes = []

    # 基础框大小计算
    base_size = min(img_width, img_height) * box_size_ratio
    base_size = max(min_box_size, min(base_size, max_box_size))

    if adaptive_sizing and len(points) > 1:
        # 计算点间距离用于自适应调整
        distances = []
        points_array = np.array(points)

        for i, point in enumerate(points_array):
            # 找到最近的几个邻居点
            other_points = np.delete(points_array, i, axis=0)
            if len(other_points) > 0:
                dists = np.sqrt(np.sum((other_points - point) ** 2, axis=1))
                # 取最近的1-3个点的平均距离
                k = min(3, len(dists))
                nearest_dists = np.partition(dists, k - 1)[:k]
                avg_dist = np.mean(nearest_dists)
                distances.append(avg_dist)

        if distances:
            mean_distance = np.mean(distances)
            # 根据平均距离调整框大小，避免重叠
            adaptive_size = max(base_size, mean_distance * 0.3)
            adaptive_size = min(adaptive_size, max_box_size)
        else:
            adaptive_size = base_size
    else:
        adaptive_size = base_size

    for point in points:
        x, y = point

        # 确保点在图像范围内
        x = max(0, min(x, img_width))
        y = max(0, min(y, img_height))

        # 当前点的框大小（可以根据位置进一步微调）
        current_box_size = adaptive_size

        # 边界框坐标计算
        half_size = current_box_size / 2

        # 确保框不超出图像边界
        x_min = max(0, x - half_size)
        y_min = max(0, y - half_size)
        x_max = min(img_width, x + half_size)
        y_max = min(img_height, y + half_size)

        # 重新计算中心点和尺寸（处理边界情况）
        box_width = x_max - x_min
        box_height = y_max - y_min
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # 转换为YOLO格式（归一化坐标）
        yolo_x = center_x / img_width
        yolo_y = center_y / img_height
        yolo_w = box_width / img_width
        yolo_h = box_height / img_height

        # 添加到结果列表（假设类别ID为0，表示人头）
        yolo_boxes.append([0, yolo_x, yolo_y, yolo_w, yolo_h])

    return yolo_boxes


def save_yolo_annotation(yolo_boxes, output_path):
    """保存YOLO格式标注文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for box in yolo_boxes:
            class_id, x_center, y_center, width, height = box
            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def process_dataset(data_root, box_size_ratio=0.08, adaptive_sizing=True):
    """
    处理整个数据集

    Args:
        data_root: 数据根目录路径
        box_size_ratio: 框大小比例
        adaptive_sizing: 是否启用自适应尺寸
    """
    data_root = Path(data_root)

    # 处理train和test数据
    for split in ['train', 'val', 'test']:
        split_path = data_root / split
        if not split_path.exists():
            print(f"跳过不存在的目录: {split_path}")
            continue

        rgb_path = split_path / 'rgb'
        labels_path = split_path / 'labels'
        annotations_output_path = split_path / 'annotations'

        if not rgb_path.exists() or not labels_path.exists():
            print(f"跳过目录 {split}: rgb或labels文件夹不存在")
            continue

        # 获取所有RGB图像文件
        rgb_files = list(rgb_path.glob('*.jpg')) + list(rgb_path.glob('*.png'))

        processed_count = 0
        error_count = 0

        for rgb_file in rgb_files:
            try:
                # 对应的XML标注文件
                xml_file = labels_path / f"{rgb_file.stem}.xml"

                if not xml_file.exists():
                    print(f"警告: 找不到对应的XML文件: {xml_file}")
                    continue

                # 读取点标注
                points = read_annotation(str(xml_file))

                if not points:
                    print(f"警告: {xml_file} 中没有找到有效的点标注")
                    continue

                # 获取图像尺寸
                img_width, img_height = get_image_size(str(rgb_file))

                # 转换为YOLO格式
                yolo_boxes = points_to_yolo_boxes(
                    points, img_width, img_height,
                    box_size_ratio=box_size_ratio,
                    adaptive_sizing=adaptive_sizing
                )

                # 保存YOLO标注
                output_file = annotations_output_path / f"{rgb_file.stem}.txt"
                save_yolo_annotation(yolo_boxes, str(output_file))

                processed_count += 1
                print(f"处理完成: {rgb_file.name} -> {output_file.name} (点数: {len(points)})")

            except Exception as e:
                error_count += 1
                print(f"处理文件 {rgb_file.name} 时出错: {str(e)}")

        print(f"\n{split} 数据集处理完成:")
        print(f"  成功处理: {processed_count} 个文件")
        print(f"  处理失败: {error_count} 个文件")
        print(f"  输出目录: {annotations_output_path}")


def visualize_conversion(image_path, xml_path, output_path=None, box_size_ratio=0.08):
    """
    可视化点到框的转换结果

    Args:
        image_path: 图像文件路径
        xml_path: XML标注文件路径
        output_path: 可视化结果保存路径（可选）
        box_size_ratio: 框大小比例
    """
    # 读取图像和标注
    img = cv2.imread(image_path)
    points = read_annotation(xml_path)

    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    if not points:
        print(f"没有找到有效的点标注: {xml_path}")
        return

    img_height, img_width = img.shape[:2]

    # 转换为YOLO格式
    yolo_boxes = points_to_yolo_boxes(
        points, img_width, img_height,
        box_size_ratio=box_size_ratio,
        adaptive_sizing=True
    )

    # 在图像上绘制
    for i, (point, yolo_box) in enumerate(zip(points, yolo_boxes)):
        # 绘制原始点
        cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

        # 绘制转换后的边界框
        _, x_center, y_center, width, height = yolo_box

        # 转换回像素坐标
        x_center_pixel = x_center * img_width
        y_center_pixel = y_center * img_height
        width_pixel = width * img_width
        height_pixel = height * img_height

        x1 = int(x_center_pixel - width_pixel / 2)
        y1 = int(y_center_pixel - height_pixel / 2)
        x2 = int(x_center_pixel + width_pixel / 2)
        y2 = int(y_center_pixel + height_pixel / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(i), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 显示或保存结果
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"可视化结果已保存到: {output_path}")
    else:
        cv2.imshow('Point to Box Conversion', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 使用示例
    data_root = "/sxs/zhoufei/P2PNet/DroneRGBT"  # 替换为您的数据根目录路径

    # 处理整个数据集
    # 参数说明:
    # - box_size_ratio: 控制框的基础大小，相对于图像较小边的比例
    # - adaptive_sizing: 是否根据点的密度自适应调整框大小
    # process_dataset(
    #     data_root=data_root,
    #     box_size_ratio=0.05,  # 可以调整这个值来控制框的大小
    #     adaptive_sizing=True  # 启用自适应尺寸以获得更精确的边界
    # )

    # 可视化单个文件的转换结果（用于调试和验证）
    visualize_conversion(
        image_path=data_root + "/test/tir/1036.jpg",
        xml_path=data_root + "/test/labels/1036.xml",
        output_path=data_root + "/visualization_result.jpg",
        box_size_ratio=0.05
    )