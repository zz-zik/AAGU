# -*- coding: utf-8 -*-
"""
@Project : MCDNet
@FileName: dataset_split.py
@Time    : 2025/5/12 上午9:37
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    :
@Usage   : 将 YOLO 格式的数据集转换为 COCO 格式
"""
import os
import shutil
import json
from PIL import Image
from tqdm import tqdm


def yolo_to_coco(yolo_label_path, image_path, image_id, labels):
    """
    将 YOLO 格式的标签转换为 COCO 格式
    参数：
        yolo_label_path : YOLO 标签文件路径
        image_path : 对应的图像文件路径
        image_id : 图像的唯一标识符
        labels : 类别信息字典
    返回：
        COCO 格式的标签数据
    """
    coco_labels = []
    # 读取图像的尺寸
    with Image.open(image_path) as img:
        width, height = img.size
    if os.path.exists(yolo_label_path):
        with open(yolo_label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO格式：类别 x_center y_center width height（归一化）
                    yolo_class_id = int(parts[0])
                    # 检查类别 ID 是否越界，如果越界则修改为 0
                    if yolo_class_id not in labels.values():
                        yolo_class_id = 0
                    # 直接将 YOLO 类别 ID 映射到 COCO 类别 ID（这里假设类别 ID 一致）
                    coco_class_id = yolo_class_id
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    bbox_width = float(parts[3]) * width
                    bbox_height = float(parts[4]) * height
                    # COCO格式：x_min, y_min, width, height
                    x_min = x_center - bbox_width / 2
                    y_min = y_center - bbox_height / 2
                    coco_labels.append({
                        "id": len(coco_labels) + 1,  # id需要唯一
                        "image_id": image_id,
                        "category_id": coco_class_id,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    })
    return coco_labels


def process_dataset(src_dir, dst_dir, labels):
    """
    处理数据集，将 YOLO 格式转换为 COCO 格式并组织目录结构
    参数：
        src_dir : 原始数据集目录路径
        dst_dir : 输出整理后的数据集目录路径
        labels : 类别信息字典
    """
    # 创建目标目录结构
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dst_dir, split, 'RGB'), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, split, 'TIR'), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, split, 'annotations'), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, split, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'test', 'RGB'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'test', 'TIR'), exist_ok=True)

    # 处理训练集和验证集
    for split in ['train', 'val']:
        with open(os.path.join(src_dir, 'train_val', f'{split}.txt'), 'r') as f:
            image_names = [line.strip() for line in f.readlines()]

        # 保存 COCO 格式的标注
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": v, "name": k} for k, v in labels.items()]
        }

        image_id = 1
        for image_name in tqdm(image_names, desc=f"Processing {split} set"):
            # 复制 RGB 和 TIR 图像
            shutil.copy(os.path.join(src_dir, 'train_val', 'RGB', f'{image_name}.jpg'),
                        os.path.join(dst_dir, split, 'RGB', f'{image_name}.jpg'))
            shutil.copy(os.path.join(src_dir, 'train_val', 'TIR', f'{image_name}.jpg'),
                        os.path.join(dst_dir, split, 'TIR', f'{image_name}.jpg'))

            # 复制 YOLO 格式的标签或创建空文件
            yolo_label_src = os.path.join(src_dir, 'train_val', 'yolo-labels', f'{image_name}.txt')
            yolo_label_dst = os.path.join(dst_dir, split, 'labels', f'{image_name}.txt')
            if os.path.exists(yolo_label_src):
                # 检查 YOLO 标签中的类别 ID 是否越界
                with open(yolo_label_src, 'r') as f:
                    lines = f.readlines()
                with open(yolo_label_dst, 'w') as f:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            yolo_class_id = int(parts[0])
                            # 检查类别 ID 是否越界，如果越界则修改为 0
                            if yolo_class_id not in labels.values():
                                yolo_class_id = 0
                            parts[0] = str(yolo_class_id)
                            f.write(' '.join(parts) + '\n')
            else:
                # 创建空的 YOLO 标签文件
                open(yolo_label_dst, 'w').close()

            # 转换 YOLO 到 COCO 格式
            image_path_rgb = os.path.join(src_dir, 'train_val', 'RGB', f'{image_name}.jpg')
            yolo_label_path = yolo_label_dst  # 使用处理后的 YOLO 标签
            coco_labels = yolo_to_coco(yolo_label_path, image_path_rgb, image_id, labels)

            # 添加图像信息到 COCO 数据
            with Image.open(image_path_rgb) as img:
                width, height = img.size
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": f"{image_name}.jpg",
                    "width": width,
                    "height": height
                })
                # 添加标注信息到 COCO 数据
                for label in coco_labels:
                    label["id"] = len(coco_data["annotations"]) + 1
                    label["image_id"] = image_id
                    coco_data["annotations"].append(label)
                image_id += 1

        # 保存 COCO 格式的 JSON 文件
        with open(os.path.join(dst_dir, split, 'annotations', f'{split}.json'), 'w') as f:
            json.dump(coco_data, f)

    # 处理测试集（直接复制 RGB 和 TIR 图像）
    shutil.copytree(os.path.join(src_dir, 'test', 'RGB'), os.path.join(dst_dir, 'test', 'RGB'), dirs_exist_ok=True)
    shutil.copytree(os.path.join(src_dir, 'test', 'TIR'), os.path.join(dst_dir, 'test', 'TIR'), dirs_exist_ok=True)

    print("✅数据集处理完成")


if __name__ == "__main__":
    src_dir = r"E:\Datasets\OdinMJ"  # 替换为你的原数据集目录路径
    dst_dir = "../dataset/OdinMJ"  # 替换为你想要输出整理后的数据集的目录路径
    # 设置类别信息
    labels = {"People": 0}  # 根据实际类别进行修改
    process_dataset(src_dir, dst_dir, labels)