# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: crowds_dataset.py
@Time    : 2025/5/15 下午10:10
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    :
@Usage   :
"""
import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# 支持的图像格式
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']


class Crowds(Dataset):
    def __init__(self, transform, train=False, test=False, **kwargs):
        self.data_path = kwargs.get('data_root', '')
        self.train = train
        self.test = test
        self.data_format = kwargs.get('data_format', 'default')  # 控制数据集格式
        self.label_format = kwargs.get('label_format', 'coco')  # 控制标签格式，"coco" 或 "yolo"
        # YOLO 标签的类别映射字典
        self.labels = {i: label for i, label in enumerate(kwargs.get('labels', []))} if isinstance(kwargs.get('labels'),
                                                                                                   list) else kwargs.get(
            'labels')

        # 根据数据集格式构建数据目录
        if self.data_format == "default":
            if self.test:
                self.data_dir = os.path.join(self.data_path, 'test')
            else:
                self.data_dir = os.path.join(self.data_path, 'train' if self.train else 'val')
            self.rgb_dir = os.path.join(self.data_dir, 'RGB')
            self.tir_dir = os.path.join(self.data_dir, 'TIR')
            if self.label_format == "yolo":
                self.labels_dir = os.path.join(self.data_dir, 'labels')
            else:
                self.labels_dir = os.path.join(self.data_dir, 'annotations')
                # COCO 格式下，标签文件名
                self.coco_label_file = os.path.join(self.labels_dir, f'{"train" if self.train else "val"}.json')
        elif self.data_format == "custom":
            self.data_dir = self.data_path
            self.rgb_dir = os.path.join(self.data_path, 'RGB')
            self.tir_dir = os.path.join(self.data_path, 'TIR')
            self.labels_dir = os.path.join(self.data_path, 'labels' if self.label_format == "yolo" else 'annotations')
        else:
            raise ValueError(f"不支持的数据集格式：{self.data_format}")

        self.img_map = {}
        self.img_list = []

        # 根据数据集格式加载图像路径
        if self.data_format == "default":
            rgb_img_paths = [filename for filename in os.listdir(self.rgb_dir) if
                               os.path.splitext(filename)[1].lower() in SUPPORTED_IMAGE_FORMATS]
            for filename in rgb_img_paths:
                rgb_img_path = os.path.join(self.rgb_dir, filename)
                tir_img_path = os.path.join(self.tir_dir, filename)
                # 对于 YOLO 格式，标签文件后缀为 .txt
                if self.label_format == "yolo":
                    label_filename = os.path.splitext(filename)[0] + '.txt'
                    label_path = os.path.join(self.labels_dir, label_filename)
                else:
                    # COCO 格式下，标签文件是统一的 JSON 文件
                    label_path = self.coco_label_file
                if os.path.isfile(rgb_img_path) and os.path.isfile(tir_img_path):
                    # 如果是测试模式，允许标签文件不存在
                    self.img_map[rgb_img_path] = (tir_img_path, label_path)
                    self.img_list.append(rgb_img_path)
        elif self.data_format == "custom":
            # 从对应的txt文件中读取图像路径
            if self.test:
                list_file = os.path.join(self.data_path, 'list', 'test.txt')
            elif self.train:
                list_file = os.path.join(self.data_path, 'list', 'train.txt')
            else:
                list_file = os.path.join(self.data_path, 'list', 'val.txt')

            if not os.path.exists(list_file):
                raise FileNotFoundError(f"未找到列表文件：{list_file}")

            with open(list_file, 'r') as f:
                for line in f:
                    filename = line.strip()
                    if filename:
                        # 检查文件扩展名是否在支持的格式中
                        if os.path.splitext(filename)[1].lower() not in SUPPORTED_IMAGE_FORMATS:
                            continue
                        rgb_img_path = os.path.join(self.rgb_dir, filename)
                        tir_img_path = os.path.join(self.tir_dir, filename)
                        # 对于 YOLO 格式，标签文件后缀为 .txt
                        if self.label_format == "yolo":
                            label_filename = os.path.splitext(filename)[0] + '.txt'
                            label_path = os.path.join(self.labels_dir, label_filename)
                        else:
                            # COCO 格式下，标签文件是统一的 JSON 文件
                            label_path = os.path.join(self.labels_dir, f'{"train" if self.train else "val"}.json')
                        if os.path.isfile(rgb_img_path) and os.path.isfile(tir_img_path):
                            # 如果是测试模式，允许标签文件不存在
                            self.img_map[rgb_img_path] = (tir_img_path, label_path)
                            self.img_list.append(rgb_img_path)

        self.img_list = sort_filenames_numerically(self.img_list)

        self.nSamples = len(self.img_list)
        self.transform = transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if not self.img_list:
            raise IndexError("img_list 为空，无法获取数据")

        rgb_img_path = self.img_list[index]
        tir_img_path, label_path = self.img_map[rgb_img_path]
        filename = os.path.basename(rgb_img_path)

        # Step 1: 使用 OpenCV 读取图像，并转为 RGB 格式
        rgb_img = cv2.imread(rgb_img_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        tir_img = cv2.imread(tir_img_path)
        tir_img = cv2.cvtColor(tir_img, cv2.COLOR_BGR2RGB)

        # 将图像转换为 PyTorch 张量，并调整维度顺序为 (channel, height, width)
        rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float()  # 将 NumPy 数组转换为 PyTorch 张量
        tir_img = torch.from_numpy(tir_img).permute(2, 0, 1).float()

        # 读取标签
        target = {"boxes": torch.tensor([]), "labels": torch.tensor([])}  # 初始化为空目标
        if not self.test or (self.test and os.path.isfile(label_path)):
            if self.label_format == "yolo":
                # 读取 YOLO 格式的标签文件
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                boxes = []
                labels = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        boxes.append([x_center, y_center, width, height])
                        labels.append(class_id)
                target["boxes"] = torch.tensor(boxes)
                target["labels"] = torch.tensor(labels)
            elif self.label_format == "coco":
                import json
                with open(label_path, 'r') as f:
                    coco_annotation = json.load(f)
                # 获取当前图像的文件名（不带路径）
                img_id = int(os.path.splitext(filename)[0])  # 假设文件名是数字形式，如 0001.png
                annotations = [ann for ann in coco_annotation["annotations"] if ann["image_id"] == img_id]

                boxes = []
                labels = []
                # 图像宽高用于归一化
                img_info = next(img for img in coco_annotation["images"] if img["id"] == img_id)
                img_w, img_h = img_info["width"], img_info["height"]

                for ann in annotations:
                    bbox = ann["bbox"]
                    x_center = (bbox[0] + bbox[2] / 2) / img_w
                    y_center = (bbox[1] + bbox[3] / 2) / img_h
                    norm_w = bbox[2] / img_w
                    norm_h = bbox[3] / img_h
                    boxes.append([x_center, y_center, norm_w, norm_h])
                    labels.append(ann["category_id"])

                target["boxes"] = torch.tensor(boxes)
                target["labels"] = torch.tensor(labels)

        # Step 2: 数据增强（在 NumPy 阶段进行）
        if self.transform:
            rgb_img, tir_img, target = self.transform(rgb_img, tir_img, target)

        if target is not None:
            if isinstance(target, np.ndarray):
                target = torch.tensor(target, dtype=torch.int64)
            elif isinstance(target, torch.Tensor):
                # 如果标签已经是张量，则保持其结构
                pass

        if self.test and not os.path.isfile(label_path):
            return rgb_img, tir_img
        else:
            return rgb_img, tir_img, target


def sort_filenames_numerically(filenames):
    def numeric_key(filename):
        numbers = list(map(int, re.findall(r'\d+', filename)))
        return (tuple(numbers), filename) if numbers else ((), filename)

    return sorted(filenames, key=numeric_key)


if __name__ == '__main__':
    from utils import load_config

    cfg = load_config("../configs/config.yaml")
    cfg.data.data_root = '../dataset/OdinMJ'
    cfg.test.test_img_dirs = '../dataset/OdinMJ/val'

    from dataloader import Transforms

    transforms_train = Transforms(**cfg.data.transforms.to_dict())
    train_dataset = Crowds(transform=transforms_train, train=True, **cfg.data.to_dict())

    transforms_val = Transforms(train=False, **cfg.data.transforms.to_dict())
    val_dataset = Crowds(transform=transforms_val, train=False, **cfg.data.to_dict())
    test_dataset = Crowds(transform=transforms_val, test=True, **cfg.data.to_dict())

    print('训练集样本数：', len(train_dataset))
    print('验证集样本数：', len(val_dataset))
    print('测试集样本数：', len(test_dataset))

    img_rgb, img_tir, label = train_dataset[0]
    print('训练集第1个样本rgb图像形状：', img_rgb.shape, 'tir图像形状：', img_tir.shape, '标注形状：', label)

    img_rgb, img_tir, label = val_dataset[0]
    print('验证集第1个样本rgb图像形状：', img_rgb.shape, 'tir图像形状：', img_tir.shape, '标注形状：', label)

    if len(test_dataset) > 0:
        sample = test_dataset[0]
        if len(sample) == 3:
            img_rgb, img_tir, label = sample
            print('测试集第1个样本rgb图像形状：', img_rgb.shape, 'tir图像形状：', img_tir.shape, '标注形状：', label)
        else:
            img_rgb, img_tir = sample
            print('测试集第1个样本rgb图像形状：', img_rgb.shape, 'tir图像形状：', img_tir.shape)
    else:
        print("测试集为空，无法获取样本")