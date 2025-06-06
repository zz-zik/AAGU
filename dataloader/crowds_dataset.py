# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: crowds_dataset.py
@Time    : 2025/5/15 下午10:10
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    :
@Usage   : target["boxes"]=[x_min, y_min, x_max, y_max]
"""
import json
import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']


class Crowds(Dataset):
    def __init__(self, transform, train=False, test=False, box_fmt='cxcywh', **kwargs):
        self.data_path = kwargs.get('data_root', '')
        self.train = train
        self.test = test
        self.box_fmt = box_fmt
        self.data_format = kwargs.get('data_format', 'default')
        self.label_format = kwargs.get('label_format', 'coco')

        # 构建数据目录
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
                self.coco_label_file = os.path.join(
                    self.labels_dir,
                    f'{"train" if self.train else "val"}.json'
                )
                if self.test:
                    self.coco_label_file = os.path.join(self.labels_dir, 'test.json')
        elif self.data_format == "custom":
            self.data_dir = self.data_path
            self.rgb_dir = os.path.join(self.data_path, 'RGB')
            self.tir_dir = os.path.join(self.data_path, 'TIR')
            self.labels_dir = os.path.join(self.data_path, 'labels' if self.label_format == "yolo" else 'annotations')
        else:
            raise ValueError(f"不支持的数据集格式：{self.data_format}")

        self.img_map = {}
        self.img_list = []

        # 加载 COCO 标签文件（如果存在）
        self.coco_annotations = None
        if self.label_format == "coco" and not self.test:
            if os.path.exists(self.coco_label_file):
                with open(self.coco_label_file, 'r') as f:
                    self.coco_annotations = json.load(f)
            else:
                raise FileNotFoundError(f"未找到 COCO 标签文件：{self.coco_label_file}")

        # 加载图像路径
        if self.data_format == "default":
            rgb_img_paths = [
                filename for filename in os.listdir(self.rgb_dir)
                if os.path.splitext(filename)[1].lower() in SUPPORTED_IMAGE_FORMATS
            ]
            for filename in rgb_img_paths:
                rgb_img_path = os.path.join(self.rgb_dir, filename)
                tir_img_path = os.path.join(self.tir_dir, filename)

                if self.label_format == "yolo":
                    label_filename = os.path.splitext(filename)[0] + ".txt"
                    label_path = os.path.join(self.labels_dir, label_filename)
                else:
                    label_path = None

                if os.path.isfile(rgb_img_path) and os.path.isfile(tir_img_path):
                    # has_targets = False
                    # if self.label_format == "yolo" and os.path.isfile(label_path):
                    #     with open(label_path, 'r') as f:
                    #         lines = f.readlines()
                    #         if lines:
                    #             has_targets = True
                    # elif self.label_format == "coco" and self.coco_annotations:
                    #     img_id = int(os.path.splitext(filename)[0])
                    #     annotations = [ann for ann in self.coco_annotations["annotations"] if ann["image_id"] == img_id]
                    #     has_targets = len(annotations) > 0
                    #
                    # if has_targets or self.test:
                    self.img_map[rgb_img_path] = (tir_img_path, label_path)
                    self.img_list.append(rgb_img_path)

        elif self.data_format == "custom":
            list_file = os.path.join(self.data_path, 'list', 'test.txt' if self.test else 'train.txt')
            if not os.path.exists(list_file):
                raise FileNotFoundError(f"未找到列表文件：{list_file}")
            with open(list_file, 'r') as f:
                for line in f:
                    filename = line.strip()
                    if not filename:
                        continue
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in SUPPORTED_IMAGE_FORMATS:
                        continue
                    rgb_img_path = os.path.join(self.rgb_dir, filename)
                    tir_img_path = os.path.join(self.tir_dir, filename)

                    if self.label_format == "yolo":
                        label_path = os.path.join(self.labels_dir, os.path.splitext(filename)[0] + ".txt")
                    else:
                        label_path = None

                    if not os.path.isfile(rgb_img_path) or not os.path.isfile(tir_img_path):
                        continue

                    if self.test and label_path is not None and not os.path.isfile(label_path):
                        self.img_map[rgb_img_path] = (tir_img_path, None)
                        self.img_list.append(rgb_img_path)
                    else:
                        # has_targets = False
                        # if self.label_format == "yolo" and os.path.isfile(label_path):
                        #     with open(label_path, 'r') as f:
                        #         lines = f.readlines()
                        #         if lines:
                        #             has_targets = True
                        # elif self.label_format == "coco" and self.coco_annotations:
                        #     img_id = int(os.path.splitext(filename)[0])
                        #     annotations = [ann for ann in self.coco_annotations["annotations"] if ann["image_id"] == img_id]
                        #     has_targets = len(annotations) > 0
                        #
                        # if has_targets or self.test:
                        self.img_map[rgb_img_path] = (tir_img_path, label_path)
                        self.img_list.append(rgb_img_path)

        self.img_list = self.sort_filenames_numerically(self.img_list)
        self.nSamples = len(self.img_list)
        self.tensor = transforms.ToTensor()
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
        rgb_img = self.tensor(rgb_img)
        # rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0

        tir_img = cv2.imread(tir_img_path)
        tir_img = cv2.cvtColor(tir_img, cv2.COLOR_BGR2RGB)
        tir_img = self.tensor(tir_img)

        # Step 2: 读取标签
        target = {
            "boxes": torch.tensor([], dtype=torch.float32),
            "labels": torch.tensor([], dtype=torch.int64),
            "image_id": torch.tensor(-1, dtype=torch.int64),
            "rgb_path": rgb_img_path,
            "tir_path": tir_img_path,
            "image_name": filename,
            "orig_size": torch.tensor([-1, -1], dtype=torch.int64)
        }

        height, width = rgb_img.shape[1], rgb_img.shape[2]

        if self.label_format == "yolo" and os.path.isfile(label_path):
            boxes = []
            labels = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        if self.box_fmt == 'xyxy':
                            x1 = xc - w / 2
                            y1 = yc - h / 2
                            x2 = xc + w / 2
                            y2 = yc + h / 2
                            boxes.append([x1, y1, x2, y2])
                        elif self.box_fmt == 'cxcywh':
                            boxes.append([xc, yc, w, h])
                        labels.append(class_id)
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)

        elif self.label_format == "coco" and self.coco_annotations:
            img_id = int(os.path.splitext(filename)[0])
            annotations = [ann for ann in self.coco_annotations["annotations"] if ann["image_id"] == img_id]
            boxes = []
            labels = []
            for ann in annotations:
                x, y, w, h = ann["bbox"]
                if self.box_fmt == 'xyxy':
                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h
                    boxes.append([x1 / width, y1 / height, x2 / width, y2 / height])  # 归一化
                elif self.box_fmt == 'cxcywh':
                    xc = x + w / 2
                    yc = y + h / 2
                    boxes.append([xc / width, yc / height, w / width, h / height])  # 归一化
                labels.append(ann["category_id"])
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)

        # ====== 如果没有检测框，使用空张量 ======
        if target["boxes"] is None or target["labels"] is None:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0,), dtype=torch.int64)

        # 添加 image_id 和 orig_size 字段
        img_id = int(os.path.splitext(filename)[0]) if self.label_format == "coco" else index
        target["image_id"] = torch.tensor(img_id, dtype=torch.int64)
        target["orig_size"] = torch.tensor([width, height], dtype=torch.int64)

        # Step 3: 数据增强
        if self.transform:
            rgb_img, tir_img, target = self.transform(rgb_img, tir_img, target)

        return rgb_img, tir_img, target

    @staticmethod
    def sort_filenames_numerically(filenames):
        def numeric_key(filename):
            numbers = list(map(int, re.findall(r'\d+', filename)))
            return (tuple(numbers), filename) if numbers else ((), filename)

        return sorted(filenames, key=numeric_key)


def has_coco_targets(coco_label_path, img_id):
    """
    检查 COCO 格式的标签文件中是否包含指定图像的目标

    Args:
        coco_label_path (str): COCO 标签文件的路径
        img_id (int): 图像的 ID

    Returns:
        bool: 如果图像有目标，返回 True；否则返回 False
    """
    if not os.path.isfile(coco_label_path):
        return False

    with open(coco_label_path, 'r') as f:
        coco_annotation = json.load(f)

    # 查找图像的标注信息
    annotations = coco_annotation.get("annotations", [])
    img_annotations = [ann for ann in annotations if ann.get("image_id") == img_id]

    return len(img_annotations) > 0


def sort_filenames_numerically(filenames):
    def numeric_key(filename):
        numbers = list(map(int, re.findall(r'\d+', filename)))
        return (tuple(numbers), filename) if numbers else ((), filename)

    return sorted(filenames, key=numeric_key)


mscoco_category2name = {
    0: "people",
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}

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
    print('验证集第1个样本tir图像形状：', img_tir.shape, 'tir图像形状：', img_tir.shape, '标注形状：', label)

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
