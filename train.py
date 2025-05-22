# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: train.py
@Time    : 2025/5/19 下午2:31
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
"""
from engine import training
from utils import load_config
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    cfg = load_config('./configs/config.yaml')
    training(cfg)


def data_loader_test():
    from utils import load_config
    from dataloader import build_dataset

    cfg = load_config('./configs/config.yaml')
    data_loader_train, data_loader_val = build_dataset(cfg)

    # 查看数据加载器的长度
    print("Training DataLoader length:", len(data_loader_train))
    print("Validation DataLoader length:", len(data_loader_val))

    # 遍历训练数据加载器
    print("Training data:")
    for i, (rgb_images, tir_images, targets) in enumerate(data_loader_train):
        print(f"Train Batch {i+1}/{len(data_loader_train)}:")
        print("RGB images shape:", rgb_images.shape)
        print("TIR images shape:", tir_images.shape)
        print("Targets:", targets)

    # 遍历验证数据加载器
    print("\nValidation data:")
    for i, (rgb_images, tir_images, targets) in enumerate(data_loader_val):
        print(f"Val Batch {i + 1}:")
        print("RGB images shape:", rgb_images.shape)
        print("TIR images shape:", tir_images.shape)
        print("Targets:", targets)


if __name__ == "__main__":
    main()
    # data_loader_test()