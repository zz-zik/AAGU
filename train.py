# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: train.py
@Time    : 2025/5/19 下午2:31
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
# """
# from engine import train
# from utils import load_config, get_output_dir, setup_logging
# # 启动训练
# if __name__ == "__main__":
#     cfg = load_config('./configs/config.yaml')
#     output_dir = get_output_dir(cfg.output_dir, cfg.name)
#     cfg.output_dir = output_dir
#     logger = setup_logging(cfg, output_dir)
#     train(cfg)


if __name__ == "__main__":
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
        print(f"Batch {i + 1}:")
        print("RGB images shape:", rgb_images.shape)
        print("TIR images shape:", tir_images.shape)
        print("Targets:", targets)

    # 遍历验证数据加载器
    print("\nValidation data:")
    for i, (rgb_images, tir_images, targets) in enumerate(data_loader_val):
        print(f"Batch {i + 1}:")
        print("RGB images shape:", rgb_images.shape)
        print("TIR images shape:", tir_images.shape)
        print("Targets:", targets)