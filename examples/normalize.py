# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: normalize.py
@Time    : 2025/5/25 上午11:19
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 计算图像的归一化均值和标准差
@Usage   :
处理图像进度: 100%|███████████████████████| 22131/22131 [10:50<00:00, 34.02it/s]
✅所有图像处理完成！
RGB 所有图像的平均均值（每个通道）： [0.35494163 0.36840918 0.27330935]
RGB 所有图像的平均标准差（每个通道）： [0.13099624 0.13373924 0.11775094]
处理图像进度: 100%|███████████████████████| 22131/22131 [11:33<00:00, 31.92it/s]
✅所有图像处理完成！
TIR 所有图像的平均均值（每个通道）： [0.61420712 0.14948837 0.37343189]
TIR 所有图像的平均标准差（每个通道）： [0.2323119  0.15830037 0.18179071]
"""
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np


def calculate_mean_std(image_path, image_type="RGB"):
    # 使用 PIL 读取图像
    image = Image.open(image_path).convert("RGB")  # 确保图像为 RGB 格式
    if image is None:
        print(f"无法加载图像: {image_path}")
        return None, None

    image = np.array(image)  # 转换为 NumPy 数组

    # 根据图像类型进行归一化
    # 根据图像类型进行归一化
    if image_type == "RGB" or image_type == "TIR":
        # 归一化到 0-1 范围
        image_normalized = image / 255.0
    else:
        print(f"不支持的图像类型: {image_type}")
        return None, None

    # 计算每个通道的均值和标准差
    mean = np.mean(image_normalized, axis=(0, 1))
    std = np.std(image_normalized, axis=(0, 1))
    return mean, std


# 定义一个函数来批量处理图像
def batch_calculate_mean_std(image_folder, image_extension="*.jpg", image_type="RGB"):
    # 获取文件夹中所有图像的路径
    image_paths = glob.glob(f"{image_folder}/{image_extension}")
    if not image_paths:
        print("未找到图像文件，请检查路径和扩展名。")
        return

    # 初始化用于存储所有图像的均值和标准差的列表
    means = []
    stds = []

    # 遍历每个图像并计算均值和标准差，并添加进度条
    for path in tqdm(image_paths, desc="处理图像进度"):  # 添加 tqdm 进度条
        mean, std = calculate_mean_std(path, image_type)
        if mean is not None and std is not None:
            means.append(mean)
            stds.append(std)

    # 计算所有图像的平均均值和平均标准差
    overall_mean = np.mean(means, axis=0)
    overall_std = np.mean(stds, axis=0)

    print("✅所有图像处理完成！")
    return overall_mean, overall_std


if __name__ == "__main__":
    # 设置图像文件夹路径
    rgb_folder = "/sxs/zhoufei/AAGU/dataset/OdinMJ2/train/RGB"  # 替换为你的图像文件夹路径
    rgb_extension = "*.jpg"  # 替换为你的图像文件扩展名，例如 "*.png" 或 "*.bmp"
    rgb_type = "RGB"
    overall_mean, overall_std = batch_calculate_mean_std(rgb_folder, rgb_extension, rgb_type)
    print(f"{rgb_type} 所有图像的平均均值（每个通道）： {overall_mean}")
    print(f"{rgb_type} 所有图像的平均标准差（每个通道）： {overall_std}")

    tir_folder = "/sxs/zhoufei/AAGU/dataset/OdinMJ2/train/TIR"  # 替换为你的图像文件夹路径
    tir_extension = "*.jpg"  # 替换为你的图像文件扩展名，例如 "*.png" 或 "*.bmp"
    tir_type = "TIR"
    # 调用函数并打印结果
    overall_mean, overall_std = batch_calculate_mean_std(tir_folder, tir_extension, tir_type)
    print(f"{tir_type} 所有图像的平均均值（每个通道）： {overall_mean}")
    print(f"{tir_type} 所有图像的平均标准差（每个通道）： {overall_std}")
