# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: rgb_tir.py
@Time    : 2025/5/28 下午3:28
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : RGB图像生成伪红外/热成像图像
@Usage   : 
"""
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


def method1_thermal_colormap(rgb_image, colormap=cv2.COLORMAP_JET):
    """
    方法1: 热敏色彩映射 + 灰度图
    最常用的伪红外生成方法
    """
    # 转换为灰度图
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # 应用热敏色彩映射
    thermal = cv2.applyColorMap(gray, colormap)

    return thermal


def method2_enhanced_thermal(rgb_image, intensity_boost=1.2, contrast_boost=1.5):
    """
    方法2: 增强版热敏映射
    增加亮度对比度增强，模拟热源突出效果
    """
    # 转换为灰度
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # 增强对比度和亮度
    enhanced = cv2.convertScaleAbs(gray, alpha=contrast_boost, beta=20)

    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 应用热敏色彩映射
    thermal = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

    # 进一步调整饱和度
    hsv = cv2.cvtColor(thermal, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * intensity_boost  # 增强饱和度
    thermal_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return thermal_enhanced


def method3_multi_channel_thermal(rgb_image, heat_emphasis='human'):
    """
    方法3: 多通道加权热成像
    基于不同颜色通道模拟不同材质的热辐射特性
    """
    b, g, r = cv2.split(rgb_image)

    if heat_emphasis == 'human':
        # 人体热成像：红色通道权重高，模拟体温
        thermal_gray = 0.1 * b + 0.3 * g + 0.6 * r
    elif heat_emphasis == 'vegetation':
        # 植被热成像：绿色通道权重高
        thermal_gray = 0.2 * b + 0.6 * g + 0.2 * r
    elif heat_emphasis == 'building':
        # 建筑物热成像：蓝色通道权重高，模拟混凝土热辐射
        thermal_gray = 0.5 * b + 0.3 * g + 0.2 * r
    else:
        # 均衡权重
        thermal_gray = 0.33 * b + 0.33 * g + 0.34 * r

    thermal_gray = np.clip(thermal_gray, 0, 255).astype(np.uint8)

    # 应用热敏色彩映射
    thermal = cv2.applyColorMap(thermal_gray, cv2.COLORMAP_JET)

    return thermal


def method4_temperature_simulation(rgb_image, temp_range=(20, 40), noise_level=0.05):
    """
    方法4: 温度模拟热成像
    根据像素亮度模拟真实温度分布
    """
    # 转换为灰度
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # 归一化到温度范围
    normalized = gray.astype(np.float32) / 255.0
    temp_min, temp_max = temp_range
    temperature_map = normalized * (temp_max - temp_min) + temp_min

    # 添加温度噪声模拟真实热成像的噪声特性
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * (temp_max - temp_min), temperature_map.shape)
        temperature_map += noise

    # 重新归一化到0-255
    temp_normalized = ((temperature_map - temp_min) / (temp_max - temp_min) * 255)
    temp_normalized = np.clip(temp_normalized, 0, 255).astype(np.uint8)

    # 应用热敏色彩映射
    thermal = cv2.applyColorMap(temp_normalized, cv2.COLORMAP_JET)

    return thermal, temperature_map


def method5_edge_enhanced_thermal(rgb_image, edge_weight=0.3):
    """
    方法5: 边缘增强热成像
    突出物体轮廓，模拟热梯度效应
    """
    # 转换为灰度
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 将边缘融合到原图像
    enhanced = gray.astype(np.float32)
    enhanced += edges.astype(np.float32) * edge_weight
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # 应用热成像色彩映射
    thermal = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)

    return thermal


def method6_custom_thermal_lut(rgb_image, custom_colors=None):
    """
    方法6: 自定义热成像查找表
    使用自定义颜色映射模拟特定设备的热成像效果
    """
    if custom_colors is None:
        # 自定义热成像颜色方案：黑->蓝->紫->红->黄->白
        custom_colors = [
            [0, 0, 0],  # 冷 - 黑色
            [128, 0, 128],  # 较冷 - 紫色
            [0, 0, 255],  # 中等 - 蓝色
            [0, 255, 255],  # 中等偏热 - 青色
            [0, 255, 0],  # 热 - 绿色
            [255, 255, 0],  # 很热 - 黄色
            [255, 0, 0],  # 极热 - 红色
            [255, 255, 255]  # 最热 - 白色
        ]

    # 转换为灰度
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # 创建查找表
    colors = np.array(custom_colors, dtype=np.uint8)

    # 创建一个基于自定义颜色的查找表
    lut = np.zeros((256,), dtype=np.uint8)

    # 将颜色映射到查找表
    for i in range(256):
        # 确定颜色位置
        color_idx = int((i / 255.0) * (len(custom_colors) - 1))
        lut[i] = custom_colors[color_idx][0]  # 这里仅使用了一个通道，可能需要调整

    # 应用查找表
    thermal = cv2.LUT(gray, lut)
    thermal = cv2.cvtColor(thermal, cv2.COLOR_RGB2BGR)

    return thermal


def method7_infrared_simulation(rgb_image, ir_wavelength='lwir'):
    """
    方法7: 红外波段模拟
    模拟不同红外波段的成像特性
    """
    # 转换到不同色彩空间进行处理
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if ir_wavelength == 'lwir':  # 长波红外 (8-14μm)
        # 主要反映温度信息，基于亮度通道
        ir_base = v
        # 增强温度对比
        ir_enhanced = cv2.equalizeHist(ir_base)
    elif ir_wavelength == 'mwir':  # 中波红外 (3-5μm)
        # 结合温度和反射信息
        ir_base = cv2.addWeighted(v, 0.7, s, 0.3, 0)
        ir_enhanced = cv2.convertScaleAbs(ir_base, alpha=1.2, beta=10)
    elif ir_wavelength == 'swir':  # 短波红外 (1-3μm)
        # 更多反射特性，类似可见光但穿透性更强
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        ir_enhanced = cv2.bilateralFilter(gray, 9, 75, 75)
    else:
        ir_enhanced = v

    # 应用红外色彩映射
    thermal = cv2.applyColorMap(ir_enhanced, cv2.COLORMAP_INFERNO)

    return thermal


def compare_methods(rgb_image, save_comparison=True, output_path="thermal_comparison.jpg"):
    """
    比较所有方法的效果
    """
    methods = [
        ("Original RGB", rgb_image),
        ("Method 1: Basic Thermal", method1_thermal_colormap(rgb_image)),
        ("Method 2: Enhanced", method2_enhanced_thermal(rgb_image)),
        ("Method 3: Multi-channel", method3_multi_channel_thermal(rgb_image, 'human')),
        ("Method 4: Temperature Sim", method4_temperature_simulation(rgb_image)[0]),
        ("Method 5: Edge Enhanced", method5_edge_enhanced_thermal(rgb_image)),
        ("Method 6: Custom LUT", method6_custom_thermal_lut(rgb_image)),
        ("Method 7: LWIR Simulation", method7_infrared_simulation(rgb_image, 'lwir'))
    ]

    # 创建比较图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (title, image) in enumerate(methods):
        if i == 0:  # 原始RGB图像
            axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:  # 热成像图像
            axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()

    if save_comparison:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"比较图已保存到: {output_path}")

    # plt.show()

    return methods


def batch_process_rgb_to_thermal(input_dir, output_dir, method='enhanced',
                                 file_extensions=['.jpg', '.png', '.bmp']):
    """
    批量处理RGB图像转伪红外图像
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有图像文件
    image_files = []
    for ext in file_extensions:
        image_files.extend(list(input_path.glob(f"*{ext}")))
        image_files.extend(list(input_path.glob(f"*{ext.upper()}")))

    processed_count = 0

    for img_file in image_files:
        try:
            # 读取RGB图像
            rgb_img = cv2.imread(str(img_file))
            if rgb_img is None:
                continue

            # 选择转换方法
            if method == 'basic':
                thermal_img = method1_thermal_colormap(rgb_img)
            elif method == 'enhanced':
                thermal_img = method2_enhanced_thermal(rgb_img)
            elif method == 'multi_channel':
                thermal_img = method3_multi_channel_thermal(rgb_img, 'human')
            elif method == 'temperature':
                thermal_img, _ = method4_temperature_simulation(rgb_img)
            elif method == 'edge_enhanced':
                thermal_img = method5_edge_enhanced_thermal(rgb_img)
            elif method == 'custom_lut':
                thermal_img = method6_custom_thermal_lut(rgb_img)
            elif method == 'infrared':
                thermal_img = method7_infrared_simulation(rgb_img, 'lwir')
            else:
                thermal_img = method2_enhanced_thermal(rgb_img)  # 默认使用增强方法

            # 保存伪红外图像
            output_file = output_path / img_file.name
            cv2.imwrite(str(output_file), thermal_img)

            processed_count += 1
            print(f"处理完成: {img_file.name} -> {output_file.name}")

        except Exception as e:
            print(f"处理文件 {img_file.name} 时出错: {str(e)}")

    print(f"\n批量处理完成，共处理 {processed_count} 个文件")
    print(f"输出目录: {output_path}")


def process_dataset_structure(data_root, method='enhanced'):
    """
    处理符合您数据集结构的RGB到伪红外转换，支持以下结构：

    1. 标准结构:
       data/train/rgb -> data/train/tir_pseudo
       data/test/rgb -> data/test/tir_pseudo

    2. 扁平结构（无 train/test 子目录）:
       data/rgb -> data/tir_pseudo
    """
    data_root = Path(data_root)

    # 处理标准结构：train/test
    for split in ['train', 'test', 'val']:
        rgb_dir = data_root / split / 'rgb'
        tir_output_dir = data_root / split / 'tir_pseudo'

        if rgb_dir.exists():
            print(f"\n处理 {split} 数据集...")
            batch_process_rgb_to_thermal(
                input_dir=str(rgb_dir),
                output_dir=str(tir_output_dir),
                method=method
            )
        else:
            print(f"跳过不存在的目录: {rgb_dir}")

    # 处理扁平结构：直接在 data_root 下的 rgb 目录
    flat_rgb_dir = data_root / 'rgb'
    flat_tir_output_dir = data_root / 'tir'

    if flat_rgb_dir.exists():
        print("\n处理扁平结构数据集（无 train/test 子目录）...")
        batch_process_rgb_to_thermal(
            input_dir=str(flat_rgb_dir),
            output_dir=str(flat_tir_output_dir),
            method=method
        )


if __name__ == "__main__":
    # 使用示例

    # 1. 单张图像测试和比较
    test_image_path = "/sxs/DataSets/DroneRGBTs/rgb/0.jpg"  # 替换为您的测试图像路径

    if os.path.exists(test_image_path):
        rgb_img = cv2.imread(test_image_path)

        # 比较所有方法
        compare_methods(rgb_img, save_comparison=True,
                        output_path="/sxs/DataSets/DroneRGBTs/thermal_methods_comparison.jpg")

        # # 生成单个最佳效果的伪红外图像
        best_thermal = method1_thermal_colormap(rgb_img)
        cv2.imwrite("/sxs/DataSets/DroneRGBTs/best_thermal_result.jpg", best_thermal)
        print("最佳伪红外图像已保存为: best_thermal_result.jpg")

    # 2. 批量处理整个数据集
    data_root = "/sxs/DataSets/DroneRGBTs/"  # 替换为您的数据根目录

    # 可选的方法: 'basic', 'enhanced', 'multi_channel', 'temperature',
    #           'edge_enhanced', 'custom_lut', 'infrared'
    process_dataset_structure(data_root, method='basic')

    # print("\n推荐使用方法:")
    # print("- 'enhanced': 适合大多数场景，效果好且快速")
    # print("- 'multi_channel': 适合人体检测，突出人体热特征")
    # print("- 'temperature': 模拟真实温度分布，适合科研用途")
    # print("- 'edge_enhanced': 适合需要突出物体轮廓的场景")
