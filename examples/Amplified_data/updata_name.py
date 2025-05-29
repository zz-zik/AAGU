# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: updata_name.py
@Time    : 2025/5/28 下午3:44
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 更新名称
@Usage   : 
"""
import os
from tkinter import Tk, filedialog


def rename_files(root_dir, start_number):
    subsets = ['train', 'test', 'val']
    file_types = ['rgb', 'tir', 'labels', 'annotations']

    for subset in subsets:
        subset_dir = os.path.join(root_dir, subset)
        if not os.path.exists(subset_dir):
            print(f"警告：未找到{subset}文件夹，跳过")
            continue

        # 获取 rgb 文件夹中的文件列表作为基准
        rgb_dir = os.path.join(subset_dir, 'rgb')
        if not os.path.exists(rgb_dir):
            print(f"警告：未找到{subset}/rgb文件夹，跳过")
            continue

        files = os.listdir(rgb_dir)
        files.sort()  # 确保顺序一致

        for idx, file_name in enumerate(files):
            base_name = os.path.splitext(file_name)[0]
            new_name = f"{start_number + idx}"

            for file_type in file_types:
                file_type_dir = os.path.join(subset_dir, file_type)
                if not os.path.exists(file_type_dir):
                    print(f"警告：未找到{subset}/{file_type}文件夹，跳过")
                    continue

                # 构造原始文件名 - 使用正确的扩展名
                if file_type in ['rgb', 'tir']:
                    original_ext = '.jpg'  # RGB和TIR图像使用.jpg
                elif file_type == 'annotations':
                    original_ext = '.txt'  # 注释文件使用.txt
                elif file_type == 'labels':
                    original_ext = '.xml'  # 标签文件使用.xml
                else:
                    _, original_ext = os.path.splitext(file_name)  # 其他情况保留原扩展名

                old_path = os.path.join(file_type_dir, f"{base_name}{original_ext}")

                # 如果文件不存在且不是'tir'类型，尝试其他常见扩展名
                if not os.path.exists(old_path) and file_type != 'tir':
                    for ext in ['.png', '.jpeg', '.bmp', '.jpg']:
                        if os.path.exists(os.path.join(file_type_dir, f"{base_name}{ext}")):
                            old_path = os.path.join(file_type_dir, f"{base_name}{ext}")
                            break

                # 如果文件仍不存在
                if not os.path.exists(old_path):
                    print(f"警告：未找到文件 {old_path}，跳过")
                    continue

                # 构造新文件名
                if file_type in ['rgb', 'tir']:
                    new_file_name = f"{new_name}.jpg"
                elif file_type == 'annotations':
                    new_file_name = f"{new_name}.txt"
                elif file_type == 'labels':
                    new_file_name = f"{new_name}.xml"
                else:
                    _, original_ext = os.path.splitext(file_name)
                    new_file_name = f"{new_name}{original_ext}"

                new_path = os.path.join(file_type_dir, new_file_name)

                try:
                    os.rename(old_path, new_path)
                    print(f"重命名: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"错误：无法重命名 {old_path} -> {new_path}: {str(e)}")

        start_number += len(files)


def main():
    # 选择根目录
    root_dir = '/sxs/DataSets/DroneRGBTs'

    start_number = 32000
    # 执行重命名
    rename_files(root_dir, start_number)
    print("重命名完成！")


if __name__ == "__main__":
    main()