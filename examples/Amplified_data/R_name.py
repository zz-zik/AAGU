# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: R_name.py
@Time    : 2025/5/28 下午2:20
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 将文件名中的 最后一个 R 删除（不区分大小写）
@Usage   : 
"""
import os


def remove_r_from_filenames(folder_path):
    valid_extensions = ('.jpg', '.xml')

    for filename in os.listdir(folder_path):
        name, ext = os.path.splitext(filename)

        if ext.lower() not in valid_extensions:
            continue

        # 从后往前找第一个 R/r
        idx = None
        for i in reversed(range(len(name))):
            if name[i].lower() == 'r':
                idx = i
                break

        if idx is not None:
            new_name = name[:idx] + name[idx + 1:] + ext
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")


# 示例用法
if __name__ == "__main__":
    folder = "/sxs/zhoufei/P2PNet/DroneRGBT/val/labels"  # 替换为你的文件夹路径
    remove_r_from_filenames(folder)
