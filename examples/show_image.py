# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: show_image.py
@Time    : 2025/5/21 上午10:43
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 测试数据和标签
@Usage   :
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from dataloader import loading_data, DeNormalize


def tensor_to_image(img_tensor):
    """将 PyTorch Tensor 转换为 NumPy 图像（HWC 格式）"""
    img = img_tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR for OpenCV


def draw_boxes(image, boxes, labels=None, label_map=None, color=(0, 255, 0), thickness=2, box_fmt='cxcywh'):
    """
    在图像上绘制边界框
    Args:
        image: NumPy array (H, W, C)
        boxes: Tensor of shape (N, 4), normalized coordinates
        labels: Tensor of shape (N,)
        label_map: Optional dict {label_id: label_name}
        color: BGR 颜色
        thickness: 线条粗细
        box_fmt: 边界框格式，支持 'xyxy' (左上/右下) 或 'cxcywh' (中心 + 宽高)
    Returns:
        image with boxes
    """
    h, w, _ = image.shape
    boxes_abs = boxes.clone()

    if box_fmt == 'cxcywh':
        # 转换 cxcywh 为 x1, y1, x2, y2
        cx = boxes_abs[:, 0] * w
        cy = boxes_abs[:, 1] * h
        cw = boxes_abs[:, 2] * w
        ch = boxes_abs[:, 3] * h
        x1 = (cx - cw / 2).round().int().tolist()
        y1 = (cy - ch / 2).round().int().tolist()
        x2 = (cx + cw / 2).round().int().tolist()
        y2 = (cy + ch / 2).round().int().tolist()
    elif box_fmt == 'xyxy':
        # 直接使用归一化坐标并转换为绝对坐标
        x1 = (boxes_abs[:, 0] * w).round().int().tolist()
        y1 = (boxes_abs[:, 1] * h).round().int().tolist()
        x2 = (boxes_abs[:, 2] * w).round().int().tolist()
        y2 = (boxes_abs[:, 3] * h).round().int().tolist()
    else:
        raise ValueError(f"不支持的边界框格式: {box_fmt}")

    for i in range(len(x1)):
        pt1 = (x1[i], y1[i])
        pt2 = (x2[i], y2[i])
        cv2.rectangle(image, pt1, pt2, color, thickness)

        if labels is not None and label_map:
            label = int(labels[i])
            text = label_map.get(label, f'{label}')
            cv2.putText(image, text, (x1[i], y1[i] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return image


# 示例 label_map（根据你的类别修改）
label_map = {
    0: 'people',
}

if __name__ == '__main__':
    from utils import load_config

    cfg = load_config("../configs/config.yaml")
    cfg.data.data_root = '../dataset/OdinMJ'
    train_dataset, val_dataset = loading_data(cfg)

    print('训练集样本数：', len(train_dataset))
    print('测试集样本数：', len(val_dataset))

    idx = 1
    img_rgb, img_tir, target = train_dataset[idx]

    # 转换为图像
    denorm_rgb = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    denorm_tir = DeNormalize(mean=[0.615, 0.116, 0.374], std=[0.236, 0.156, 0.188])

    # rgb_img = tensor_to_image(denorm_rgb(img_rgb.clone()))
    # tir_img = tensor_to_image(denorm_tir(img_tir.clone()))

    rgb_img = tensor_to_image(img_rgb.clone())
    tir_img = tensor_to_image(img_tir.clone())

    # 绘制边界框
    boxes = target['boxes']
    labels = target['labels']

    rgb_with_boxes = draw_boxes(rgb_img.copy(), boxes, labels, label_map=label_map)
    tir_with_boxes = draw_boxes(tir_img.copy(), boxes, labels, label_map=label_map)  # 新增这一行

    # 显示图像
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("RGB Image with Boxes")
    plt.imshow(cv2.cvtColor(rgb_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("TIR Image with Boxes")
    plt.imshow(cv2.cvtColor(tir_with_boxes, cv2.COLOR_BGR2RGB))  # 修改这里
    plt.axis('off')

    plt.savefig("output.png")  # 保存图像到文件
    print("图像已保存为 output.png")
