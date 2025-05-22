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
    img = (img * 255).astype(np.uint8)  # [0,1] -> [0,255]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR for OpenCV


def draw_boxes(image, boxes, labels=None, label_map=None, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制边界框
    Args:
        image: NumPy array (H, W, C)
        boxes: Tensor of shape (N, 4), normalized coordinates
        labels: Tensor of shape (N,)
        label_map: Optional dict {label_id: label_name}
        color: BGR 颜色
    Returns:
        image with boxes
    """
    h, w, _ = image.shape
    boxes_abs = boxes.clone()
    boxes_abs[:, [0, 2]] *= w  # x
    boxes_abs[:, [1, 3]] *= h  # y
    boxes_abs = boxes_abs.round().int()

    for i, box in enumerate(boxes_abs):
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        if labels is not None and label_map:
            label = int(labels[i])
            text = label_map.get(label, f'{label}')
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
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
    denorm_tir = DeNormalize(mean=[0.492, 0.168, 0.430], std=[0.317, 0.174, 0.191])

    rgb_img = tensor_to_image(denorm_rgb(img_rgb.clone()))
    tir_img = tensor_to_image(denorm_tir(img_tir.clone()))

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
