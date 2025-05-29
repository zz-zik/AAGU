# -*- coding: utf-8 -*-
"""
@Project : RT-FINE
@FileName: transforms.py
@Time    : 2025/5/14 下午9:42
@Author  : ZhouFei
@Email   : zhoufei21@s.nuit.edu.cn
@Desc    : 
@Usage   :
"""
import random

import torchvision
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from torchvision.transforms import functional as F, Compose, Normalize


class Transforms(nn.Module):
    """根据配置参数动态选择和组合数据增强操作"""

    def __init__(self, train=True, box_fmt='cxcywh', **kwargs):
        super().__init__()
        self.train = train
        self.box_fmt = box_fmt

        # 定义标准化变换
        # self.rgb_transform = Compose([
        #     Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     ),
        # ])
        #
        # self.tir_transform = Compose([
        #     Normalize(
        #         mean=[0.492, 0.168, 0.430],
        #         std=[0.317, 0.174, 0.191]
        #     ),
        # ])
        self.rgb_transform = Compose([
            Normalize(
                mean=[0.341, 0.355, 0.258],
                std=[0.131, 0.135, 0.118]
            ),
        ])

        self.tir_transform = Compose([
            Normalize(
                mean=[0.615, 0.116, 0.374],
                std=[0.236, 0.156, 0.188]
            ),
        ])

        # 获取数据增强的参数
        self.transforms = []

        transform_params = {
            'RandomFlip': kwargs.get('random_flip_prob', 0.0),
            'RandomRotation': kwargs.get('random_rotation', 0.0),
            'RandomResize': kwargs.get('random_resize', (0.0, 0.0)),
            'ColorJitter': kwargs.get('color_jitter', (0.0, 0.0, 0.0, 0.0)),
            'GammaCorrection': kwargs.get('gamma_correction', (0.0, 0.0)),
            'RandomErasing': kwargs.get('random_erase_prob', 0.0),
            'GaussianBlur': kwargs.get('blur_sigma_prob', (0.0, 0.0, 0.0)),
            'RandomJitter': kwargs.get('jitter_drift', 0.0),
        }
        prob = kwargs.get('prob', 0.0)

        for name, param in transform_params.items():
            # 第一个值要大于0
            if isinstance(param, (list, tuple)) and param[0] > 0.0:
                if name == 'RandomResize':
                    if isinstance(param, (list, tuple)) and len(param) == 2:
                        scale_range = (float(param[0]), float(param[1]))
                        self.transforms.append(RandomResize(scale_range=scale_range, prob=prob))
                elif name == 'ColorJitter':
                    if isinstance(param, (list, tuple)) and len(param) == 4:
                        brightness, contrast, saturation, hue = param
                        self.transforms.append(
                            ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue,
                                        prob=prob))
                elif name == 'GammaCorrection':
                    if isinstance(param, (list, tuple)) and len(param) == 2:
                        gamma_range = (float(param[0]), float(param[1]))
                        self.transforms.append(GammaCorrection(gamma_range=gamma_range, prob=prob))
                elif name == 'GaussianBlur':
                    if isinstance(param, (list, tuple)) and len(param) == 3:
                        kernel_size = int(param[0])
                        sigma_range = (float(param[1]), float(param[2]))
                        self.transforms.append(GaussianBlur(kernel_size=kernel_size, sigma=sigma_range, prob=prob))
            elif isinstance(param, (int, float)) and param > 0.0:
                if name == 'RandomFlip':
                    self.transforms.append(RandomFlip(prob=param))
                elif name == 'RandomRotation':
                    self.transforms.append(RandomRotation(degrees=param, prob=prob))
                elif name == 'RandomErasing':
                    self.transforms.append(RandomErasing(p=param, scale=(0.02, 0.33), ratio=(0.3, 3.3)))
                elif name == 'RandomJitter':
                    self.transforms.append(RandomJitter(max_drift=param, prob=prob))

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, target: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Any]]:

        if not isinstance(rgb, torch.Tensor) and not isinstance(tir, torch.Tensor):
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
            tir = torch.from_numpy(tir).permute(2, 0, 1).float()

        # 应用数据增强操作
        if self.train:
            # 如果是 cxcywh 格式，先转换成 xyxy 以便统一处理
            original_boxes = target.get("boxes", torch.empty((0, 4), device=rgb.device))
            if self.box_fmt == 'cxcywh' and len(original_boxes) > 0:
                boxes_xyxy = cxcywh_to_xyxy(original_boxes)
                target["boxes"] = boxes_xyxy
            for transform in self.transforms:
                rgb, tir, target = transform(rgb, tir, target)

            # 恢复原始的 box_fmt 格式（如 cxcywh）
            if self.box_fmt == 'cxcywh' and len(target.get("boxes", [])) > 0:
                target["boxes"] = xyxy_to_cxcywh(target["boxes"])

        # 应用标准化变换
        rgb = self.rgb_transform(rgb)
        tir = self.tir_transform(tir)

        return rgb, tir, target


class RandomFlip(nn.Module):
    """随机水平或垂直翻转"""

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, target: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if torch.rand(1) < self.prob:
            # 随机选择水平或垂直翻转
            if torch.rand(1) < 0.5:
                rgb = F.hflip(rgb)
                tir = F.hflip(tir)
                # 更新目标框（仅当 boxes 存在时）
                if "boxes" in target and len(target["boxes"]) > 0:
                    boxes = target["boxes"]
                    boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
                    target["boxes"] = boxes
            else:
                rgb = F.vflip(rgb)
                tir = F.vflip(tir)
                # 更新目标框（仅当 boxes 存在时）
                if "boxes" in target and len(target["boxes"]) > 0:
                    boxes = target["boxes"]
                    boxes[:, [1, 3]] = 1.0 - boxes[:, [3, 1]]
                    target["boxes"] = boxes

        return rgb, tir, target


class RandomRotation(nn.Module):
    """随机旋转"""

    def __init__(self, degrees: float = 10.0, prob: float = 0.5):
        super().__init__()
        self.degrees = degrees
        self.prob = prob

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, target: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if torch.rand(1) < self.prob:
            _, h, w = rgb.shape
            angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees))

            # 获取仿射变换矩阵（仅旋转）
            matrix = torchvision.transforms.functional._get_inverse_affine_matrix(
                center=(w / 2, h / 2),
                angle=angle,
                translate=(0., 0.),
                scale=1.0,
                shear=(0.0, 0.0)
            )
            matrix = torch.tensor(matrix, dtype=torch.float32, device=rgb.device).view(2, 3)

            # 应用变换
            rgb = F.rotate(rgb, angle)
            tir = F.rotate(tir, angle)

            # 更新目标框
            if "boxes" in target and len(target["boxes"]) > 0:
                target["boxes"] = apply_affine_to_boxes(target["boxes"], matrix, w, h)

        return rgb, tir, target


class RandomResize(nn.Module):
    """随机缩放并保持原尺寸"""

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), prob: float = 0.5):
        super().__init__()
        self.scale_range = scale_range
        self.prob = prob

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, target: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if torch.rand(1) < self.prob:
            scale = float(torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]))
            img_h, img_w = rgb.shape[1], rgb.shape[2]
            new_size = (int(img_h * scale), int(img_w * scale))

            # 缩放图像
            rgb = F.resize(rgb, new_size)
            tir = F.resize(tir, new_size)

            # 更新目标框
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes *= torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)  # 归一化 -> 绝对坐标
                boxes *= scale  # 缩放

                # 裁剪或填充图像和边界框
                if scale > 1.0:  # 放大：裁剪到原始尺寸
                    top = (rgb.shape[1] - img_h) // 2
                    left = (rgb.shape[2] - img_w) // 2
                    rgb = rgb[:, top:top + img_h, left:left + img_w]
                    tir = tir[:, top:top + img_h, left:left + img_w]
                    boxes -= torch.tensor([left, top, left, top], device=boxes.device)  # 调整边界框坐标
                    boxes /= torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)  # 再次归一化
                else:  # 缩小：填充到原始尺寸
                    pad_top = (img_h - rgb.shape[1]) // 2
                    pad_left = (img_w - rgb.shape[2]) // 2
                    pad_bottom = img_h - rgb.shape[1] - pad_top
                    pad_right = img_w - rgb.shape[2] - pad_left
                    rgb = F.pad(rgb, (pad_left, pad_top, pad_right, pad_bottom))
                    tir = F.pad(tir, (pad_left, pad_top, pad_right, pad_bottom))
                    boxes += torch.tensor([pad_left, pad_top, pad_left, pad_top], device=boxes.device)  # 调整边界框坐标
                    boxes /= torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)  # 再次归一化

                target["boxes"] = boxes

            # 确保最终尺寸一致
            rgb = F.resize(rgb, (img_h, img_w))
            tir = F.resize(tir, (img_h, img_w))

        assert rgb.shape[1:] == tir.shape[1:], "RGB 和 TIR 图像尺寸不一致"
        return rgb, tir, target



class ColorJitter(nn.Module):
    """颜色扰动"""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2, hue: float = 0.1,
                 prob: float = 0.5):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.prob = prob

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, target: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if torch.rand(1) < self.prob:
            rgb = self.color_jitter(rgb)
        return rgb, tir, target


class GammaCorrection(nn.Module):
    """Gamma校正"""

    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2), prob: float = 0.5):
        super().__init__()
        self.gamma_range = gamma_range
        self.prob = prob

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, target: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if torch.rand(1) < self.prob:
            gamma = float(torch.empty(1).uniform_(self.gamma_range[0], self.gamma_range[1]))
            rgb = F.adjust_gamma(rgb, gamma)
        return rgb, tir, target


class AffineTransform(nn.Module):
    """仿射变换"""

    def __init__(self, degrees: float = 10.0, translate: Tuple[float, float] = (0.1, 0.1),
                 scale: Tuple[float, float] = (0.9, 1.1), shear: float = 5.0, prob: float = 0.5):
        super().__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.prob = prob

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, target: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if torch.rand(1) < self.prob:
            _, h, w = rgb.shape
            angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees))
            translate_x = float(torch.empty(1).uniform_(-self.translate[0], self.translate[0]))
            translate_y = float(torch.empty(1).uniform_(-self.translate[1], self.translate[1]))
            scale = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]))
            shear = float(torch.empty(1).uniform_(-self.shear, self.shear))

            # 获取仿射变换矩阵
            matrix = torchvision.transforms.functional._get_inverse_affine_matrix(
                center=(w / 2, h / 2),
                angle=angle,
                translate=(translate_x * w, translate_y * h),
                scale=scale,
                shear=shear
            )
            matrix = torch.tensor(matrix, dtype=torch.float32, device=rgb.device).view(2, 3)

            # 应用变换
            rgb = F.affine(rgb, angle, (translate_x, translate_y), scale, shear)
            tir = F.affine(tir, angle, (translate_x, translate_y), scale, shear)

            # 更新 boxes
            if "boxes" in target and len(target["boxes"]) > 0:
                target["boxes"] = apply_affine_to_boxes(target["boxes"], matrix, w, h)

        return rgb, tir, target


def apply_affine_to_boxes(boxes: torch.Tensor, matrix: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """
    Apply affine transformation to boxes in normalized [x_min, y_min, x_max, y_max] format.

    Args:
        boxes (Tensor): shape (N, 4), normalized coordinates
        matrix (Tensor): affine matrix of shape (2, 3)
        img_w (int): image width
        img_h (int): image height

    Returns:
        Tensor: transformed boxes in normalized format
    """
    if len(boxes) == 0:
        return boxes  # 直接返回空 boxes

    # Convert boxes from normalized [0, 1] to absolute pixel values
    boxes_abs = boxes * torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)

    # Convert boxes to corner points
    points = torch.cat([
        boxes_abs[:, :2].unsqueeze(1),  # top-left
        torch.stack([boxes_abs[:, 2], boxes_abs[:, 1]], dim=1).unsqueeze(1),  # top-right
        boxes_abs[:, 2:].unsqueeze(1),  # bottom-right
        torch.stack([boxes_abs[:, 0], boxes_abs[:, 3]], dim=1).unsqueeze(1)  # bottom-left
    ], dim=1)  # (N, 4, 2)

    # Add homogeneous coordinate
    ones = torch.ones(points.shape[0], points.shape[1], 1, device=points.device)
    points_homogeneous = torch.cat([points, ones], dim=-1)  # (N, 4, 3)

    # Apply affine transform
    transformed_points = torch.matmul(points_homogeneous, matrix.T)  # (N, 4, 2)

    # Extract new coordinates
    new_x = transformed_points[:, :, 0]
    new_y = transformed_points[:, :, 1]

    # Build new box
    new_boxes_abs = torch.stack([
        new_x.min(dim=1).values,
        new_y.min(dim=1).values,
        new_x.max(dim=1).values,
        new_y.max(dim=1).values,
    ], dim=1)  # (N, 4)

    # Clamp and normalize
    new_boxes_abs = new_boxes_abs.clamp(min=0)
    new_boxes = new_boxes_abs / torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
    return new_boxes.clamp(max=1.0)


class RandomErasing(nn.Module):
    """随机擦除"""

    def __init__(self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33),
                 ratio: Tuple[float, float] = (0.3, 3.3), value: int = 0):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, target: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if torch.rand(1) < self.p:
            if torch.rand(1) < 0.5:
                rgb = torchvision.transforms.RandomErasing(p=1.0, scale=self.scale, ratio=self.ratio, value=self.value)(
                    rgb)
            else:
                tir = torchvision.transforms.RandomErasing(p=1.0, scale=self.scale, ratio=self.ratio, value=self.value)(
                    tir)
        return rgb, tir, target


class GaussianBlur(nn.Module):
    """高斯模糊"""

    def __init__(self, kernel_size: int = 5, sigma: Tuple[float, float] = (0.1, 2.0), prob: float = 0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.prob = prob

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, target: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if torch.rand(1) < self.prob:
            sigma = float(torch.empty(1).uniform_(self.sigma[0], self.sigma[1]))
            rgb = F.gaussian_blur(rgb, kernel_size=self.kernel_size, sigma=sigma)
        return rgb, tir, target


class RandomJitter(nn.Module):
    """空间抖动增强：通过裁剪 + 缩放实现RGB图像的偏移，不依赖填充"""

    def __init__(self, max_drift: float = 0.01, prob: float = 0.5):
        """
        Args:
            max_drift (float): 裁剪比例，表示裁剪区域比原图小多少（范围 [0, 1)）
            prob (float): 触发该变换的概率
        """
        super().__init__()
        if not 0 <= max_drift < 1:
            raise ValueError(f"max_drift must be in range [0, 1), got {max_drift}")
        self.max_drift = max_drift
        self.prob = prob

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, target: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if torch.rand(1) < self.prob:
            _, h, w = rgb.shape

            # 计算裁剪区域大小
            crop_h = int(h * (1 - self.max_drift))
            crop_w = int(w * (1 - self.max_drift))

            # 防止裁剪区域过小
            crop_h = max(crop_h, 16)
            crop_w = max(crop_w, 16)

            # 随机选择裁剪起点
            start_h = random.randint(0, h - crop_h)
            start_w = random.randint(0, w - crop_w)

            # 对裁剪区域进行轻微偏移（防止边缘越界）
            offset_range = min(h - start_h - crop_h, w - start_w - crop_w, 10)
            if offset_range > 0:
                dx = random.randint(-offset_range, offset_range)
                dy = random.randint(-offset_range, offset_range)
                start_h += dy
                start_w += dx

                # 确保不越界
                start_h = max(0, min(start_h, h - crop_h))
                start_w = max(0, min(start_w, w - crop_w))

            # 裁剪并缩放回原图尺寸
            cropped_rgb = rgb[:, start_h:start_h + crop_h, start_w:start_w + crop_w]
            resized_rgb = F.resize(cropped_rgb, size=(h, w))

            return resized_rgb, tir, target

        return rgb, tir, target


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """将 [cx, cy, w, h] 转换为 [x1, y1, x2, y2]"""
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """将 [x1, y1, x2, y2] 转换为 [cx, cy, w, h]"""
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


if __name__ == "__main__":
    # 创建示例输入数据
    rgb_image = torch.randn(3, 512, 640)  # RGB 图像 (channel, height, width)
    tir_image = torch.randn(3, 512, 640)  # TIR 图像 (channel, height, width)
    target = {
        "boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]]),  # 目标边界框
        "labels": torch.tensor([1, 2])  # 目标类别标签
    }

    # 定义数据增强操作的参数
    transform_params = {
        "prob": 0.6,
        "random_flip": 0.5,
        "random_rotation": 10.0,
        "random_resize": [0.6, 1.4],
        "color_jitter": [0.2, 0.2, 0.2, 0.2],
        "gamma_correction": [0.8, 1.2],
        "random_erase": 0.0,
        "blur_sigma": [5.0, 0.1, 2.0],
        "jitter_drift": 0.01,
    }

    # 创建 Transforms 实例
    transforms = Transforms(train=True, **transform_params)

    # 应用数据增强
    rgb_transformed, tir_transformed, target_transformed = transforms(rgb_image, tir_image, target)

    # 打印结果
    print("Original RGB Image Shape:", rgb_image.shape)
    print("Transformed RGB Image Shape:", rgb_transformed.shape)
    print("Original TIR Image Shape:", tir_image.shape)
    print("Transformed TIR Image Shape:", tir_transformed.shape)
    print("Original Target:", target)
    print("Transformed Target:", target_transformed)
