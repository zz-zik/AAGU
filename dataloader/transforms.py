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

import torchvision
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from torchvision.transforms import functional as F, Compose, ToTensor, Normalize


class Transforms(nn.Module):
    """根据配置参数动态选择和组合数据增强操作"""

    def __init__(self, train=True, **kwargs):
        super().__init__()
        self.train = train

        # 定义标准化变换
        self.rgb_transform = Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.tir_transform = Compose([
            Normalize(
                mean=[0.492, 0.168, 0.430],
                std=[0.317, 0.174, 0.191]
            ),
        ])

        # 获取数据增强的参数
        self.random_flip = kwargs.get('random_flip', 0.0)
        self.random_rotation = kwargs.get('random_rotation', 0.0)
        self.random_resize = kwargs.get('random_resize', 0.0)
        self.color_jitter = kwargs.get('color_jitter', 0.0)
        self.gamma_correction = kwargs.get('gamma_correction', 0.0)
        self.random_erase = kwargs.get('random_erase', 0.0)
        self.blur_sigma = kwargs.get('blur_sigma', 0.0)

        # 构建数据增强操作列表
        self.transforms = []
        if self.random_flip > 0.0:
            self.transforms.append(RandomFlip(self.random_flip))
        if self.random_rotation > 0.0:
            self.transforms.append(RandomRotation(self.random_rotation))
        if self.random_resize > 0.0:
            self.transforms.append(RandomResize(prob=self.random_resize))
        if self.color_jitter > 0.0:
            self.transforms.append(ColorJitter(self.color_jitter))
        if self.gamma_correction > 0.0:
            self.transforms.append(GammaCorrection(prob=self.gamma_correction))
        if self.random_erase > 0.0:
            self.transforms.append(RandomErasing(self.random_erase))
        if self.blur_sigma > 0.0:
            self.transforms.append(GaussianBlur(prob=self.blur_sigma))

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, target: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """应用数据增强操作

        Args:
            rgb (torch.Tensor): RGB 图像
            tir (torch.Tensor): TIR 图像
            target (Dict[str, Any]): 目标数据

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: 增强后的 RGB 图像、TIR 图像和目标数据
        """
        # 应用数据增强操作
        if self.train:
            for transform in self.transforms:
                rgb, tir, target = transform(rgb, tir, target)

        # 判断rgb和tir的格式是否是tensor格式,如果不是，则转换为tensor
        if not isinstance(rgb, torch.Tensor) and not isinstance(tir, torch.Tensor):
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
            tir = torch.from_numpy(tir).permute(2, 0, 1).float()

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
                # 更新目标框
                target["boxes"][:, [0, 2]] = 1.0 - target["boxes"][:, [2, 0]]
            else:
                rgb = F.vflip(rgb)
                tir = F.vflip(tir)
                # 更新目标框
                target["boxes"][:, [1, 3]] = 1.0 - target["boxes"][:, [3, 1]]
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
            angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees))
            rgb = F.rotate(rgb, angle)
            tir = F.rotate(tir, angle)
            # 更新目标框（这里需要根据实际情况调整，旋转后的边界框可能需要重新计算）
        return rgb, tir, target


class RandomResize(nn.Module):
    """随机缩放"""

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), prob: float = 0.5):
        super().__init__()
        self.scale_range = scale_range
        self.prob = prob

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, target: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if torch.rand(1) < self.prob:
            scale = float(torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]))
            new_size = (int(rgb.size(1) * scale), int(rgb.size(2) * scale))
            rgb = F.resize(rgb, new_size)
            tir = F.resize(tir, new_size)
            # 更新目标框（根据缩放比例调整位置）
            target["boxes"][:, :4] *= scale
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
            angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees))
            translate_x = float(torch.empty(1).uniform_(-self.translate[0], self.translate[0]))
            translate_y = float(torch.empty(1).uniform_(-self.translate[1], self.translate[1]))
            scale = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]))
            shear = float(torch.empty(1).uniform_(-self.shear, self.shear))
            rgb = F.affine(rgb, angle, (translate_x, translate_y), scale, shear)
            tir = F.affine(tir, angle, (translate_x, translate_y), scale, shear)
            # 更新目标框（根据仿射变换矩阵调整位置）
        return rgb, tir, target


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
        "random_flip": 0.5,
        "random_rotation": 10.0,
        "random_resize": 0.2,
        "color_jitter": 0.2,
        "gamma_correction": 0.5,
        "random_erase": 0.1,
        "blur_sigma": 0.5
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
