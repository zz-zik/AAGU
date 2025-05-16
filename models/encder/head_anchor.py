# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: head_anchor.py
@Time    : 2025/5/16 下午4:32
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
"""
import torch
from torch import nn

__all__ = ['CenterHead']


class CenterHead(nn.Module):
    def __init__(self, in_channels, num_anchors=100):
        super().__init__()
        self.center_reg = nn.Conv2d(in_channels, 2, kernel_size=1)  # (dx, dy)
        self.center_conf = nn.Conv2d(in_channels, 1, kernel_size=1)  # confidence
        self.num_anchors = num_anchors

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            anchors_with_conf: Tensor (B, num_anchors, 3)，格式为 (x, y, conf)
        """
        B, C, H, W = x.shape

        # 回归坐标偏移和置信度
        reg_map = self.center_reg(x)  # (B, 2, H, W)
        conf_map = self.center_conf(x).sigmoid()  # (B, 1, H, W)

        # 展平
        reg_map = reg_map.permute(0, 2, 3, 1).reshape(B, -1, 2)  # (B, H*W, 2)
        conf_map = conf_map.view(B, -1)  # (B, H*W)

        # 取 top-k 最可能的目标中心
        scores = torch.norm(reg_map, dim=-1) * conf_map  # 结合偏移距离和置信度
        topk_indices = torch.topk(scores, k=self.num_anchors, dim=1).indices  # (B, K)

        batch_indices = torch.arange(B).unsqueeze(1).expand_as(topk_indices).to(x.device)

        selected_centers = reg_map[batch_indices, topk_indices]  # (B, K, 2)
        selected_confs = conf_map[batch_indices, topk_indices].unsqueeze(-1)  # (B, K, 1)

        # 转换为中心绝对坐标
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='xy')
        grid = torch.stack([grid_x, grid_y], dim=-1).float().to(x.device)  # (H, W, 2)
        grid = grid.view(-1, 2).expand(B, -1, -1)  # (B, H*W, 2)

        selected_grids = grid[batch_indices, topk_indices]
        centers_abs = selected_grids + selected_centers

        # 合并坐标和置信度
        anchors_with_conf = torch.cat([centers_abs, selected_confs], dim=-1)  # (B, K, 3)

        return anchors_with_conf


class YoloCenterHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.reg = nn.Conv2d(in_channels, 5, kernel_size=1)  # xc, yc, w, h, conf

    def forward(self, x):
        reg = self.reg(x)  # (B, 5, H, W)
        xywh = reg[:, :4].sigmoid()
        conf = reg[:, 4:].sigmoid()

        # 提取中心点
        centers = xywh[:, :2]  # (B, 2, H, W)

        # 合并坐标和置信度
        centers_with_conf = torch.cat([
            centers.permute(0, 2, 3, 1),
            conf.permute(0, 2, 3, 1)
        ], dim=-1)  # (B, H, W, 3)

        return centers_with_conf.view(x.shape[0], -1, 3)  # (B, H*W, 3)


if __name__ == '__main__':
    x = torch.randn(2, 256, 128, 128)
    cent_head = CenterHead(256, num_anchors=10)
    out = cent_head(x)
    print(out.shape)

    yolo_cent_head = YoloCenterHead(256)
    out = yolo_cent_head(x)
    print(out.shape)
