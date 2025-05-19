# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: engine.py
@Time    : 2025/5/19 下午2:14
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
"""
import os
import time
import logging
from collections import defaultdict
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # 引入 tqdm
from models import DFINE, build_model


class TrainingEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        # 构建模型和损失函数
        self.model, self.criterion = build_model(cfg, training=True)
        self.device = cfg.device
        self.model_dir = os.path.join(cfg.output_dir, "checkpoints")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.training.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma
        )
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))

    def train(self, data_loader_train, data_loader_val):
        best_val_loss = float("inf")
        self.model.train()
        for epoch in range(self.cfg.training.epochs):
            logging.info(f"Epoch {epoch + 1}/{self.cfg.training.epochs}")
            total_losses = defaultdict(float)
            start_time = time.time()

            # 确保数据加载器的长度正确
            logging.info(f"Training DataLoader length: {len(data_loader_train)}")

            # 使用 tqdm 包裹训练数据加载器，添加进度条
            with tqdm(data_loader_train, desc=f'Epoch {epoch + 1} [Training]', total=len(data_loader_train), unit='batch') as pbar:
                for batch_idx, (rgb_images, tir_images, targets) in enumerate(pbar):
                    rgb_images = rgb_images.to(self.device)
                    tir_images = tir_images.to(self.device)
                    targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                               for target in targets]

                    # 前向传播
                    outputs = self.model(rgb_images, tir_images, targets)

                    # 计算损失
                    loss_dict = self.criterion(outputs, targets)

                    # 添加 total_loss 到 loss_dict
                    total_loss = sum(loss for loss in loss_dict.values())
                    loss_dict["total_loss"] = total_loss

                    # 反向传播
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    # 更新学习率
                    self.scheduler.step()

                    # 日志记录
                    for k, v in loss_dict.items():
                        total_losses[k] += v.item()
                    if batch_idx % self.cfg.training.log_interval == 0:
                        log_str = f"Batch {batch_idx + 1}/{len(data_loader_train)} - "
                        log_str += ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                        logging.info(log_str)
                        self.writer.add_scalars("Training Loss", {k: v.item() for k, v in loss_dict.items()},
                                                epoch * len(data_loader_train) + batch_idx)

                    # 更新进度条信息
                    pbar.set_postfix(loss=total_loss.item())

            avg_losses = {k: v / len(data_loader_train) for k, v in total_losses.items()}
            log_str = f"Epoch {epoch + 1} completed. Time: {time.time() - start_time:.2f}s | "
            log_str += ", ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
            logging.info(log_str)

            # 验证
            val_loss = self.validate(data_loader_val)
            self.writer.add_scalar("Validation Total Loss", val_loss, epoch)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(os.path.join(self.model_dir, "best_model.pth"))

            # 修正：在每个 epoch 结束后调用 scheduler.step()
            self.scheduler.step()

    def validate(self, data_loader_val):
        self.model.eval()
        total_losses = defaultdict(float)

        # 确保数据加载器的长度正确
        logging.info(f"Validation DataLoader length: {len(data_loader_val)}")

        # 使用 tqdm 包裹验证数据加载器，添加进度条
        with tqdm(data_loader_val, desc='Validation', total=len(data_loader_val), unit='batch') as pbar:
            with torch.no_grad():
                for batch_idx, (rgb_images, tir_images, targets) in enumerate(pbar):
                    rgb_images = rgb_images.to(self.device)
                    tir_images = tir_images.to(self.device)
                    targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in target.items()} for
                               target in targets]

                    outputs = self.model(rgb_images, tir_images, targets)
                    loss_dict = self.criterion(outputs, targets)

                    # 添加 total_loss 到 loss_dict
                    total_loss = sum(loss for loss in loss_dict.values())
                    loss_dict["total_loss"] = total_loss

                    for k, v in loss_dict.items():
                        total_losses[k] += v.item()

                    # 更新进度条信息
                    pbar.set_postfix(loss=total_loss.item())

        avg_losses = {k: v / len(data_loader_val) for k, v in total_losses.items()}
        log_str = "Validation - " + ", ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
        logging.info(log_str)

        return avg_losses["total_loss"]

    def save_model(self, path):
        """
        保存模型和训练状态（模型权重 + optimizer + scheduler）
        Args:
            path: 保存路径，例如 "output/checkpoint.pth"
        """
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'cfg': self.cfg,
        }
        torch.save(state, path)
        logging.info(f"Model and training state saved to {path}")


def train(cfg):
    from dataloader import build_dataset
    data_loader_train, data_loader_val = build_dataset(cfg)
    engine = TrainingEngine(cfg)
    engine.train(data_loader_train, data_loader_val)