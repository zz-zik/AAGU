# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: engine.py
@Time    : 2025/5/19 下午2:14
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 训练引擎
@Usage   :
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
"""
import os
import time
import logging
from collections import defaultdict
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models import DFINE, build_model
import pandas as pd

import numpy as np
import torch.nn.functional as F
from torchvision.ops import box_iou

from utils import setup_logging, get_output_dir


class TrainingEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.resume = cfg.resume
        self.output_dir = get_output_dir(cfg.output_dir, cfg.name)
        self.logger = setup_logging(cfg, self.output_dir)
        # 构建模型和损失函数
        self.model, self.criterion = build_model(cfg, training=True)
        self.device = cfg.device
        self.model_dir = os.path.join(cfg.output_dir, "checkpoints")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.to(self.device)

        # 对模型的不同部分使用不同的优化参数
        param_dicts = [
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad],
                "lr": cfg.training.lr
            },
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.training.lr_backbone,
            }
        ]
        self.optimizer = torch.optim.AdamW(param_dicts, lr=cfg.training.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma
        )

        # 用于恢复训练的状态变量
        self.start_epoch = cfg.training.start_epoch
        self.best_val_loss = float("inf")

        # 如果需要恢复训练，则加载之前的检查点
        if self.resume is not None:
            self._load_checkpoint()

        # 创建 tensorboard writer
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))

        # 初始化 CSV 文件路径
        self.csv_file_path = os.path.join(cfg.output_dir, 'result.csv')
        # 初始化结果 DataFrame
        self.results_df = pd.DataFrame(columns=[
            'epoch', 'train_loss', 'train_IoU50_95', 'train_AP50_95',
            'val_loss', 'val_mAP', 'val_IoU50', 'val_IoU75', 'val_IoU50_95'
        ])

    def _load_checkpoint(self):
        """加载最新的检查点以恢复训练"""
        if os.path.exists(self.resume):
            logging.info(f"正在恢复训练，从检查点 {self.resume} 加载")
            checkpoint = torch.load(self.resume, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float("inf"))
            logging.info(f"恢复训练成功，将从epoch {self.start_epoch}开始，最佳验证损失: {self.best_val_loss:.4f}")
        else:
            logging.warning(f"未找到检查点 {self.resume}，将从头开始训练")

    def train(self, data_loader_train, data_loader_val):
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info('number of params: %d', n_parameters)
        best_val_loss = self.best_val_loss
        self.model.train()
        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            epoch_start_time = time.time()
            total_losses = defaultdict(float)
            total_ious = []
            total_aps = []

            with tqdm(data_loader_train, desc=f'Epoch {epoch}/{self.cfg.training.epochs} [Training]',
                      total=len(data_loader_train), unit='batch') as pbar:
                for batch_idx, (rgb_images, tir_images, targets) in enumerate(pbar):
                    rgb_images = rgb_images.to(self.device)
                    tir_images = tir_images.to(self.device)
                    targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                               for target in targets]
                    outputs = self.model(rgb_images, tir_images, targets)
                    loss_dict = self.criterion(outputs, targets)
                    loss = sum(loss_dict.values())
                    loss_dict["total_loss"] = loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    for k, v in loss_dict.items():
                        total_losses[k] += v.item()

                    current_avg_losses = {k: v / (batch_idx + 1) for k, v in total_losses.items()}
                    pbar.set_postfix(avg_loss=current_avg_losses["total_loss"])

                    if self.cfg.training.log_interval > 0 and batch_idx % self.cfg.training.log_interval == 0:
                        elapsed_time = time.time() - epoch_start_time
                        current_lr = self.optimizer.param_groups[0]['lr']
                        log_str = f"[ep {epoch + 1}][lr {current_lr:.7f}][{elapsed_time:.2f}s] "
                        log_str += f"Batch {batch_idx + 1}/{len(data_loader_train)} - "
                        log_str += ", ".join([f"{k}: {v:.4f}" for k, v in current_avg_losses.items()])
                        logging.info(log_str)
                        self.writer.add_scalars("Training Loss", current_avg_losses,
                                                epoch * len(data_loader_train) + batch_idx)

                    # 计算当前批次的 IoU 和 AP
                    eval_metrics = self.evaluate_batch(outputs, targets)
                    total_ious.append(eval_metrics['iou'])
                    total_aps.append(eval_metrics['ap'])

            avg_losses = {k: v / len(data_loader_train) for k, v in total_losses.items()}
            epoch_elapsed_time = time.time() - epoch_start_time

            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 汇总一个 epoch 的评估结果
            avg_iou = np.mean(total_ious) if total_ious else 0
            avg_ap = np.mean(total_aps) if total_aps else 0

            # 更新指标到 DataFrame
            self.results_df.loc[epoch] = {'epoch': epoch, 'train_loss': avg_losses,
                                          'train_IoU50_95': avg_iou, 'train_AP50_95': avg_ap, 'val_loss': '',
                                          'val_mAP': '', 'val_IoU50': '', 'val_IoU75': '', 'val_IoU50_95': ''
                                          }

            # 日志打印
            log_str = f"[ep {epoch}][lr {current_lr:.7f}][{epoch_elapsed_time:.2f}s] Epoch {epoch} completed. "
            log_str += ", ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
            if 'train_IoU50_95' in self.results_df.columns and not pd.isna(
                    self.results_df.loc[epoch, 'train_IoU50_95']):
                log_str += f", train_IoU50_95: {self.results_df.loc[epoch, 'train_IoU50_95']:.4f}"
            if 'train_AP50_95' in self.results_df.columns and not pd.isna(self.results_df.loc[epoch, 'train_AP50_95']):
                log_str += f", train_AP50_95: {self.results_df.loc[epoch, 'train_AP50_95']:.4f}"
            logging.info(log_str)

            # 验证
            val_start_time = time.time()
            val_loss = self.validate(data_loader_val, epoch)
            val_elapsed_time = time.time() - val_start_time
            self.writer.add_scalar("Validation Total Loss", val_loss, epoch)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(os.path.join(self.model_dir, "best_model.pth"), is_best=True, val_loss=val_loss)
                logging.info(f"发现更好的模型，验证损失从 {self.best_val_loss:.4f} 改进至 {val_loss:.4f}")
                self.best_val_loss = val_loss

            # 保存最新模型（每轮结束都保存）
            self.save_model(
                os.path.join(self.model_dir, "latest.pth"),
                is_best=False,
                epoch=epoch,
                val_loss=val_loss
            )

            # 更新学习率调度器
            self.scheduler.step()

            # 每个 epoch 后保存一次 CSV
            self.results_df.to_csv(self.csv_file_path, index=False)
            # logging.info(f"Epoch {epoch + 1} results saved to {self.csv_file_path}")

    def validate(self, data_loader_val, epoch):
        self.model.eval()
        total_losses = defaultdict(float)
        total_aps = []
        total_ious_50 = []
        total_ious_75 = []
        total_ious_50_95 = []
        val_start_time = time.time()

        with tqdm(data_loader_val, desc='Validation', total=len(data_loader_val), unit='batch') as pbar:
            with torch.no_grad():
                for batch_idx, (rgb_images, tir_images, targets) in enumerate(pbar):
                    rgb_images = rgb_images.to(self.device)
                    tir_images = tir_images.to(self.device)
                    targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                               for target in targets]

                    outputs = self.model(rgb_images, tir_images, targets)
                    loss_dict = self.criterion(outputs, targets)
                    loss = sum(loss_dict.values())
                    loss_dict["total_loss"] = loss

                    for k, v in loss_dict.items():
                        total_losses[k] += v.item()

                    current_avg_losses = {k: v / (batch_idx + 1) for k, v in total_losses.items()}
                    pbar.set_postfix(avg_loss=current_avg_losses["total_loss"])

                    eval_metrics = self.evaluate_batch(outputs, targets)

                    iou_50_95 = eval_metrics['iou']
                    ap = eval_metrics['ap']
                    iou_50 = self.calculate_iou_thresholded(outputs, targets, threshold=0.5)
                    iou_75 = self.calculate_iou_thresholded(outputs, targets, threshold=0.75)

                    total_ious_50_95.append(iou_50_95)
                    total_aps.append(ap)
                    total_ious_50.append(iou_50)
                    total_ious_75.append(iou_75)

        # 计算验证集上的平均损失
        avg_losses = {k: v / len(data_loader_val) for k, v in total_losses.items()}

        # 获取当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']

        # 更新 DataFrame
        self.results_df.loc[epoch, ['val_loss', 'val_mAP', 'val_IoU50', 'val_IoU75', 'val_IoU50_95']] = [
            avg_losses["total_loss"], np.mean(total_aps), np.mean(total_ious_50), np.mean(total_ious_75), np.mean(total_ious_50_95)]

        # 日志打印
        log_str = f"[ep {epoch}][lr {current_lr:.7f}][{(time.time() - val_start_time):.2f}s] Validation - "
        log_str += ", ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
        if 'val_mAP' in self.results_df.columns and not pd.isna(self.results_df.loc[epoch, 'val_mAP']):
            log_str += f", mAP: {self.results_df.loc[epoch, 'val_mAP']:.4f}"
        if 'val_IoU50' in self.results_df.columns and not pd.isna(self.results_df.loc[epoch, 'val_IoU50']):
            log_str += f", IoU50: {self.results_df.loc[epoch, 'val_IoU50']:.4f}"
        if 'val_IoU75' in self.results_df.columns and not pd.isna(self.results_df.loc[epoch, 'val_IoU75']):
            log_str += f", IoU75: {self.results_df.loc[epoch, 'val_IoU75']:.4f}"
        if 'val_IoU50_95' in self.results_df.columns and not pd.isna(self.results_df.loc[epoch, 'val_IoU50_95']):
            log_str += f", IoU50-95: {self.results_df.loc[epoch, 'val_IoU50_95']:.4f}"
        logging.info(log_str)

        return avg_losses["total_loss"]

    def save_model(self, path, is_best=False, **kwargs):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'cfg': self.cfg,
            **kwargs
        }
        torch.save(state, path)
        # logging.info(f"Model saved to {path}")

    def calculate_iou(self, pred_boxes, true_boxes):
        return box_iou(pred_boxes, true_boxes)

    def compute_ap(self, recalls, precisions):
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        indices = np.where(mrec[1:] != mrec[:-1])[0] + 1
        ap = np.sum((mrec[indices] - mrec[indices - 1]) * mpre[indices])
        return ap

    def calculate_iou_thresholded(self, outputs, targets, threshold=0.5):
        pred_boxes = outputs['pred_boxes'].detach().cpu()
        true_boxes = torch.stack([target['boxes'].cpu() for target in targets]).detach().cpu()

        # 移除批次维度
        pred_boxes = pred_boxes.squeeze(0)
        true_boxes = true_boxes.squeeze(0)

        ious = self.calculate_iou(pred_boxes, true_boxes)
        diag_ious = torch.diag(ious)
        matched = diag_ious > threshold
        if len(diag_ious) == 0:
            return 0.0
        return diag_ious[matched].mean().item() if matched.any() else 0.0

    def evaluate_batch(self, outputs, targets):
        all_pred_boxes = outputs['pred_boxes'].detach().cpu()
        all_pred_logits = outputs['pred_logits'].detach().cpu()
        all_true_boxes = [target['boxes'].cpu() for target in targets]
        all_true_labels = [target['labels'].cpu() for target in targets]

        ious_all = []
        aps = []

        for i in range(len(targets)):
            pred_boxes = all_pred_boxes[i]
            pred_logits = all_pred_logits[i]
            true_boxes = all_true_boxes[i]
            true_labels = all_true_labels[i]

            if len(pred_boxes) == 0 or len(true_boxes) == 0:
                continue

            ious = self.calculate_iou(pred_boxes, true_boxes)
            ious_all.append(torch.diag(ious).mean().item())

            pred_scores, _ = torch.max(F.softmax(pred_logits, dim=1), dim=1)
            _, sorted_indices = torch.sort(pred_scores, descending=True)
            pred_boxes_sorted = pred_boxes[sorted_indices]

            tp = torch.zeros(pred_boxes_sorted.shape[0])
            fp = torch.zeros(pred_boxes_sorted.shape[0])

            matched_gt = torch.zeros(true_boxes.shape[0])

            for j in range(pred_boxes_sorted.shape[0]):
                ious_j = box_iou(pred_boxes_sorted[j].unsqueeze(0), true_boxes)
                max_iou, argmax_iou = ious_j.max(dim=1)
                if max_iou.item() > 0.5:
                    if matched_gt[argmax_iou.item()] == 0:
                        tp[j] = 1
                        matched_gt[argmax_iou.item()] = 1
                    else:
                        fp[j] = 1
                else:
                    fp[j] = 1

            tp_cumsum = torch.cumsum(tp, dim=0)
            fp_cumsum = torch.cumsum(fp, dim=0)
            recalls = tp_cumsum / (len(true_boxes) + 1e-6)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

            ap = self.compute_ap(recalls.numpy(), precisions.numpy())
            aps.append(ap)

        avg_iou = np.mean(ious_all) if ious_all else 0
        avg_ap = np.mean(aps) if aps else 0
        return {
            'iou': avg_iou,
            'ap': avg_ap
        }


def train(cfg):
    from dataloader import build_dataset
    data_loader_train, data_loader_val = build_dataset(cfg)
    engine = TrainingEngine(cfg)
    engine.train(data_loader_train, data_loader_val)
