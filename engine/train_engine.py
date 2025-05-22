# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: train_engine.py
@Time    : 2025/5/21 下午4:01
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 训练引擎
@Usage   : pycocotools
"""
import pprint
import random
import torch
import numpy as np
import pandas as pd
import os
import time
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from optim.optim import build_lr_scheduler, build_optimizer
from utils import setup_logging, get_environment_info, get_rank
from dataloader import build_dataset
from models import build_model
from utils import get_output_dir
import warnings
from models.dfine.box_ops import box_iou
warnings.filterwarnings('ignore')


class TrainingEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = get_output_dir(cfg.output_dir, cfg.name)
        self.logger = setup_logging(cfg, self.output_dir)
        self.logger.info('Train Log %s' % time.strftime("%c"))
        self.env_info = get_environment_info()
        self.logger.info(self.env_info)
        self.logger.info('Running with config:')
        self.logger.info(pprint.pformat(cfg.__dict__))
        self.device = cfg.device

        # Fix random seed for reproducibility
        seed = cfg.seed + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Initialize model and criterion
        self.model, self.criterion = build_model(cfg, training=True)
        self.model.to(self.device)
        if isinstance(self.criterion, tuple) and len(self.criterion) == 2:
            for loss in self.criterion:
                loss.to(self.device)
        else:
            self.criterion.to(self.device)

        self.model_without_ddp = self.model
        self.n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info('number of params: %d', self.n_parameters)

        # Configure optimizer using the new function
        self.optimizer = build_optimizer(self.model_without_ddp, cfg)

        # Configure learning rate scheduler using the new function
        self.lr_scheduler = build_lr_scheduler(self.optimizer, cfg)

        # Log optimizer information
        optimizer_info = f"optimizer: Adam(lr={cfg.training.lr})"
        optimizer_info += " with parameter groups "
        for i, param_group in enumerate(self.optimizer.param_groups):
            optimizer_info += f"{len(param_group['params'])} weight(decay={param_group['weight_decay']}), "
        optimizer_info = optimizer_info.rstrip(', ')
        self.logger.info(optimizer_info)

        # Setup data loaders
        self.dataloader_train, self.dataloader_val = build_dataset(cfg=cfg)

        # Resume training if specified
        if cfg.resume:
            self.logger.info('------------------------ Continue training ------------------------')
            checkpoint = torch.load(cfg.resume, map_location='cpu')
            self.model_without_ddp.load_state_dict(checkpoint['model'])
            if not cfg.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.start_epoch = checkpoint['epoch'] + 1
        else:
            self.start_epoch = cfg.training.start_epoch

        # Setup tensorboard and csv file for results
        self.tensorboard_dir = os.path.join(str(self.output_dir), 'tensorboard')
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tensorboard_dir)

        self.csv_file_path = os.path.join(str(self.output_dir), 'result.csv')
        self.results_df = pd.DataFrame(columns=[
            'epoch', 'train_loss',
            'val_loss', 'val_ap', 'val_iou_50', 'val_iou_50_95'
        ])

        self.checkpoint_dir = os.path.join(str(self.output_dir), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Store best metrics
        self.best_ap = -1
        self.best_iou_50 = -1
        self.best_iou_50_95 = -1

    def _save_model(self, epoch, step, is_best=False):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save({
            'epoch': epoch,
            'step': step,
            'model': self.model_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'loss': None,  # Placeholder
        }, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save({
                'epoch': epoch,
                'step': step,
                'model': self.model_without_ddp.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'loss': None,  # Placeholder
            }, best_path)

    def run(self):
        self.logger.info("------------------------ Start training ------------------------")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            t1 = time.time()
            stat = self._train(epoch)
            t2 = time.time()

            # Log training loss
            self.logger.info("[ep %d][lr %.7f][%.2fs] loss: %.4f", epoch, self.optimizer.param_groups[0]['lr'], t2 - t1,
                            stat['loss'])
            self.writer.add_scalar('loss/loss', stat['loss'], epoch)

            # Update training results
            self.results_df.loc[epoch] = {
                'epoch': epoch,
                'train_loss': stat['loss'],
                'val_loss': None,
                'val_ap': None,
                'val_iou_50': None,
                'val_iou_50_95': None
            }

            # Adjust learning rate
            if self.cfg.training.scheduler == 'step':
                self.lr_scheduler.step()

            # Perform evaluation
            if epoch % self.cfg.training.eval_freq == 0 and epoch >= self.cfg.training.start_eval:
                t_eval_start = time.time()
                metrics = self._evaluate(epoch)
                t_eval_end = time.time()

                if self.cfg.training.scheduler == 'plateau':
                    self.lr_scheduler.step(metrics['loss'])

                # Log evaluation results
                self.logger.info(
                    "[ep %d][%.3fs][%.5ffps] loss: %.4f, AP: %.4f, IOU@50: %.4f, IOU@50-95: %.4f ---- @best AP: %.4f, @best IOU@50: %.4f, @best IOU@50-95: %.4f" % \
                    (epoch, t_eval_end - t_eval_start, len(self.dataloader_val.dataset) / (t_eval_end - t_eval_start),
                     metrics['loss'], metrics['ap'], metrics['iou_50'], metrics['iou_50_95'],
                     self.best_ap, self.best_iou_50, self.best_iou_50_95)
                )

                # Log to tensorboard
                self.writer.add_scalar('metric/val_loss', metrics['loss'], epoch)
                self.writer.add_scalar('metric/AP', metrics['ap'], epoch)
                self.writer.add_scalar('metric/IOU_50', metrics['iou_50'], epoch)
                self.writer.add_scalar('metric/IOU_50_95', metrics['iou_50_95'], epoch)

                # Update validation results
                self.results_df.loc[epoch, ['val_loss', 'val_ap', 'val_iou_50', 'val_iou_50_95']] = [
                    metrics['loss'], metrics['ap'], metrics['iou_50'], metrics['iou_50_95']
                ]

                # Save best model
                if metrics['ap'] > self.best_ap:
                    self.best_ap = metrics['ap']
                    self._save_model(epoch, 0, is_best=True)
                if metrics['iou_50'] > self.best_iou_50:
                    self.best_iou_50 = metrics['iou_50']
                    self._save_model(epoch, 0, is_best=True)
                if metrics['iou_50_95'] > self.best_iou_50_95:
                    self.best_iou_50_95 = metrics['iou_50_95']
                    self._save_model(epoch, 0, is_best=True)

            # Save latest model and results
            self._save_model(epoch, 0)
            self.results_df.to_csv(self.csv_file_path, index=False)

        # Training summary
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('Summary of results')
        self.logger.info(self.env_info)
        self.logger.info('Training time {}'.format(total_time_str))
        self.logger.info("Best AP: %.4f, Best IOU@50: %.4f, Best IOU@50-95: %.4f" % (
            self.best_ap, self.best_iou_50, self.best_iou_50_95))
        self.logger.info('Results saved to {}'.format(self.cfg.output_dir))

    def _train(self, epoch):
        self.model.train()
        self.criterion.train()
        total_loss = 0.0
        total_samples = 0

        with tqdm(self.dataloader_train, desc=f'Epoch {epoch} [Training]') as pbar:
            for batch_idx, (rgb_images, tir_images, targets) in enumerate(pbar):
                rgb_images = rgb_images.to(self.device)
                tir_images = tir_images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                           for target in targets]
                outputs = self.model(rgb_images, tir_images, targets)
                # print(f"pred_logits: {outputs['pred_logits']}")
                # print(f"pred_boxes: {outputs['pred_boxes']}")
                # Compute loss using D-FINE criterion
                loss_dict = self.criterion(outputs, targets)
                loss = sum(loss_dict.values())
                loss_dict["total_loss"] = loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * rgb_images.size(0)
                total_samples += rgb_images.size(0)

                pbar.set_postfix({
                    'loss': total_loss / total_samples
                })

        epoch_loss = total_loss / total_samples
        return {'loss': epoch_loss}

    def _evaluate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_aps = []
        all_iou_50 = []
        all_iou_50_95 = []

        with torch.no_grad(), tqdm(self.dataloader_val, desc=f'Epoch {epoch} [Validation]') as pbar:
            for batch_idx, (rgb_images, tir_images, targets) in enumerate(pbar):
                rgb_images = rgb_images.to(self.device)
                tir_images = tir_images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                           for target in targets]
                outputs = self.model(rgb_images, tir_images, targets)

                # Compute loss using D-FINE criterion
                loss_dict = self.criterion(outputs, targets)
                loss = sum(loss_dict.values())
                total_loss += loss.item() * rgb_images.size(0)
                total_samples += rgb_images.size(0)

                # Compute detection metrics (AP and IOU)
                ap, iou_50, iou_50_95 = compute_detection_metrics(outputs, targets)
                all_aps.append(ap)
                all_iou_50.append(iou_50)
                all_iou_50_95.append(iou_50_95)

                pbar.set_postfix({
                    'loss': total_loss / total_samples
                })

        val_loss = total_loss / total_samples

        # Aggregate detection metrics
        avg_ap = np.mean(all_aps) if all_aps else 0.0
        avg_iou_50 = np.mean(all_iou_50) if all_iou_50 else 0.0
        avg_iou_50_95 = np.mean(all_iou_50_95) if all_iou_50_95 else 0.0

        metrics = {
            'loss': val_loss,
            'ap': avg_ap,
            'iou_50': avg_iou_50,
            'iou_50_95': avg_iou_50_95
        }

        return metrics


def compute_detection_metrics(outputs, targets, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)

    pred_logits = outputs['pred_logits']  # [B, Q, C]
    pred_boxes = outputs['pred_boxes']    # [B, Q, 4]

    batch_size = pred_logits.shape[0]
    num_classes = pred_logits.shape[-1]

    all_matches = [[] for _ in range(len(iou_thresholds))]
    all_scores = []

    for i in range(batch_size):
        logits = pred_logits[i]       # [Q, C]
        boxes = pred_boxes[i]         # [Q, 4]
        target = targets[i]           # {'labels': [...], 'boxes': [...]}
        t_boxes = target['boxes']     # [K, 4]
        t_labels = target['labels']   # [K]

        # softmax 分类置信度
        scores = torch.sigmoid(logits).squeeze(-1)  # shape: [Q]
        labels = (scores > 0.5).long()  # 根据阈值得到类别标签（0 或 1）
        max_scores = scores  # 置信度就是 sigmoid 输出值

        # 获取预测框和真实框
        pred_boxes_valid = boxes
        pred_scores_valid = max_scores

        # 对每一类分别处理
        for cls in range(num_classes):  # 遍历所有类别
            # 获取该类别的预测和真实框
            cls_pred_mask = (labels == cls)
            cls_preds = pred_boxes_valid[cls_pred_mask]
            cls_scores = pred_scores_valid[cls_pred_mask]
            cls_targets = t_boxes[t_labels == cls]

            if len(cls_preds) == 0 or len(cls_targets) == 0:
                continue

            # 计算 IoU 矩阵
            ious, _ = box_iou(cls_preds, cls_targets)  # [P, T]

            # 存储每个预测对应的匹配情况
            for ti, thresh in enumerate(iou_thresholds):
                matched_gt = set()
                for p_idx in range(len(cls_preds)):
                    match = (ious[p_idx] >= thresh).nonzero(as_tuple=True)
                    if len(match[0]) == 0:
                        all_matches[ti].append(0)
                        all_scores.append(cls_scores[p_idx].item())
                        continue
                    match_indices = match[0]
                    match_indices = match_indices[torch.tensor([t_idx not in matched_gt for t_idx in match_indices], device=match_indices.device)]
                    if len(match_indices) == 0:
                        all_matches[ti].append(0)
                    else:
                        matched_gt.add(match_indices[0].item())
                        all_matches[ti].append(1)
                    all_scores.append(cls_scores[p_idx].item())

    # 如果没有匹配项，则返回 0
    if not all_scores:
        return 0.0, 0.0, 0.0

    # 将所有得分排序并计算 PR 曲线下的面积 (AP)
    scores_sorted = sorted(zip(all_scores, sum(all_matches, [])), key=lambda x: -x[0])
    scores_sorted = list(zip(*scores_sorted))
    tp = np.array(scores_sorted[1])
    fp = 1 - tp
    tp_cumu = np.cumsum(tp)
    fp_cumu = np.cumsum(fp)
    recalls = tp_cumu / (tp_cumu[-1] + 1e-6)
    precisions = tp_cumu / (tp_cumu + fp_cumu + 1e-6)
    ap = np.trapz(precisions, recalls)

    # IoU@0.5
    idx_50 = iou_thresholds.tolist().index(0.5)
    tp_50 = np.array(all_matches[idx_50])
    iou_50 = tp_50.mean() if len(tp_50) else 0.0

    # IoU@0.5:0.95
    iou_50_95 = np.mean([np.array(m).mean() for m in all_matches]) if any(all_matches) else 0.0

    return ap, iou_50, iou_50_95


def train(cfg):
    engine = TrainingEngine(cfg)
    engine.run()