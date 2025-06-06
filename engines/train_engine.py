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
from tensorboardX import SummaryWriter

from models.dfine import DFINEPostProcessor
from optim.optim import build_lr_scheduler, build_optimizer
from utils import setup_logging, get_environment_info, get_rank
from dataloader import build_dataset
from models import build_model
from utils import get_output_dir
import warnings
from engines import train, evaluate

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
        self.model, self.criterions = build_model(cfg, training=True)
        self.model.to(self.device)
        if isinstance(self.criterions, tuple) and len(self.criterions) == 2:
            for loss in self.criterions:
                loss.to(self.device)
        else:
            self.criterions.to(self.device)

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

        # **关键部分：配置PostProcessor**
        self.postprocessor = DFINEPostProcessor(
            num_classes=1,  # 类别数
            use_focal_loss=True,  # 是否使用Focal Loss
            num_top_queries=300,
            # remap_mscoco_category=True
        )
        self.postprocessor.to(self.device)

        # Setup data loaders
        self.train_dataloader, self.val_dataloader = build_dataset(cfg=cfg)

        # Resume training if specified
        if cfg.resume:
            self.logger.info('------------------------ Continue training ------------------------')
            checkpoint = torch.load(cfg.resume, map_location='cpu')
            self.model_without_ddp.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
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
            'val_loss', 'val_iou_50', 'val_iou_50_95'
        ])

        self.checkpoint_dir = os.path.join(str(self.output_dir), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Store best metrics
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
            stat = train(self.model, self.criterions, self.train_dataloader, self.optimizer, self.device,
                         epoch)
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
                metrics = evaluate(self.model, self.criterions, self.postprocessor, self.val_dataloader, self.device, epoch)
                t_eval_end = time.time()

                if self.cfg.training.scheduler == 'plateau':
                    self.lr_scheduler.step(metrics['loss'])

                # Log evaluation results
                self.logger.info(
                    "[ep %d][%.3fs][%.5ffps] loss: %.4f, IOU@50: %.4f, IOU@50-95: %.4f ---- @best IOU@50: %.4f, @best IOU@50-95: %.4f" % \
                    (epoch, t_eval_end - t_eval_start, len(self.val_dataloader.dataset) / (t_eval_end - t_eval_start),
                     metrics['loss'], metrics['iou_50'], metrics['iou_50_95'],
                     self.best_iou_50, self.best_iou_50_95)
                )

                # Log to tensorboard
                self.writer.add_scalar('metric/val_loss', metrics['loss'], epoch)
                self.writer.add_scalar('metric/IOU_50', metrics['iou_50'], epoch)
                self.writer.add_scalar('metric/IOU_50_95', metrics['iou_50_95'], epoch)

                # Update validation results
                self.results_df.loc[epoch, ['val_loss', 'val_iou_50', 'val_iou_50_95']] = [
                    metrics['loss'], metrics['iou_50'], metrics['iou_50_95']
                ]

                # Save best model
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
        self.logger.info("Best IOU@50: %.4f, Best IOU@50-95: %.4f" % (
            self.best_iou_50, self.best_iou_50_95))
        self.logger.info('Results saved to {}'.format(self.cfg.output_dir))


def training(cfg):
    engine = TrainingEngine(cfg)
    engine.run()
