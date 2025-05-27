# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: test.py
@Time    : 2025/5/24 下午6:40
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   : 
"""
from torch.utils.data import DataLoader
from dataloader import Crowds, Transforms
from engines import _test, load_model
import pprint
import time

from models.dfine import DFINEPostProcessor
from utils import get_args_config, get_output_dir, setup_logging, load_config, collate_fn_crowds


def main():
    # cfg = get_args_config()
    cfg = load_config('./configs/config.yaml')
    output_dir = get_output_dir(cfg.test.output_dir, cfg.test.name)
    logger = setup_logging(cfg, output_dir)
    logger.info('Test Log %s' % time.strftime("%c"))
    logger.info('Running with config:')
    logger.info(pprint.pformat(cfg.__dict__))
    device = cfg.test.device

    model = load_model(cfg, cfg.test.weights_dir, cfg.test.device)

    # Build postprocessor
    postprocessor = DFINEPostProcessor(
        num_classes=1,  # 类别数
        use_focal_loss=True,  # 是否使用Focal Loss
        num_top_queries=300,
        remap_mscoco_category=True
    )

    # Build test dataset
    transforms_val = Transforms(train=False, **cfg.data.transforms.to_dict())
    test_dataset = Crowds(transform=transforms_val, test=True, **cfg.data.to_dict())
    test_loader = DataLoader(test_dataset, batch_size=cfg.test.batch_size, collate_fn=collate_fn_crowds, shuffle=False)

    # Run evaluation
    logger.info("Start testing...")
    preds = _test(model, postprocessor, test_loader, device, cfg.test.threshold, output_dir=output_dir, show=cfg.test.show)


if __name__ == '__main__':
    main()
