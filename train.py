# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: train.py
@Time    : 2025/5/19 下午2:31
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
python train.py -c ./configs/config.yaml
python train.py -c ./configs/config_swin.yaml
"""
from engines import training
from utils import get_args_config
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    cfg = get_args_config()
    training(cfg)


if __name__ == "__main__":
    main()