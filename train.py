# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: train.py
@Time    : 2025/5/19 下午2:31
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   :
"""
from engines import training
from utils import load_config
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    cfg = load_config('./configs/config.yaml')
    training(cfg)


if __name__ == "__main__":
    main()