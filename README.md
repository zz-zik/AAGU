# AAGU

## 项目介绍

## 系统要求

- Python 3.12
- CUDA + PyTorch
- HuggingFace
- 稳定的网络连接
- 高质量代理IP（重要）

## 安装步骤

### 1. 创建虚拟环境

```bash
conda create -n aagu python=3.12 -y
conda activate aagu
```

### 2. 安装依赖包

```bash
pip install -r requirements.txt
```

## 数据集介绍
支持两种数据集结构格式：
### 1. 默认结构：
数据集的结构如下：
```text
data/
  ├── train/               # 训练集
  │   ├── RGB/             # RGB可见光图像
  │   ├── TIR/             # TIR红外光图像
  │   ├── annotations/     # coco标签文件
  │   │    ├── train.json  # 训练集标签文件
  │   │    ├── val.json    # 验证集标签文件
  │   │    └── val.json    # 测试集标签文件
  │   └── labels/          # yolo标签文件
  ├── val/                 # 验证集
  │   ├── RGB/             # RGB可见光图像
  │   ├── TIR/             # TIR红外光图像
  │   │    ├── train.json  # 训练集标签文件
  │   │    ├── val.json    # 验证集标签文件
  │   │    └── val.json    # 测试集标签文件
  │   ├── annotations/     # coco标签文件
  │   └── label/           # yolo标签文件    
  └── test/                # 测试集
      ├── RGB/             # RGB可见光图像
      ├── TIR/             # TIR红外光图像
      ├── annotations/     # coco标签文件(可选)
      │    ├── train.json  # 训练集标签文件
      │    ├── val.json    # 验证集标签文件
      │    └── val.json    # 测试集标签文件
      └── labels/           # yolo标签文件(可选)
```
修改参数文件[configs](./configs/config.yaml)中的`data_format`为`default`。

### 2. 自定义结构：
数据集的结构如下：
```text
data/
  ├── RGB/                # RGB可见光图像
  │── TIR/                # TIR红外光图像
  ├── annotations/        # coco标签文件
  │── label/              # yolo标签文件
  └── list                # 列表文件
      ├── train.txt       # 训练集列表
      ├── val.txt         # 训练集列表
      └── test.txt        # 验证集列表
```
修改参数文件[configs](./configs/config.yaml)中的`data_format`为`custom`。

## 训练

### 命令行训练
```shell
python train.py -c ./configs/config.yaml
```

## 测试
```shell
python test.py -c ./configs/config.yaml
```

## 贡献

欢迎提交问题和代码改进。请确保遵循项目的代码风格和贡献指南。

## 许可证

本项目使用 [Apache License 2.0](LICENSE)