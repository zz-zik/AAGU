# SegChange-R1

## 项目介绍

SegChange-R1 是一个基于深度学习的变化检测模型项目，主要用于分析和识别图像中的变化区域（如建筑物变化等）。该项目结合了视觉编码器与文本描述信息，通过 双时相视觉编码器 提取双时态图像的多尺度特征，利用特征差异模块进行特征差异建模，并引入多尺度特征融合模块融合多尺度特征。此外，它支持集成文本描述信息以增强检测能力，使用掩码预测头生成最终的变化掩码。项目还提供了完整的训练、测试流程及损失函数配置，适用于遥感图像、城市规划、环境监测等领域中的变化检测任务。

## 系统要求

- Python 3.12
- CUDA + PyTorch
- HuggingFace
- 稳定的网络连接
- 高质量代理IP（重要）

## 安装步骤

### 1. 创建虚拟环境

```bash
conda create -n segchange python=3.12 -y
conda activate segchange
```

### 2. 安装依赖包

```bash
pip install -r requirements.txt
```

### 3. 配置HuggingFace镜像

```bash
vim ~/.bashrc
 
export HF_ENDPOINT="https://hf-mirror.com"

source ~/.bashrc
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

### 生成词嵌入文件
如果需要只做固定类别的变化检测，例如房屋变化检测，可先生成固定的描述文本词嵌入文件，并保存在`weights/embeddings.pt`文件中，将嵌入文件地址填入 参数文件[configs](./configs/config.yaml)中`desc_embs`，这样的好处是，训练速度会更快，训练结果更准确。
```shell
python text_embs.py -c ./configs/config.yaml
```
如果需要做多类别变化检测，则需要将描述文本词嵌入文件保存在`weights/embeddings.pt`文件中，并修改参数文件[configs](./configs/config.yaml)中的`desc_embs`为`None`，直接训练即可。

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

本项目使用 [MIT许可证](LICENSE)


## 参考
https://blog.csdn.net/weixin_45679938/article/details/142030784
https://www.arxiv.org/pdf/2503.11070
https://www.arxiv.org/abs/2503.16825
https://zhuanlan.zhihu.com/p/627646794