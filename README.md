# 植物病害检测系统

## 项目概述

本项目是一个基于深度学习的植物疾病检测系统，能够识别多达59种不同的植物病害。系统利用最新的深度学习技术（EfficientNet-B4）进行训练，可以用于农业病害的快速识别和诊断。

## 功能特点

- 支持59种不同植物疾病的分类
- 使用先进的EfficientNet-B4深度学习模型
- 实现了混合精度训练，提高训练速度
- 包含丰富的数据增强方法，提高模型鲁棒性
- 支持早停策略，防止过拟合
- 提供详细的训练日志和性能指标

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA (推荐用于GPU加速)
- 主要依赖：
  - numpy
  - pandas
  - scikit-learn
  - PIL
  - opencv-python
  - tqdm
  - skimage

## 安装说明

1. 克隆代码库

```bash
git clone https://github.com/yourusername/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集准备


## 使用说明

### 1. 数据准备

将原始数据整理到项目的数据目录结构中：

```bash
python move.py
```

### 2. 数据增强（可选）

通过数据增强扩充训练集：

```bash
python data_aug.py
```

### 3. 训练模型

使用以下命令开始训练：

```bash
python train.py
```

您可以在`config.py`中调整训练参数，包括：

- 学习率
- 批量大小
- 训练轮数
- 图像尺寸
- 早停策略
- 模型选择（EfficientNet/ConvNeXt）

### 4. 模型预测

模型训练完成后，会自动对测试集进行预测，并将结果保存在`./submit/baseline.json`文件中。

## 深度学习模型

本项目使用了两种先进的深度学习模型：

1. **EfficientNet-B4** (默认)：Google开发的高效网络，平衡了准确度和计算效率。
2. **ConvNeXt**：更现代的卷积神经网络架构，可与Transformer相媲美的性能。

您可以在`models/model.py`的`get_net()`函数中选择使用哪个模型。

## 性能优化

- **混合精度训练**：通过使用FP16和FP32混合精度，加速训练过程
- **多线程数据加载**：提高数据加载效率
- **参数冻结**：仅微调模型的高层特征，减少过拟合风险
- **早停策略**：当验证损失连续多轮未改善时自动停止训练

## 故障排除

1. **显存不足**：
   - 减小批量大小
   - 降低图像分辨率
   - 使用较小的模型
   
2. **训练速度慢**：
   - 确保启用了混合精度训练
   - 增加数据加载器的`num_workers`
   - 确保使用GPU训练

3. **无法读取某些图像**：
   - 检查图像格式是否支持
   - 确保图像路径中没有中文字符
   - 查看日志文件了解详细错误信息

## 许可证

1. 将训练数据放在 `./data/train/` 目录下，每个类别有一个子目录
2. 将测试数据放在 `./data/test/images/` 目录下

可以使用 `move.py` 脚本来帮助组织数据集:
```bash
python move.py
```

如需进行数据增强:
```bash
python data_aug.py
```

## 训练模型

使用以下命令开始训练:
```bash
python train.py
```

可以在 `config.py` 中调整训练参数，包括:
- 学习率
- 批次大小
- 训练轮数
- 图像尺寸
- 设备选择 (CPU/GPU)

## 预测

训练完成后，模型会自动对测试集进行预测，并将结果保存在 `./submit/baseline.json` 文件中。

## 项目结构

```bash
.
├── config.py # 配置文件
├── train.py # 训练脚本
├── data_aug.py # 数据增强脚本
├── move.py # 数据移动脚本
├── utils.py # 工具函数
├── dataset/ # 数据集相关代码
│ ├── init.py
│ └── dataloader.py # 数据加载器
├── models/ # 模型定义
│ ├── init.py
│ └── model.py # 模型架构
├── checkpoints/ # 模型检查点
│ └── best_model/ # 最佳模型存储
├── logs/ # 训练日志
└── submit/ # 预测结果
```

### 5. 注意事项

1. 数据预处理：
   - 由于现在不支持中文的一些符号。数据增强时，保证数据集里不要有中文命名的文件，否则将跳过它。

2. GPU 训练：
   - 确保已安装 CUDA 和对应版本的 PyTorch
   - 可在 config.py 中设置 GPU 相关参数

3. CPU 训练：
   - 无需特殊配置，程序会自动使用 CPU
   - 训练速度会较 GPU 慢

4. 内存使用：
   - 根据实际内存大小调整 batch_size
   - CPU 训练时建议适当减小 batch_size
   
## 性能指标

- 当前最佳模型Top-2准确率: 88.67%

### 6. 联系方式

如果有任何问题或建议，欢迎联系：
- Email: 961521953@qq.com

### 7. 许可
[GNU General Public License v3.0](LICENSE)

## 联系方式

1. 将训练数据放在 `./data/train/` 目录下，每个类别有一个子目录
2. 将测试数据放在 `./data/test/images/` 目录下

可以使用 `move.py` 脚本来帮助组织数据集:
```bash
python move.py
```

如需进行数据增强:
```bash
python data_aug.py
```

## 训练模型

使用以下命令开始训练:
```bash
python train.py
```

可以在 `config.py` 中调整训练参数，包括:
- 学习率
- 批次大小
- 训练轮数
- 图像尺寸
- 设备选择 (CPU/GPU)

## 预测

训练完成后，模型会自动对测试集进行预测，并将结果保存在 `./submit/baseline.json` 文件中。

## 项目结构

```bash
.
├── config.py # 配置文件
├── train.py # 训练脚本
├── data_aug.py # 数据增强脚本
├── move.py # 数据移动脚本
├── utils.py # 工具函数
├── dataset/ # 数据集相关代码
│ ├── init.py
│ └── dataloader.py # 数据加载器
├── models/ # 模型定义
│ ├── init.py
│ └── model.py # 模型架构
├── checkpoints/ # 模型检查点
│ └── best_model/ # 最佳模型存储
├── logs/ # 训练日志
└── submit/ # 预测结果
```

### 5. 注意事项

1. 数据预处理：
   - 由于现在不支持中文的一些符号。数据增强时，保证数据集里不要有中文命名的文件，否则将跳过它。

2. GPU 训练：
   - 确保已安装 CUDA 和对应版本的 PyTorch
   - 可在 config.py 中设置 GPU 相关参数

3. CPU 训练：
   - 无需特殊配置，程序会自动使用 CPU
   - 训练速度会较 GPU 慢

4. 内存使用：
   - 根据实际内存大小调整 batch_size
   - CPU 训练时建议适当减小 batch_size
   
## 性能指标

- 当前最佳模型Top-2准确率: 88.67%

### 6. 联系方式

如果有任何问题或建议，欢迎联系：
- Email: 961521953@qq.com

### 7. 许可
[GNU General Public License v3.0](LICENSE)

## Author 作者

Created by Chen Runsen 
