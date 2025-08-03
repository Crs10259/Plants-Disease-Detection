# Plants Disease Detection

植物病害检测系统，基于深度学习技术实现高效的病害分类。

## 功能特性

- 支持多种深度学习模型
- 数据增强和预处理
- 训练过程可视化
- 模型评估和测试

## 运行方式

### 标准版本
```bash
python main.py
```

### 改进版本 (推荐)
```bash
python improved_main.py
```

改进版本包含以下优化：
- 自适应学习率调度
- 类别不平衡处理
- 混合精度训练
- 梯度裁剪
- 更详细的训练统计

## 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
├── config/          # 配置文件
├── dataset/         # 数据集处理
├── libs/            # 训练和推理库
├── models/          # 模型定义
├── tools/           # 辅助工具
├── utils/           # 实用工具
├── weight/          # 模型权重
├── main.py          # 主程序入口
├── improved_main.py # 改进版主程序
└── README.md        # 项目说明
```

## 模型支持

- EfficientNet系列
- DenseNet169
- ConvNeXt
- Swin Transformer
- 混合模型

## 贡献指南

欢迎提交Pull Request改进项目。