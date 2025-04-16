from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import os
from pathlib import Path

@dataclass
class PathConfig:
    """路径配置类，包含所有文件路径和目录"""
    
    # 基础目录
    base_dir: str = "./"
    data_dir: str = "./data/"
    models_dir: str = "./models/"
    
    # 数据目录
    train_dir: str = "./data/train/"
    test_dir: str = "./data/test/"
    test_images_dir: str = "./data/test/images/"
    val_dir: str = "./data/val/"
    temp_dir: str = "./data/temp/"
    temp_images_dir: str = "./data/temp/images/"
    temp_labels_dir: str = "./data/temp/labels/"
    
    # 合并数据集目录
    merged_train_dir: str = "./data/merged_train/"
    merged_test_dir: str = "./data/merged_test/"
    merged_val_dir: str = "./data/merged_val/"
    
    # 数据增强目录
    aug_dir: str = "./data/aug/"
    aug_train_dir: str = "./data/aug/train/"
    
    # 模型目录
    weights_dir: str = "./weights/"
    best_weights_dir: str = "./weights/best/"
    
    # 输出目录
    submit_dir: str = "./submit/"
    logs_dir: str = "./logs/"
    
    # 日志文件
    training_log: str = "./logs/training.log"
    data_aug_log: str = "./logs/data_augmentation.log"
    data_proc_log: str = "./logs/data_processing.log"
    inference_log: str = "./logs/inference.log"
    utils_log: str = "./logs/utils.log"
    
    # 数据文件
    train_annotations: str = "./data/temp/labels/AgriculturalDisease_train_annotations.json"
    val_annotations: str = "./data/temp/labels/AgriculturalDisease_validation_annotations.json"
    prediction_file: str = "./submit/prediction.json"
    
    def __post_init__(self):
        """确保所有路径都有一致的格式"""
        # 规范化路径格式
        for attr_name in self.__annotations__:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, str) and attr_name.endswith(('_dir', '_path')):
                # 确保目录路径以斜杠结尾
                if attr_name.endswith('_dir') and not attr_value.endswith('/'):
                    setattr(self, attr_name, attr_value + '/')
        
        # 创建关键目录
        os.makedirs(self.logs_dir, exist_ok=True)

def get_path_config():
    """创建PathConfig实例的工厂函数"""
    return PathConfig()

@dataclass
class DefaultConfigs:
    """配置类，包含所有训练和数据处理的参数设置"""
    
    # 创建路径配置 - 使用default_factory避免可变默认值的问题
    paths: PathConfig = field(default_factory=get_path_config)
    
    # 以下字段需要在__post_init__中初始化，因为它们依赖于paths
    train_data: str = field(default="")  # 训练数据路径
    test_data: str = field(default="")  # 测试数据路径
    val_data: str = "none"  # 验证数据路径
    model_name: str = "efficientnet_b4"  # 模型名称
    weights: str = field(default="")  # 权重保存路径
    best_weights: str = field(default="")  # 最佳模型保存路径
    submit: str = field(default="")  # 提交结果保存路径
    logs: str = field(default="")  # 日志保存路径
    
    # 数据集合并配置 Dataset Merging Configuration
    merge_datasets: bool = True  # 是否合并多个数据集
    dataset_to_use: str = "auto"  # 不合并时选择使用哪个数据集: "auto"(最大的), "first", "last", "specific"
    specific_dataset: str = ""  # 指定使用的数据集名称，当dataset_to_use="specific"时有效
    duplicate_test_to_common: bool = True  # 是否将测试集复制到通用测试目录
    merge_force: bool = False  # 是否强制重新合并已存在的数据集
    merge_on_startup: bool = True  # 是否在程序启动时自动合并数据集

    # 测试集处理配置 Test Dataset Configuration
    test_name_pattern: str = "ai_challenger_pdr2018_test*.zip"  # 测试集ZIP文件的匹配模式
    use_all_test_datasets: bool = True  # 是否使用所有测试集
    primary_test_dataset: str = "testa"  # 如果不使用所有测试集，指定使用哪个测试集

    # 数据增强路径配置 Data Augmentation Path Configuration
    aug_source_path: str = field(default="")  # 数据增强源数据路径
    aug_target_path: str = field(default="")  # 数据增强结果保存路径

    # 设备配置 Device Configuration
    device: str = 'auto'  # 训练设备选择: "auto", "cuda", "cpu"
    gpus: str = '0'  # GPU设备ID: "0" 或 "0,1"

    # 训练参数 Training Parameters
    epoch: int = 40  # 训练轮数
    train_batch_size: int = 24  # 训练批次大小
    val_batch_size: int = 32  # 验证批次大小
    test_batch_size: int = 32  # 测试批次大小
    num_workers: str = 'auto'  # 数据加载线程数
    img_height: int = 380  # 图像高度
    img_weight: int = 380  # 图像宽度
    num_classes: int = 59  # 类别数量
    seed: int = 888  # 随机种子
    lr: float = 1e-4  # 初始学习率
    lr_decay: float = 1e-4  # 学习率衰减
    weight_decay: float = 1e-4  # 权重衰减

    # 优化器配置 Optimizer Configuration
    optimizer: str = 'adamw'  # 优化器选择: 'adam', 'adamw', 'sgd', 'ranger'
    use_lookahead: bool = True  # 是否使用Lookahead优化器包装

    # 学习率调度配置 Learning Rate Scheduler Configuration
    scheduler: str = 'cosine'  # 学习率调度器: 'step', 'cosine', 'onecycle'
    warmup_epochs: int = 3  # 预热轮数
    warmup_factor: float = 0.1  # 预热因子

    # 数据增强参数 Data Augmentation Parameters
    use_mixup: bool = True  # 是否使用Mixup
    mixup_alpha: float = 0.2  # Mixup alpha参数
    cutmix_prob: float = 0.5  # CutMix概率
    use_random_erasing: bool = True  # 是否使用随机擦除

    # 数据增强配置 Data Augmentation Configuration
    use_data_aug: bool = True  # 是否启用数据增强
    use_mode: str = 'merge'  # 数据增强模式: 'merge', 'replace'
    aug_noise: bool = True  # 是否添加噪声
    aug_brightness: bool = True  # 是否调整亮度
    aug_flip: bool = True  # 是否进行翻转
    aug_contrast: bool = True  # 是否调整对比度
    remove_error_images: bool = True  # 是否删除错误图片
    aug_num_workers: int = 8  # 数据增强处理线程数

    # 数据增强处理参数 Data Augmentation Processing Parameters
    aug_max_workers: int = 8  # 数据增强最大线程数
    aug_noise_var: float = 0.01  # 高斯噪声方差
    aug_brightness_range: Tuple[float, float] = (0.5, 1.5)  # 亮度调整范围
    aug_contrast_factor: float = 1.5  # 对比度增强因子

    # 训练策略参数 Training Strategy Parameters
    use_amp: bool = True  # 是否使用混合精度训练
    use_early_stopping: bool = True  # 是否使用早停
    early_stopping_patience: int = 7  # 早停耐心值
    use_focal_loss: bool = True  # 是否使用Focal Loss

    # 正则化参数 Regularization Parameters
    label_smoothing: float = 0.1  # 标签平滑系数
    use_gradient_checkpointing: bool = True  # 是否使用梯度检查点
    gradient_clip_val: float = 1.0  # 梯度裁剪值

    # 高级功能 Advanced Features
    use_ema: bool = True  # 是否使用指数移动平均
    ema_decay: float = 0.995  # EMA衰减率
    progressive_resizing: bool = False  # 是否使用渐进式缩放
    progressive_sizes: List[int] = field(default_factory=lambda: [224, 320, 380])  # 渐进式缩放的图像尺寸

    def __post_init__(self):
        """初始化后的验证和设置"""
        # 初始化依赖于paths的字段
        self.train_data = self.paths.train_dir
        self.test_data = self.paths.test_images_dir
        self.weights = self.paths.weights_dir
        self.best_weights = self.paths.best_weights_dir
        self.submit = self.paths.submit_dir
        self.logs = self.paths.logs_dir
        self.aug_source_path = self.paths.train_dir
        self.aug_target_path = self.paths.aug_train_dir
        
        # 验证设备设置
        if self.device not in ['auto', 'cuda', 'cpu']:
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'")

        # 设置工作线程数
        if self.num_workers == 'auto':
            import psutil
            self.num_workers = psutil.cpu_count(logical=False)
        elif not isinstance(self.num_workers, int) or self.num_workers <= 0:
            raise ValueError("num_workers must be 'auto' or a positive integer")

        # 设置数据增强线程数
        if self.aug_num_workers == 'auto':
            import psutil
            self.aug_num_workers = psutil.cpu_count(logical=False)
        elif not isinstance(self.aug_num_workers, int) or self.aug_num_workers <= 0:
            raise ValueError("aug_num_workers must be 'auto' or a positive integer")

config = DefaultConfigs()
paths = config.paths
