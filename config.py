class DefaultConfigs(object):
    # 1.字符串参数
    train_data: str = "./data/train/"
    test_data: str = "./data/test/images/"
    val_data: str = "no"
    model_name: str = "efficientnetv2_s"  # 更新为性能更好的模型
    weights: str = "./checkpoints/"
    best_models: str = weights + "best_model/"
    submit: str = "./submit/"
    logs: str = "./logs/"

    # 数据增强路径配置
    aug_source_path: str = "./data/train/"  # 源数据路径
    aug_target_path: str = "./aug/train/"   # 增强后保存路径

    # 设备配置
    device: str = 'auto'  # 选项: "auto", "cuda", "cpu"
    gpus: str = '0'  # 指定要使用的GPU设备, 例如 "0" 或 "0,1"

    # 2.训练参数
    epoch: int = 40
    train_batch_size: int = 24  # 增大批量大小以提高训练速度
    val_batch_size: int = 32
    test_batch_size: int = 32
    num_workers: str = 'auto'
    img_height: int = 380  
    img_weight: int = 380 
    num_classes: int = 59
    seed: int = 888
    lr: float = 1e-4  # 提高学习率，配合学习率调度器
    lr_decay: float = 1e-4
    weight_decay: float = 1e-4
    
    # 优化器配置
    optimizer: str = 'adamw'  # 选项: 'adam', 'adamw', 'sgd', 'ranger'
    use_lookahead: bool = True  # 是否使用Lookahead优化器包装
    
    # 学习率调度配置
    scheduler: str = 'cosine'  # 选项: 'step', 'cosine', 'onecycle'
    warmup_epochs: int = 3  # 预热epochs
    warmup_factor: float = 0.1  # 预热因子
    
    # 数据增强参数
    use_mixup: bool = True  # 使用Mixup数据增强
    mixup_alpha: float = 0.2
    cutmix_prob: float = 0.5  # 使用CutMix的概率
    use_random_erasing: bool = True  # 使用随机擦除增强
    
    # 数据增强配置
    use_data_aug: bool = True  # 启用数据增强
    aug_noise: bool = True      # 是否添加噪声
    aug_brightness: bool = True  # 是否调整亮度
    aug_flip: bool = True       # 是否进行翻转
    aug_contrast: bool = True   # 是否调整对比度
    remove_error_images: bool = True  # 是否删除错误图片
    aug_num_workers: int = 8  # 数据增强多线程数

    # 数据增强处理参数
    aug_max_workers: int = 8  # 数据增强多线程数
    aug_noise_var: float = 0.01  # 高斯噪声方差
    aug_brightness_range: tuple = (0.5, 1.5)  # 亮度调整范围
    aug_contrast_factor: float = 1.5  # 对比度增强因子
    
    # 训练策略参数
    use_amp: bool = True  # 使用混合精度训练
    use_early_stopping: bool = True
    early_stopping_patience: int = 7  # 增加早停耐心，避免过早停止
    use_focal_loss: bool = True  # 使用Focal Loss处理类别不平衡
    
    # 正则化参数
    label_smoothing: float = 0.1  # 标签平滑系数
    use_gradient_checkpointing: bool = True  # 梯度检查点，减少内存使用
    gradient_clip_val: float = 1.0  # 梯度裁剪值
    
    # 高级功能
    use_ema: bool = True  # 使用指数移动平均
    ema_decay: float = 0.995  # EMA衰减率
    progressive_resizing: bool = False  # 渐进式缩放训练（从小图像开始，逐渐增大）
    progressive_sizes: list = [224, 320, 380]  # 渐进式缩放的图像尺寸

    def __init__(self):
        if self.device in ['auto', 'cuda', 'cpu']:
            self.device = self.device
        else:
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'")
        
        if self.num_workers == 'auto':
            import psutil
            self.num_workers = psutil.cpu_count(logical=False)
        elif self.num_workers > 0:
            self.num_workers = self.num_workers
        else:
            raise ValueError("num_workers must be 'auto' or a positive integer")
        
        if self.aug_num_workers == 'auto':
            import psutil
            self.aug_num_workers = psutil.cpu_count(logical=False)
        elif self.aug_num_workers > 0:
            self.aug_num_workers = self.aug_num_workers
        else:
            raise ValueError("aug_num_workers must be 'auto' or a positive integer")
        
config = DefaultConfigs()
