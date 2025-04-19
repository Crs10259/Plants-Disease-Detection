import shutil
import torch
import sys
import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Set, Type
from config.config import config, paths
from torch import nn
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import ModelEmaV2
from torch.optim.lr_scheduler import OneCycleLR
from datetime import datetime
import glob
import random
import time
import warnings

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(paths.utils_log),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Utils')

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

def save_checkpoint(state: Dict[str, Any], is_best: bool, fold: int) -> None:
    """保存模型检查点
    
    参数:
        state: 要保存的状态字典
        is_best: 是否是最佳模型
        fold: 折叠编号
    """
    os.makedirs(os.path.join(config.weights, config.model_name, str(fold)), exist_ok=True)
    os.makedirs(os.path.join(config.best_weights, config.model_name, str(fold)), exist_ok=True)
    
    filename = os.path.join(config.weights, config.model_name, str(fold), "_checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(config.best_weights, config.model_name, str(fold), 'model_best.pth.tar'))
        logger.info(f"Saved best model checkpoint to {filename}")

class AverageMeter:
    """计算并存储平均值和当前值的类"""

    def __init__(self):
        """初始化平均值计算器"""
        self.reset()

    def reset(self) -> None:
        """重置所有统计值"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """更新统计值
        
        参数:
            val: 当前值
            n: 当前批次大小
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int) -> None:
    """调整学习率
    
    参数:
        optimizer: 优化器
        epoch: 当前轮次
    """
    lr = config.lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logger.info(f"Adjusted learning rate to {lr}")

def get_optimizer(model: nn.Module, name: str = 'adamw') -> torch.optim.Optimizer:
    """获取优化器
    
    参数:
        model: 模型
        name: 优化器名称
    
    返回:
        优化器实例
    """
    if name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
    elif name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.lr, 
            momentum=0.9, 
            weight_decay=config.weight_decay
        )
    elif name == 'ranger':
        # Ranger优化器（RAdam + Lookahead）
        try:
            from ranger_adabelief import Ranger
            optimizer = Ranger(
                model.parameters(), 
                lr=config.lr, 
                weight_decay=config.weight_decay
            )
        except ImportError:
            logger.warning("Ranger optimizer not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=config.lr, 
                weight_decay=config.weight_decay
            )
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    
    # 如果使用Lookahead，包装优化器
    if config.use_lookahead and name != 'ranger':
        try:
            from lookahead import Lookahead
            optimizer = Lookahead(optimizer, k=5, alpha=0.5)
            logger.info("Applied Lookahead wrapper to optimizer")
        except ImportError:
            logger.warning("Lookahead not available")
    
    return optimizer

def get_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int, 
                 steps_per_epoch: Optional[int] = None) -> torch.optim.lr_scheduler._LRScheduler:
    """获取学习率调度器
    
    参数:
        optimizer: 优化器
        num_epochs: 总轮次
        steps_per_epoch: 每轮的步数（对于OneCycleLR需要）
    
    返回:
        学习率调度器
    """
    if config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=10, 
            gamma=0.1
        )
    elif config.scheduler == 'cosine':
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=1e-6,
            warmup_lr_init=config.lr * config.warmup_factor,
            warmup_t=config.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True
        )
    elif config.scheduler == 'onecycle':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch must be provided for OneCycleLR")
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=1000.0
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")
    
    logger.info(f"Created {config.scheduler} scheduler")
    return scheduler

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """执行Mixup数据增强
    
    参数:
        x: 输入数据批次
        y: 标签批次
        alpha: mixup强度参数
    
    返回:
        混合后的输入和标签，以及混合权重
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """执行CutMix数据增强
    
    参数:
        x: 输入数据批次
        y: 标签批次
        alpha: cutmix强度参数
    
    返回:
        混合后的输入和标签，以及混合权重
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    # 生成随机裁剪区域
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # 随机选择裁剪位置
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # 执行裁剪和混合
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # 调整混合权重
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, 
                   y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Mixup损失函数
    
    参数:
        criterion: 基础损失函数
        pred: 预测值
        y_a: 第一个标签
        y_b: 第二个标签
        lam: 混合权重
        
    返回:
        混合后的损失值
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """计算top-k准确率
    
    参数:
        output: 模型输出
        target: 目标标签
        topk: 要计算的top-k值
        
    返回:
        top-k准确率列表
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Logger:
    """日志记录器类"""
    
    def __init__(self):
        """初始化日志记录器"""
        self.file = None

    def open(self, file_path: Union[str, Path], mode: str = None) -> None:
        """打开日志文件
        
        参数:
            file_path: 日志文件路径
            mode: 打开模式
        """
        # 确保路径在logs目录下
        if isinstance(file_path, str) and not file_path.startswith(paths.logs_dir):
            file_path = os.path.join(paths.logs_dir, os.path.basename(file_path))
        elif isinstance(file_path, Path) and 'logs' not in file_path.parts:
            file_path = Path(paths.logs_dir) / file_path.name
        
        # 确保logs目录存在
        os.makedirs(paths.logs_dir, exist_ok=True)
        
        self.file = open(file_path, mode) if mode else open(file_path, 'w')

    def write(self, message: str, is_terminal: bool = 1, is_file: bool = 1) -> None:
        """写入日志消息
        
        参数:
            message: 日志消息
            is_terminal: 是否输出到终端
            is_file: 是否写入文件
        """
        if is_terminal:
            print(message, end='')
            sys.stdout.flush()
            
        if is_file:
            self.file.write(message)
            self.file.flush()

    def flush(self) -> None:
        """刷新日志文件"""
        self.file.flush()

def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """获取当前学习率
    
    参数:
        optimizer: 优化器
        
    返回:
        当前学习率
    """
    return optimizer.param_groups[0]['lr']

def time_to_str(t: float, mode: str = 'min') -> str:
    """将时间转换为字符串格式
    
    参数:
        t: 时间（秒）
        mode: 输出模式
        
    返回:
        格式化的时间字符串
    """
    if mode == 'min':
        if t > 3600:
            return '{:.1f}h'.format(t / 3600)
        elif t > 60:
            return '{:.1f}m'.format(t / 60)
        else:
            return '{:.1f}s'.format(t)
    elif mode == 'full':
        h = t // 3600
        m = (t - h * 3600) // 60
        s = t - h * 3600 - m * 60
        return '{:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s))
    else:
        raise ValueError(f"Unsupported time format mode: {mode}")

class FocalLoss(nn.Module):
    """Focal Loss实现"""
    
    def __init__(self, focusing_param: float = 2, balance_param: float = 0.25):
        """初始化Focal Loss
        
        参数:
            focusing_param: 聚焦参数gamma
            balance_param: 平衡参数alpha
        """
        super(FocalLoss, self).__init__()
        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        参数:
            output: 模型输出
            target: 目标标签
            
        返回:
            损失值
        """
        cross_entropy = F.cross_entropy(output, target, reduction='none')
        pt = torch.exp(-cross_entropy)
        focal_loss = self.balance_param * (1-pt)**self.focusing_param * cross_entropy
        return focal_loss.mean()

class ImprovedFocalLoss(nn.Module):
    """改进的Focal Loss实现，包含标签平滑"""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, label_smoothing: float = 0.1):
        """初始化改进的Focal Loss
        
        参数:
            gamma: 聚焦参数
            alpha: 平衡参数
            label_smoothing: 标签平滑系数
        """
        super(ImprovedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        参数:
            input: 模型输出
            target: 目标标签
            
        返回:
            损失值
        """
        ce_loss = F.cross_entropy(input, target, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingCrossEntropy(nn.Module):
    """带标签平滑的交叉熵损失"""
    
    def __init__(self, smoothing: float = 0.1):
        """初始化带标签平滑的交叉熵损失
        
        参数:
            smoothing: 平滑系数
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        参数:
            input: 模型输出
            target: 目标标签
            
        返回:
            损失值
        """
        logprobs = F.log_softmax(input, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def get_loss_function(device: torch.device) -> nn.Module:
    """获取损失函数
    
    参数:
        device: 设备
        
    返回:
        损失函数
    """
    if config.use_focal_loss:
        if config.label_smoothing > 0:
            criterion = ImprovedFocalLoss(
                gamma=2.0,
                alpha=0.25,
                label_smoothing=config.label_smoothing
            ).to(device)
            logger.info("Using Improved Focal Loss with label smoothing")
        else:
            criterion = FocalLoss().to(device)
            logger.info("Using Focal Loss")
    else:
        if config.label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(config.label_smoothing).to(device)
            logger.info("Using Cross Entropy with label smoothing")
        else:
            criterion = nn.CrossEntropyLoss().to(device)
            logger.info("Using standard Cross Entropy Loss")
    
    return criterion

class MyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理NumPy数据类型"""
    
    def default(self, obj: Any) -> Any:
        """处理特殊类型的对象
        
        参数:
            obj: 要编码的对象
            
        返回:
            可JSON序列化的对象
        """
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def create_model_ema(model: nn.Module) -> Optional[ModelEmaV2]:
    """创建EMA模型
    
    参数:
        model: 基础模型
        
    返回:
        EMA模型实例或None
    """
    if config.use_ema:
        model_ema = ModelEmaV2(model, decay=config.ema_decay)
        logger.info(f"Created EMA model with decay rate {config.ema_decay}")
        return model_ema
    return None

def update_ema(ema: ModelEmaV2, model: nn.Module, iter: int) -> None:
    """更新EMA模型
    
    参数:
        ema: EMA模型
        model: 基础模型
        iter: 当前迭代次数
    """
    if iter == 0:
        ema.reset_parameters(model)
    else:
        ema.update(model)

def handle_datasets(data_type: str = "train", list_only: bool = False) -> Union[str, List[str]]:
    """查找并处理特定类型的多个数据集
    
    参数:
        data_type: 数据集类型 'train', 'test' 或 'val'
        list_only: 是否只返回数据集列表而不选择单个数据集
        
    返回:
        最终使用的数据集路径或数据集路径列表
    """
    base_dir = paths.data_dir
    target_dirs = []
    
    # 查找所有相关的数据集目录
    if data_type == "train":
        # 检查合并目录是否存在且有内容
        if config.merge_datasets and os.path.exists(paths.merged_train_dir):
            merged_files = glob.glob(os.path.join(paths.merged_train_dir, "**", "*.*"), recursive=True)
            if merged_files:
                if not list_only:
                    logger.info(f"Using merged training dataset: {paths.merged_train_dir}")
                    return paths.merged_train_dir
                else:
                    # 如果只是列出所有数据集，则包含合并目录
                    target_dirs.append(paths.merged_train_dir)
        
        # 搜索训练数据集目录
        train_dirs = [d for d in glob.glob(f"{base_dir}/**/train", recursive=True)]
        target_dirs.extend(train_dirs)
        
        # 检查增强数据目录
        if config.use_data_aug and os.path.exists(paths.aug_train_dir):
            aug_files = glob.glob(os.path.join(paths.aug_train_dir, "**", "*.*"), recursive=True)
            if aug_files:
                if not list_only:
                    logger.info(f"Found augmented training data: {paths.aug_train_dir}")
                
                # 如果只是列出所有数据集或者需要合并增强数据，则包含增强目录
                if list_only or config.merge_augmented_data:
                    if paths.aug_train_dir not in target_dirs:
                        target_dirs.append(paths.aug_train_dir)
                        
    elif data_type == "test":
        # 检查合并目录是否存在且有内容
        if config.merge_datasets and os.path.exists(paths.merged_test_dir):
            merged_files = glob.glob(os.path.join(paths.merged_test_dir, "*.*"))
            if merged_files:
                if not list_only:
                    logger.info(f"Using merged test dataset: {paths.merged_test_dir}")
                    return paths.merged_test_dir
                else:
                    # 如果只是列出所有数据集，则包含合并目录
                    target_dirs.append(paths.merged_test_dir)
        
        # 查找所有测试目录，包括子目录
        test_dirs = [d for d in glob.glob(f"{base_dir}/**/test/images", recursive=True)]
        test_specific_dirs = [d for d in glob.glob(f"{base_dir}/test/*", recursive=False) if os.path.isdir(d)]
        target_dirs.extend(test_dirs + test_specific_dirs)
        
        # 如果设置了不使用所有测试集，则过滤只保留指定的测试集
        if not config.use_all_test_datasets and not list_only:
            filtered_dirs = []
            for d in target_dirs:
                if config.primary_test_dataset in d:
                    filtered_dirs.append(d)
            
            if filtered_dirs:
                target_dirs = filtered_dirs
                logger.info(f"Using only test dataset matching '{config.primary_test_dataset}'")
            else:
                logger.warning(f"No test dataset matching '{config.primary_test_dataset}' found, using all available")
                
    elif data_type == "val":
        # 检查合并目录是否存在且有内容
        if config.merge_datasets and os.path.exists(paths.merged_val_dir):
            merged_files = glob.glob(os.path.join(paths.merged_val_dir, "*.*"))
            if merged_files:
                if not list_only:
                    logger.info(f"Using merged validation dataset: {paths.merged_val_dir}")
                    return paths.merged_val_dir
                else:
                    # 如果只是列出所有数据集，则包含合并目录
                    target_dirs.append(paths.merged_val_dir)
        
        # 搜索验证数据集目录
        val_dirs = [d for d in glob.glob(f"{base_dir}/**/val", recursive=True)]
        target_dirs.extend(val_dirs)
    
    logger.info(f"Found {len(target_dirs)} {data_type} datasets: {target_dirs}")
    
    # 如果只需要列出数据集，直接返回列表
    if list_only:
        return target_dirs
    
    # 如果没有找到，返回默认路径
    if not target_dirs:
        if data_type == "train":
            return config.train_data
        elif data_type == "test":
            return config.test_data
        else:
            return config.val_data
    
    # 只有一个数据集时，直接返回
    if len(target_dirs) == 1:
        logger.info(f"Using single {data_type} dataset: {target_dirs[0]}")
        return target_dirs[0]
    
    # 多个数据集时的处理逻辑
        # 根据策略选择一个数据集
        selected_dir = None
        
        if config.dataset_to_use == "first":
            selected_dir = target_dirs[0]
        logger.info(f"Selected first {data_type} dataset: {selected_dir}")
        
        elif config.dataset_to_use == "last":
            selected_dir = target_dirs[-1]
        logger.info(f"Selected last {data_type} dataset: {selected_dir}")
        
        elif config.dataset_to_use == "specific":
            # 查找特定名称的数据集
            for dir_path in target_dirs:
                if config.specific_dataset in dir_path:
                    selected_dir = dir_path
                logger.info(f"Found specified {data_type} dataset: {selected_dir}")
                    break
            if selected_dir is None:
            logger.warning(f"Could not find specified {data_type} dataset: {config.specific_dataset}, using largest dataset instead")
                selected_dir = max(target_dirs, key=lambda d: len(glob.glob(os.path.join(d, "**/*"), recursive=True)))
        
        else:  # "auto" 或其他未知选项
            # 使用文件数量最多的数据集
            selected_dir = max(target_dirs, key=lambda d: len(glob.glob(os.path.join(d, "**/*"), recursive=True)))
    logger.info(f"Auto-selected largest {data_type} dataset: {selected_dir}")
        
        return selected_dir

def test_dataset_handling():
    """测试数据集处理功能"""
    logger.info("=== Testing Dataset Handling ===")
    
    # 测试列出数据集
    logger.info("Testing dataset listing...")
    train_datasets = handle_datasets(data_type="train", list_only=True)
    logger.info(f"Found {len(train_datasets)} training datasets")
    
    test_datasets = handle_datasets(data_type="test", list_only=True)
    logger.info(f"Found {len(test_datasets)} test datasets")
    
    val_datasets = handle_datasets(data_type="val", list_only=True)
    logger.info(f"Found {len(val_datasets)} validation datasets")
    
    # 测试数据集合并
    logger.info("Testing dataset merging...")
    old_setting = config.merge_datasets
    config.merge_datasets = True
    
    # 测试训练集处理
    train_path = handle_datasets(data_type="train")
    logger.info(f"Selected training dataset path: {train_path}")
    
    # 测试测试集处理
    test_path = handle_datasets(data_type="test")
    logger.info(f"Selected test dataset path: {test_path}")
    
    # 测试验证集处理
    val_path = handle_datasets(data_type="val")
    logger.info(f"Selected validation dataset path: {val_path}")
    
    # 恢复原始设置
    config.merge_datasets = old_setting
    
    logger.info("=== Testing Complete ===")
    return train_path, test_path, val_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Processing Tool")
    parser.add_argument('--test', action='store_true', help='Test dataset handling functionality')
    parser.add_argument('--merge', action='store_true', help='Set dataset merging mode')
    parser.add_argument('--mode', choices=['auto', 'first', 'last', 'specific'], 
                        default='auto', help='Dataset selection mode')
    parser.add_argument('--dataset', type=str, default='', help='Specific dataset name')
    
    args = parser.parse_args()
    
    if args.merge:
        config.merge_datasets = True
        logger.info("Dataset merging enabled")
        
    if args.mode:
        config.dataset_to_use = args.mode
        logger.info(f"Dataset selection mode set to: {args.mode}")
        
    if args.dataset:
        config.specific_dataset = args.dataset
        logger.info(f"Specific dataset name set to: {args.dataset}")
    
    if args.test:
        test_dataset_handling()

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """计算top-k准确率
    
    参数:
        output: 模型输出
        target: 目标标签
        topk: 要计算的top-k值
        
    返回:
        top-k准确率列表
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res 
