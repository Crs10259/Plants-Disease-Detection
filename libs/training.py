import os 
import random 
import torch
import numpy as np 
import warnings
import logging
from config import config, paths
from torch.utils.data import DataLoader
from dataset.dataloader import *    
from timeit import default_timer as timer
from models.model import *
from utils.utils import *
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Any
from utils.utils import ModelEmaV2

class Trainer:
    """训练管理器类，封装所有训练功能"""
    
    def __init__(self, config, logger=None):
        """初始化训练器
        
        参数:
            config: 配置对象
            logger: 可选的日志记录器实例
        """
        self.config = config
        
        # 设置日志记录
        self.logger = logger or self._setup_logger()
        
        # 初始化环境和设备
        self.setup_environment()
        self.device = self.get_device()
        self.create_directories()
        
        # 初始化性能监控
        self.memory_tracker = MemoryTracker()
        self.performance_metrics = PerformanceMetrics()
        
    def _setup_logger(self):
        """设置并返回训练日志记录器"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(paths.training_log, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        
        # 确保训练日志记录器正确设置
        logger = logging.getLogger('Training')
        
        # 删除旧的handler以避免重复
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # 添加新的handler
        file_handler = logging.FileHandler(paths.training_log, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # 设置日志级别
        logger.setLevel(logging.DEBUG)
        
        return logger
        
    def setup_environment(self):
        """设置随机种子和CUDA环境"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpus
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        torch.backends.cudnn.benchmark = True
        warnings.filterwarnings('ignore')

    def get_device(self):
        """获取训练设备"""
        if self.config.device == "cuda":
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
            else:
                self.logger.warning('CUDA requested but not available, falling back to CPU')
                device = torch.device('cpu')
                self.logger.info('Using CPU')
        elif self.config.device == "cpu":
            device = torch.device('cpu')
            self.logger.info('Using CPU (as requested)')
        else:
            # Auto mode
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
            else:
                device = torch.device('cpu')
                self.logger.info('Using CPU (GPU is not available)')
        return device

    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.submit, 
            self.config.weights, 
            self.config.best_weights, 
            self.config.logs,
            self.config.test_data
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            if directory in [self.config.best_weights, self.config.weights]:
                os.makedirs(os.path.join(directory, self.config.model_name, "0"), exist_ok=True)

    def train_epoch(self, model, train_dataloader, 
                   criterion, optimizer, epoch, 
                   log=None, scaler=None, 
                   model_ema=None):
        """训练单个轮次
        
        参数:
            model: 模型
            train_dataloader: 训练数据加载器
            criterion: 损失函数
            optimizer: 优化器
            epoch: 当前轮次
            log: 日志记录器，如果为None则创建
            scaler: 用于混合精度的梯度缩放器
            model_ema: EMA模型
            
        返回:
            训练损失和准确率
        """
        # Create a logger if not provided
        if log is None:
            log = init_logger(f'train_epoch_{epoch}.log')
            
        train_losses = AverageMeter()
        train_top1 = AverageMeter()
        train_top2 = AverageMeter()
        
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}')

        error_files = []
        batch_times = []
        
        # 强制初始垃圾回收
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 添加异常计数器和最大尝试次数
        error_count = 0
        max_errors = 50  # 允许的最大连续错误数
        consecutive_errors = 0
        
        for iter, batch in enumerate(progress_bar):
            try:
                # 检查是否应该终止训练（如果连续错误过多）
                if consecutive_errors >= max_errors:
                    log.write(f"Too many consecutive errors ({consecutive_errors}), stopping training\n")
                    self.logger.error(f"Too many consecutive errors ({consecutive_errors}), stopping training")
                    break
                    
                batch_start = timer()
                
                # Ensure batch is valid
                if len(batch) != 2:
                    log.write(f"Skipping error batch in iteration {iter}\n")
                    consecutive_errors += 1
                    continue
                    
                input, target = batch
                
                # Ensure data is valid
                if input is None or len(input) == 0:
                    log.write(f"Skipping empty input in iteration {iter}\n")
                    consecutive_errors += 1
                    continue
                
                # 检查输入张量是否有空值或无穷大
                if torch.isnan(input).any() or torch.isinf(input).any():
                    log.write(f"Skipping batch with NaN or Inf values in iteration {iter}\n")
                    consecutive_errors += 1
                    continue
                    
                input = input.to(self.device)
                target = torch.tensor(target).to(self.device)
                
                # Apply Mixup or CutMix data augmentation
                if self.config.use_mixup:
                    try:
                        # Randomly choose between Mixup and CutMix
                        r = np.random.rand(1)
                        if r < self.config.cutmix_prob:
                            # Use CutMix
                            input, target_a, target_b, lam = cutmix_data(input, target, self.config.mixup_alpha)
                            use_mixup = True
                        else:
                            # Use Mixup
                            input, target_a, target_b, lam = mixup_data(input, target, self.config.mixup_alpha)
                            use_mixup = True
                    except Exception as e:
                        log.write(f"Error applying mixup/cutmix in iteration {iter}: {str(e)}, continuing without augmentation\n")
                        use_mixup = False
                else:
                    use_mixup = False
                
                # Use mixed precision training
                if self.config.use_amp and scaler is not None:
                    with autocast():
                        # Forward pass
                        output = model(input)
                        
                        # Calculate loss
                        if use_mixup:
                            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
                        else:
                            loss = criterion(output, target)
                    
                    # Backward pass (with gradient scaling)
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.config.gradient_clip_val > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_val)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Forward pass
                    output = model(input)
                    
                    # Calculate loss
                    if use_mixup:
                        loss = mixup_criterion(criterion, output, target_a, target_b, lam)
                    else:
                        loss = criterion(output, target)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_val)
                    
                    optimizer.step()
                
                # Update EMA model
                if model_ema is not None:
                    try:
                        update_ema(model_ema, model, iter)
                    except Exception as e:
                        log.write(f"Error updating EMA in iteration {iter}: {str(e)}\n")
                
                # Calculate metrics
                if use_mixup:
                    # For Mixup, use the first label to calculate accuracy (approximation)
                    precision1_train, precision2_train = accuracy(output, target_a, topk=(1, 2))
                else:
                    precision1_train, precision2_train = accuracy(output, target, topk=(1, 2))
                    
                train_losses.update(loss.item(), input.size(0))
                train_top1.update(precision1_train.item(), input.size(0))
                train_top2.update(precision2_train.item(), input.size(0))
                
                # Calculate batch time
                batch_time = timer() - batch_start
                batch_times.append(batch_time)
                
                # 重置连续错误计数器
                consecutive_errors = 0
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{train_losses.avg:.3f}',
                    'top1': f'{train_top1.avg:.3f}',
                    'top2': f'{train_top2.avg:.3f}',
                    'batch_time': f'{np.mean(batch_times[-100:]):.3f}s'
                })
                
                # Monitor memory usage
                if iter % 10 == 0:  # Check every 10 batches
                    self.memory_tracker.update()
                    if self.memory_tracker.should_warn():
                        self.logger.warning(self.memory_tracker.get_warning())
                
                # 周期性清理内存
                if iter % 100 == 0:  # 每100个批次进行一次垃圾回收
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Free memory
                del output
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                consecutive_errors += 1
                error_count += 1
                if hasattr(batch, '__getitem__') and len(batch) > 1 and isinstance(batch[1], list) and len(batch[1]) > 0:
                    # Record error files
                    error_files.extend(batch[1])
                    log.write(f"Error in training iteration {iter}: {str(e)} - these files will be skipped in the future\n")
                else:
                    log.write(f"Error in training iteration {iter}: {str(e)}\n")
                continue
                
        if error_files:
            log.write(f"Found {len(error_files)} problematic files in this epoch.\n")
        
        if error_count > 0:
            log.write(f"Total errors in this epoch: {error_count}\n")
        
        # 最终清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Update performance metrics
        self.performance_metrics.update_epoch_metrics(
            epoch=epoch,
            loss=train_losses.avg,
            top1=train_top1.avg,
            top2=train_top2.avg,
            batch_time=np.mean(batch_times),
            memory_usage=self.memory_tracker.get_current_usage()
        )
        
        return train_losses.avg, train_top1.avg, train_top2.avg

    def _get_val_loader(self):
        """基于配置设置创建并返回验证数据加载器
        
        返回基于合并数据集的适当数据加载器（如果有），
        否则返回最佳可用数据集。
        """
        # 使用handle_datasets函数获取适当的验证数据路径
        val_path = handle_datasets(data_type="val")
        
        self.logger.info(f"Using validation data from: {val_path}")
        
        # 检查数据集是否存在
        if not os.path.exists(val_path):
            self.logger.error(f"Validation data path does not exist: {val_path}")
            if os.path.exists(self.config.val_data):
                self.logger.info(f"Falling back to default validation data: {self.config.val_data}")
                val_path = self.config.val_data
            else:
                self.logger.warning(f"Cannot find validation data at {val_path} or {self.config.val_data}")
                return None
        
        # 检查目录中是否有图像文件
        image_files = 0
        for root, _, files in os.walk(val_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files += 1
        
        if image_files == 0:
            self.logger.warning(f"No image files found in validation path: {val_path}")
            return None
        
        # 获取数据集文件列表
        val_files = get_files(val_path, mode="val")
        
        # 记录验证文件数量
        self.logger.info(f"Validation dataset contains {len(val_files)} images")
        
        # 创建数据集和数据加载器
        val_dataset = PlantDiseaseDataset(val_files, sampling_threshold=config.sampling_threshold)
        return DataLoader(
            val_dataset, 
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def train(self, epochs=None, train_loader=None, val_loader=None):
        """完整的训练循环，包含验证
        
        参数:
            epochs: 轮次数（默认使用config.epoch）
            train_loader: 训练数据加载器（如果为None则创建）
            val_loader: 验证数据加载器（如果为None则创建）
            
        返回:
            包含训练结果的字典
        """
        epochs = epochs or self.config.epoch
        
        # 获取训练和验证数据加载器
        train_loader = train_loader or self._get_train_loader()
        if val_loader is None:
            val_loader = self._get_val_loader()
            if val_loader is None:
                self.logger.warning("No validation data available. Training will proceed without validation.")
        
        # Check for existing models and number of trained epochs
        start_epoch = 0
        best_acc = 0.0
        checkpoint_path = os.path.join(self.config.weights, self.config.model_name, "0", "_latest_model.pth.tar")
        best_model_path = os.path.join(self.config.best_weights, self.config.model_name, "0", "best_model.pth.tar")
        
        if os.path.exists(checkpoint_path) or os.path.exists(best_model_path):
            model_path = best_model_path if os.path.exists(best_model_path) else checkpoint_path
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                start_epoch = checkpoint.get('epoch', 0)
                best_acc = checkpoint.get('best_acc', 0.0)
                
                if start_epoch >= epochs:
                    self.logger.info(f"Model already trained for {start_epoch} epochs (configured: {epochs})")
                    return {"completed": True, "epochs_trained": start_epoch, "best_acc": best_acc}
                else:
                    self.logger.info(f"Continuing training from epoch {start_epoch}/{epochs}")
            except Exception as e:
                self.logger.warning(f"Error loading existing checkpoint: {str(e)}. Starting from epoch 0.")
                start_epoch = 0
        
        # Setup for training
        model = get_net(model_name=config.model_name, num_classes=config.num_classes, pretrained=config.pretrained)
        
        # Load weights if we're continuing training
        if start_epoch > 0:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                    if "optimizer" in checkpoint:
                        optimizer_state = checkpoint["optimizer"]
                else:
                    model.load_state_dict(checkpoint)
                self.logger.info(f"Loaded weights from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load weights: {str(e)}. Starting with fresh model.")
        
        # 确保模型和数据都转移到正确的设备
        model = model.to(self.device)
        self.logger.info(f"Model moved to {self.device}")
        
        # 验证CUDA是否可用
        if torch.cuda.is_available():
            self.logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            self.logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            self.logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Create a detailed training log
        train_log = init_logger('train_detailed.log')
        self.logger.info("Training started")
        
        # Get loss function and optimizer
        criterion = get_loss_function(self.device)
        optimizer = get_optimizer(model, self.config.optimizer)
        
        # Restore optimizer state if continuing training
        if start_epoch > 0 and 'optimizer_state' in locals():
            try:
                optimizer.load_state_dict(optimizer_state)
                self.logger.info("Restored optimizer state")
            except Exception as e:
                self.logger.warning(f"Failed to restore optimizer state: {str(e)}")
        
        # Set up learning rate scheduler
        scheduler = get_scheduler(optimizer, epochs, len(train_loader))
        
        # Initialize mixed precision training
        scaler = None
        if self.config.use_amp and self.device.type == 'cuda':
            scaler = GradScaler()
            
        # Create model EMA
        model_ema = None
        if self.config.use_ema:
            model_ema = create_model_ema(model)
            
        # Training loop
        for epoch in range(start_epoch, epochs):
            # Train for one epoch
            train_loss, train_acc, _ = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch, train_log, scaler, model_ema
            )
            
            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    # OneCycleLR doesn't need epoch parameter
                    scheduler.step()
                else:
                    # For other schedulers (CosineLRScheduler, StepLR), pass epoch
                    scheduler.step(epoch)
                
            # Validate if validation loader is available
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = self.validate(model, val_loader, criterion, epoch)
                
                # Save checkpoint if validation accuracy improved
                is_best = val_acc > best_acc
                best_acc = max(val_acc, best_acc)
                
                save_latest_model({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best, fold=0)
                
                # Log validation results
                self.logger.info(f'Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | '
                           f'Val loss: {val_loss:.4f}, acc: {val_acc:.4f}')
                
                # Early stopping
                if self.config.use_early_stopping and self.performance_metrics.should_stop(val_loss):
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                # Log training results when no validation
                self.logger.info(f'Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, acc: {train_acc:.4f}')
                
                # Save checkpoint based on training accuracy
                is_best = train_acc > best_acc
                best_acc = max(train_acc, best_acc)
                
                save_latest_model({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best, fold=0)
        
        # Return training summary
        self.logger.info("Training completed successfully")
        return self.performance_metrics.get_summary()
    
    def validate(self, model, val_loader, criterion, epoch):
        """验证模型
        
        参数:
            model: 要验证的模型
            val_loader: 验证数据加载器
            criterion: 损失函数
            epoch: 当前轮次
            
        返回:
            验证损失和准确率
        """
        val_losses = AverageMeter()
        val_top1 = AverageMeter()
        
        # Switch to evaluation mode
        model.eval()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
            for input, target in pbar:
                input = input.to(self.device)
                target = torch.tensor(target).to(self.device)
                
                # Forward pass
                output = model(input)
                loss = criterion(output, target)
                
                # Measure accuracy and record loss
                prec1, _ = accuracy(output, target, topk=(1, 2))
                val_losses.update(loss.item(), input.size(0))
                val_top1.update(prec1.item(), input.size(0))
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{val_losses.avg:.3f}',
                    'top1': f'{val_top1.avg:.3f}'
                })
        
        return val_losses.avg, val_top1.avg
    
    def _get_train_loader(self):
        """基于配置设置创建并返回训练数据加载器
        
        返回基于合并数据集的适当数据加载器（如果有），
        否则返回最佳可用数据集。
        """
        # 使用handle_datasets函数获取适当的训练数据路径
        # 该函数已经修改为优先使用合并的数据集(如果有)，否则使用配置策略选择单个数据集
        train_path = handle_datasets(data_type="train")
        
        self.logger.info(f"Using training data from: {train_path}")
        
        # 检查数据集是否存在
        if not os.path.exists(train_path):
            self.logger.error(f"Training data path does not exist: {train_path}")
            if os.path.exists(self.config.train_data):
                self.logger.info(f"Falling back to default training data: {self.config.train_data}")
                train_path = self.config.train_data
            else:
                raise FileNotFoundError(f"Cannot find training data at {train_path} or {self.config.train_data}")
        
        # 检查目录中是否有图像文件
        image_files = 0
        for root, _, files in os.walk(train_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files += 1
        
        if image_files == 0:
            self.logger.error(f"No image files found in training path: {train_path}")
            raise ValueError(f"No training images found in {train_path}")
        
        # 获取数据集文件列表
        train_files = get_files(train_path, mode="train")
        
        # 记录训练文件数量
        self.logger.info(f"Training dataset contains {len(train_files)} images")
        
        # 创建数据集和数据加载器
        train_dataset = PlantDiseaseDataset(train_files, sampling_threshold=config.sampling_threshold)
        return DataLoader(
            train_dataset, 
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

class MemoryTracker:
    """内存使用监视器"""
    
    def __init__(self, warning_threshold: float = 0.9):
        """初始化内存监视器
        
        参数:
            warning_threshold: 内存使用警告阈值（占总内存的比例）
        """
        self.warning_threshold = warning_threshold
        self.current_usage = 0
        self.peak_usage = 0
        
    def update(self) -> None:
        """更新内存使用统计"""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.peak_usage = max(self.peak_usage, current)
            self.current_usage = current
        
    def should_warn(self) -> bool:
        """检查是否应该发出警告"""
        return self.current_usage > self.warning_threshold
        
    def get_warning(self) -> str:
        """获取警告消息"""
        return f"High memory usage: {self.current_usage:.1%} of available memory"
        
    def get_current_usage(self) -> float:
        """获取当前内存使用率"""
        return self.current_usage

class PerformanceMetrics:
    """性能指标跟踪器"""
    
    def __init__(self):
        """初始化性能指标跟踪器"""
        self.metrics = {
            'loss': [],
            'top1': [],
            'top2': [],
            'batch_time': [],
            'memory_usage': [],
            'val_loss': []  # For early stopping
        }
        self.epochs = []
        self.patience_counter = 0
        
    def update_epoch_metrics(self, epoch: int, loss: float, top1: float, 
                           top2: float, batch_time: float, memory_usage: float,
                           val_loss: float = None) -> None:
        """更新每轮性能指标
        
        参数:
            epoch: 轮次
            loss: 损失值
            top1: Top-1准确率
            top2: Top-2准确率
            batch_time: 平均批次时间
            memory_usage: 内存使用率
            val_loss: 验证损失（可选）
        """
        self.epochs.append(epoch)
        self.metrics['loss'].append(loss)
        self.metrics['top1'].append(top1)
        self.metrics['top2'].append(top2)
        self.metrics['batch_time'].append(batch_time)
        self.metrics['memory_usage'].append(memory_usage)
        
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        
    def should_stop(self, val_loss: float) -> bool:
        """检查训练是否应该提前停止
        
        参数:
            val_loss: 当前验证损失
            
        返回:
            表示是否停止训练的布尔值
        """
        if not self.metrics['val_loss']:
            self.metrics['val_loss'].append(val_loss)
            return False
            
        if val_loss >= min(self.metrics['val_loss']):
            self.patience_counter += 1
        else:
            self.patience_counter = 0
            
        self.metrics['val_loss'].append(val_loss)
        return self.patience_counter >= config.early_stopping_patience
        
    def get_summary(self) -> Dict[str, Any]:
        """获取性能指标摘要"""
        return {
            'best_top1': max(self.metrics['top1']),
            'best_epoch': self.epochs[np.argmax(self.metrics['top1'])],
            'avg_batch_time': np.mean(self.metrics['batch_time']),
            'peak_memory': max(self.metrics['memory_usage'])
        }

def init_logger(log_name='train_details.log') -> Logger:
    """初始化Logger对象
    
    参数:
        log_name: 日志文件名
        
    返回:
        初始化后的Logger对象
    """
    # Ensure logs directory exists
    os.makedirs(paths.log_dir, exist_ok=True)
    
    log = Logger()
    log.open(os.path.join(paths.log_dir, log_name))
    return log

def train_model(cfg=None):
    """使用给定配置训练模型
    
    参数:
        cfg: 可选配置对象（如果为None则使用默认配置）
        
    返回:
        包含训练结果的字典
    """
    try:
        # Use provided config or default
        trainer = Trainer(cfg or config)
        return trainer.train()
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        return {"error": str(e)} 