import os 
import random 
import json
import torch
import numpy as np 
import pandas as pd 
import warnings
import psutil
import logging
import glob
from datetime import datetime
from torch import nn, optim
from config.config import config, paths
from collections import OrderedDict
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split    
from timeit import default_timer as timer
from models.model import *
from utils.utils import *
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import timm.utils
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

class Trainer:
    """Training manager class that encapsulates all training functionality"""
    
    def __init__(self, config, logger=None):
        """Initialize the trainer
        
        Args:
            config: Configuration object
            logger: Optional logger instance
        """
        self.config = config
        
        # Set up logging
        self.logger = logger or self._setup_logger()
        
        # Initialize environment and devices
        self.setup_environment()
        self.device = self.get_device()
        self.create_directories()
        
        # Initialize performance monitoring
        self.memory_tracker = MemoryTracker()
        self.performance_metrics = PerformanceMetrics()
        
    def _setup_logger(self):
        """Set up and return a logger for training"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(paths.training_log),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('Training')
        
    def setup_environment(self) -> None:
        """Set random seeds and CUDA environment"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpus
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        torch.backends.cudnn.benchmark = True
        warnings.filterwarnings('ignore')

    def get_device(self) -> torch.device:
        """Get training device"""
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

    def create_directories(self) -> None:
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

    def train_epoch(self, model: nn.Module, train_dataloader: DataLoader, 
                   criterion: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                   log: Optional[Logger] = None, scaler: Optional[GradScaler] = None, 
                   model_ema: Optional[ModelEmaV2] = None) -> Tuple[float, float, float]:
        """Train for a single epoch
        
        Args:
            model: Model
            train_dataloader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            epoch: Current epoch
            log: Log recorder, created if None
            scaler: Gradient scaler for mixed precision
            model_ema: EMA model
            
        Returns:
            Training loss and accuracy
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
        
        for iter, batch in enumerate(progress_bar):
            batch_start = timer()
            
            try:
                # Ensure batch is valid
                if len(batch) != 2:
                    log.write(f"Skipping error batch in iteration {iter}\n")
                    continue
                    
                input, target = batch
                
                # Ensure data is valid
                if input is None or len(input) == 0:
                    log.write(f"Skipping empty input in iteration {iter}\n")
                    continue
                    
                input = input.to(self.device)
                target = torch.tensor(target).to(self.device)
                
                # Apply Mixup or CutMix data augmentation
                if self.config.use_mixup:
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
                    update_ema(model_ema, model, iter)
                
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
                
                # Free memory
                del output
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                if hasattr(batch, '__getitem__') and len(batch) > 1 and isinstance(batch[1], list) and len(batch[1]) > 0:
                    # Record error files
                    error_files.extend(batch[1])
                    log.write(f"Error in training iteration {iter}: {str(e)} - these files will be skipped in the future\n")
                else:
                    log.write(f"Error in training iteration {iter}: {str(e)}\n")
                continue
                
        if error_files:
            log.write(f"Found {len(error_files)} problematic files in this epoch.\n")
        
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

    def train(self, epochs=None, train_loader=None, val_loader=None):
        """Full training loop with validation
        
        Args:
            epochs: Number of epochs (defaults to config.epoch)
            train_loader: Training data loader (created if None)
            val_loader: Validation data loader (created if None)
            
        Returns:
            Dictionary with training results
        """
        epochs = epochs or self.config.epoch
        
        # Check for existing models and number of trained epochs
        start_epoch = 0
        best_acc = 0.0
        checkpoint_path = os.path.join(self.config.weights, self.config.model_name, "0", "_checkpoint.pth.tar")
        best_model_path = os.path.join(self.config.best_weights, self.config.model_name, "0", "model_best.pth.tar")
        
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
        model = get_net()
        
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
        
        model = model.to(self.device)
        
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
        train_loader = train_loader or self._get_train_loader()
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
                scheduler.step()
                
            # Validate if validation loader is available
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = self.validate(model, val_loader, criterion, epoch)
                
                # Save checkpoint if validation accuracy improved
                is_best = val_acc > best_acc
                best_acc = max(val_acc, best_acc)
                
                save_checkpoint({
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
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best, fold=0)
        
        # Return training summary
        self.logger.info("Training completed successfully")
        return self.performance_metrics.get_summary()
    
    def validate(self, model, val_loader, criterion, epoch):
        """Validate the model
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            criterion: Loss function
            epoch: Current epoch
            
        Returns:
            Validation loss and accuracy
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
        """Create and return training data loader based on configuration settings
        
        Returns the appropriate dataloader based on merged dataset if available,
        otherwise falls back to the best available dataset.
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
        train_dataset = PlantDiseaseDataset(train_files, train=True)
        return DataLoader(
            train_dataset, 
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

class MemoryTracker:
    """Memory usage monitor"""
    
    def __init__(self, warning_threshold: float = 0.9):
        """Initialize memory monitor
        
        Args:
            warning_threshold: Memory usage warning threshold (as proportion of total memory)
        """
        self.warning_threshold = warning_threshold
        self.current_usage = 0
        self.peak_usage = 0
        
    def update(self) -> None:
        """Update memory usage statistics"""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.peak_usage = max(self.peak_usage, current)
            self.current_usage = current
        
    def should_warn(self) -> bool:
        """Check if warning should be issued"""
        return self.current_usage > self.warning_threshold
        
    def get_warning(self) -> str:
        """Get warning message"""
        return f"High memory usage: {self.current_usage:.1%} of available memory"
        
    def get_current_usage(self) -> float:
        """Get current memory usage"""
        return self.current_usage

class PerformanceMetrics:
    """Performance metrics tracker"""
    
    def __init__(self):
        """Initialize performance metrics tracker"""
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
        """Update per-epoch performance metrics
        
        Args:
            epoch: Epoch
            loss: Loss value
            top1: Top-1 accuracy
            top2: Top-2 accuracy
            batch_time: Average batch time
            memory_usage: Memory usage rate
            val_loss: Validation loss (optional)
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
        """Check if training should stop early
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            Boolean indicating whether to stop training
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
        """Get performance metrics summary"""
        return {
            'best_top1': max(self.metrics['top1']),
            'best_epoch': self.epochs[np.argmax(self.metrics['top1'])],
            'avg_batch_time': np.mean(self.metrics['batch_time']),
            'peak_memory': max(self.metrics['memory_usage'])
        }

def init_logger(log_name='train_details.log') -> Logger:
    """Initialize a Logger object
    
    Args:
        log_name: Log file name
        
    Returns:
        Initialized Logger object
    """
    # Ensure logs directory exists
    os.makedirs(paths.logs_dir, exist_ok=True)
    
    log = Logger()
    log.open(os.path.join(paths.logs_dir, log_name))
    return log

def train_model(cfg=None):
    """Train model with given configuration
    
    Args:
        cfg: Optional configuration object (uses default if None)
        
    Returns:
        Dictionary with training results
    """
    try:
        # Use provided config or default
        trainer = Trainer(cfg or config)
        return trainer.train()
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        return {"error": str(e)} 