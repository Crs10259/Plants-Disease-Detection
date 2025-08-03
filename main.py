#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import time
import glob
from typing import Dict, Any

# 将项目根目录添加到路径
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

# 导入配置
from config.config import config, paths
from dataset.data_prep import normalize_path, setup_data
from libs.inference import predict

# 确保处理器的级别正确设置
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.setLevel(logging.INFO)

logger = logging.getLogger('Main')

def add_train_arguments(train_parser: argparse.ArgumentParser) -> None:
    """添加训练相关的参数到解析器
    
    参数:
        train_parser: 训练命令的参数解析器
    """
    # 基本训练参数
    train_parser.add_argument('--epochs', type=int, 
                             help='Number of epochs (default: from config)')
    train_parser.add_argument('--model', type=str, 
                             help='Model name (default: from config)')
    train_parser.add_argument('--batch-size', type=int, 
                             help='Batch size (default: from config)')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    
    # 数据准备标志
    train_parser.add_argument('--prepare', action='store_true', 
                             help='Run data preparation before training')
    train_parser.add_argument('--no-prepare', action='store_true', 
                             help='Skip data preparation before training')
    train_parser.add_argument('--force-train', action='store_true', 
                             help='Force retraining even if model is already trained')
    train_parser.add_argument('--merge-augmented', action='store_true', 
                             help='Merge augmented data with original training data')
    train_parser.add_argument('--no-merge-augmented', action='store_true', 
                             help='Do not merge augmented data with original training data')
    train_parser.add_argument('--dataset-path', type=str, 
                             help='Custom path to dataset files or directory')
    train_parser.add_argument('--cleanup', action='store_true', 
                             help='Clean up temporary files after processing and training')
    train_parser.add_argument('--force-cleanup', action='store_true', 
                             help='Force cleanup without asking for confirmation')
    
    # 优化器选项
    train_parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd', 'ranger'],
                             help='Optimizer selection (default: from config)')
    train_parser.add_argument('--weight-decay', type=float,
                             help='Weight decay for optimizer (default: from config)')
    train_parser.add_argument('--no-lookahead', action='store_true',
                             help='Disable Lookahead optimizer wrapper')
    
    # 学习率调度器选项
    train_parser.add_argument('--scheduler', type=str, choices=['step', 'cosine', 'onecycle'],
                             help='Learning rate scheduler (default: from config)')
    train_parser.add_argument('--warmup-epochs', type=int,
                             help='Number of warmup epochs (default: from config)')
    train_parser.add_argument('--warmup-factor', type=float,
                             help='Warmup factor (default: from config)')
    
    # Mixup 和 CutMix 选项
    train_parser.add_argument('--no-mixup', action='store_true',
                             help='Disable Mixup data augmentation')
    train_parser.add_argument('--mixup-alpha', type=float,
                             help='Mixup alpha parameter (default: from config)')
    train_parser.add_argument('--cutmix-prob', type=float,
                             help='CutMix probability (default: from config)')
    train_parser.add_argument('--no-random-erasing', action='store_true',
                             help='Disable random erasing augmentation')
    train_parser.add_argument('--disable-augmentation', action='store_true',
                             help='Disable all data augmentation (overrides config.use_data_aug)')
    train_parser.add_argument('--enable-augmentation', action='store_true',
                             help='Enable all data augmentation (overrides config.use_data_aug)')
    
    # 早停参数
    train_parser.add_argument('--no-early-stopping', action='store_true',
                             help='Disable early stopping')
    train_parser.add_argument('--patience', type=int,
                             help='Early stopping patience (default: from config)')
    
    # 混合精度选项
    train_parser.add_argument('--no-amp', action='store_true',
                             help='Disable automatic mixed precision training')
    
    # 梯度裁剪选项
    train_parser.add_argument('--gradient-clip-val', type=float,
                             help='Gradient clipping value (default: from config)')
    
    # EMA 选项
    train_parser.add_argument('--no-ema', action='store_true', 
                             help='Disable Exponential Moving Average (EMA)')
    train_parser.add_argument('--ema-decay', type=float,
                             help='EMA decay rate (default: from config)')
    
    # 设备选项
    train_parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'],
                             help='Device to use for training (default: from config)')
    train_parser.add_argument('--gpus', type=str,
                             help='GPU device IDs to use, comma separated (e.g., "0,1")')
    
    # 高级选项
    train_parser.add_argument('--seed', type=int,
                             help='Random seed for reproducibility (default: from config)')
    train_parser.add_argument('--label-smoothing', type=float,
                             help='Label smoothing coefficient (default: from config)')
    train_parser.add_argument('--no-gradient-checkpointing', action='store_true',
                             help='Disable gradient checkpointing')

def setup_parser() -> argparse.ArgumentParser:
    """设置命令行参数解析器
    
    返回:
        配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        description="Plant Disease Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 为不同的命令创建子解析器
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # 数据准备命令
    prep_parser = subparsers.add_parser('prepare', 
                                        help='Prepare datasets (extract, process, augment)')
    prep_parser.add_argument('--extract', action='store_true', 
                            help='Extract dataset archives')
    prep_parser.add_argument('--process', action='store_true', 
                            help='Process images from temp directory into training directory')
    prep_parser.add_argument('--augment', action='store_true', 
                            help='Perform data augmentation')
    prep_parser.add_argument('--status', action='store_true', 
                            help='Check data preparation status')
    prep_parser.add_argument('--all', action='store_true', 
                            help='Perform all data preparation steps')
    prep_parser.add_argument('--merge', choices=['train', 'test', 'val', 'all'], 
                            help='Merge datasets')
    prep_parser.add_argument('--dataset-path', type=str, 
                            help='Custom path to dataset files or directory')
    prep_parser.add_argument('--merge-augmented', action='store_true', 
                            help='Merge augmented data with original data')
    prep_parser.add_argument('--no-merge-augmented', action='store_true', 
                            help='Do not merge augmented data with original data')
    prep_parser.add_argument('--cleanup', action='store_true', 
                            help='Clean up temporary files after processing')
    prep_parser.add_argument('--force-cleanup', action='store_true', 
                            help='Force cleanup without asking for confirmation')
    prep_parser.add_argument('--disable-augmentation', action='store_true',
                            help='Disable data augmentation (overrides config.use_data_aug)')
    prep_parser.add_argument('--enable-augmentation', action='store_true',
                            help='Enable data augmentation (overrides config.use_data_aug)')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', 
                                         help='Train a model')
    # 使用专用函数添加训练参数
    add_train_arguments(train_parser)
    
    # 推理命令
    infer_parser = subparsers.add_parser('predict', 
                                         help='Run inference with a trained model')
    infer_parser.add_argument('--model', type=str, required=True, 
                             help='Path to model weights file')
    infer_parser.add_argument('--input', type=str, required=True, 
                             help='Path to image folder or single image')
    infer_parser.add_argument('--output', type=str, 
                             help=f'Output JSON file path (default: {paths.prediction_file})')
    
    return parser

def prepare_data(args: argparse.Namespace) -> Dict[str, Any]:
    """运行数据准备流程
    
    参数:
        args: 命令行参数
        
    返回:
        数据准备结果字典
    """
    logger.info("Starting data preparation")
    
    # 确定要执行的操作
    extract = args.extract if hasattr(args, 'extract') else False
    process = args.process if hasattr(args, 'process') else False
    augment = args.augment if hasattr(args, 'augment') else False
    status = args.status if hasattr(args, 'status') else False
    merge = args.merge if hasattr(args, 'merge') else None
    
    # 如果指定了--all，执行所有操作
    if hasattr(args, 'all') and args.all:
        extract = process = augment = status = True
        merge = "all"
    
    # 获取其他参数
    custom_dataset_path = getattr(args, 'dataset_path', None)
    
    # 处理merge_augmented和no_merge_augmented
    merge_augmented = None
    if hasattr(args, 'merge_augmented') and args.merge_augmented:
        merge_augmented = True
        if hasattr(args, 'no_merge_augmented') and args.no_merge_augmented:
            logger.warning("Both --merge-augmented and --no-merge-augmented specified, using --merge-augmented")
    elif hasattr(args, 'no_merge_augmented') and args.no_merge_augmented:
        merge_augmented = False
    
    # 处理数据增强选项
    if hasattr(args, 'enable_augmentation') and args.enable_augmentation:
        if hasattr(args, 'disable_augmentation') and args.disable_augmentation:
            logger.warning("Both --enable-augmentation and --disable-augmentation specified, using --enable-augmentation")
        config.use_data_aug = True
        logger.info("Data augmentation enabled for training")
    elif hasattr(args, 'disable_augmentation') and args.disable_augmentation:
        config.use_data_aug = False
        logger.info("Data augmentation explicitly disabled for training")
    else:
        logger.info("Data augmentation enabled for training")
    
    # 获取清理临时文件的参数
    cleanup_temp = getattr(args, 'cleanup', False)
    force_cleanup = getattr(args, 'force_cleanup', False)
    
    # 运行数据准备
    result = setup_data(
        extract=extract,
        process=process, 
        augment=augment,
        status=status,
        merge=merge,
        cleanup_temp=cleanup_temp,
        custom_dataset_path=custom_dataset_path,
        merge_augmented=merge_augmented,
        force_cleanup=force_cleanup
    )
    
    return result

def train_pipeline(args: argparse.Namespace) -> None:
    """训练模型流程
    
    参数:
        args: 命令行参数
    """
    start_time = time.time()
    
    # 配置日志
    logger.info("Starting training pipeline")
    
    # 首先检查是否需要数据准备
    if args.prepare:
        logger.info("Running data preparation before training")
        # Create prepare data arguments
        prepare_args = argparse.Namespace(
            extract=True,
            process=True,
            augment=True,
            status=True,
            merge="all",
            dataset_path=getattr(args, 'dataset_path', None),
            merge_augmented=getattr(args, 'merge_augmented', None),
            no_merge_augmented=getattr(args, 'no_merge_augmented', None),
            cleanup=True  # 启用清理临时文件
        )
        prepare_data(prepare_args)
    else:
        # 检查测试数据是否存在，不存在时仅处理测试数据
        test_images_path = normalize_path(paths.test_images_dir)
        if not os.path.exists(test_images_path) or len(glob.glob(os.path.join(test_images_path, "*.*"))) == 0:
            logger.warning(f"Test image directory does not exist or is empty: {test_images_path}")
            logger.info("Preparing test data only")
            
            # 初始化DataPreparation对象并仅提取和处理测试数据
            from dataset.data_prep import DataPreparation
            data_prep = DataPreparation()
            
            # 设置测试集合并标志
            config.merge_test_datasets = True
            logger.info("Test dataset merging enabled")
            
            # 获取data目录下所有包含test的zip文件
            data_dir = normalize_path(paths.data_dir)
            test_files = []
            for ext in config.supported_dataset_formats:
                test_files.extend(glob.glob(os.path.join(data_dir, f"*test*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, f"*TEST*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, "**", f"*test*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, "**", f"*TEST*{ext}")))
            
            if test_files:
                logger.info(f"Found the following test dataset files: {test_files}")
                
                # 确保测试解压目录存在
                extract_to = normalize_path(os.path.join(paths.temp_dataset_dir, "AgriculturalDisease_testset"))
                os.makedirs(extract_to, exist_ok=True)
                
                # 解压所有测试数据集文件
                for test_file in test_files:
                    logger.info(f"Extracting test dataset file: {test_file}")
                    if data_prep.extract_zip_file(test_file, extract_to):
                        logger.info(f"Successfully extracted {test_file}")
                    else:
                        logger.error(f"Failed to extract {test_file}")
                
                # 确保测试目录存在
                os.makedirs(paths.test_images_dir, exist_ok=True)
                
                # 查找所有可能的测试图像目录
                potential_image_dirs = []
                potential_image_dirs_abs = []  # 存储绝对路径，用于检查
                
                # 1. 直接检查标准images目录
                standard_images_dir = normalize_path(os.path.join(extract_to, "images"))
                if os.path.exists(standard_images_dir) and os.path.isdir(standard_images_dir):
                    potential_image_dirs.append(standard_images_dir)
                    potential_image_dirs_abs.append(os.path.abspath(standard_images_dir))
                
                # 2. 检查可能的测试A/B子目录中的images
                test_subdirs = glob.glob(os.path.join(extract_to, "AgriculturalDisease_test*"))
                for subdir in test_subdirs:
                    subdir_images = os.path.join(subdir, "images")
                    if os.path.exists(subdir_images) and os.path.isdir(subdir_images):
                        potential_image_dirs.append(subdir_images)
                        potential_image_dirs_abs.append(os.path.abspath(subdir_images))
                
                # 3. 递归搜索其他可能包含图像的目录
                for root, dirs, files in os.walk(extract_to):
                    root_abs = os.path.abspath(root)
                    # 检查当前目录是否已经在潜在图像目录列表中或是其子目录
                    already_included = False
                    for dir_abs in potential_image_dirs_abs:
                        if root_abs == dir_abs or root_abs.startswith(dir_abs + os.sep):
                            already_included = True
                            break
                    
                    if not already_included:
                        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic'))]
                        if image_files and os.path.basename(root) != "labels":
                            potential_image_dirs.append(root)
                            potential_image_dirs_abs.append(root_abs)
                
                if potential_image_dirs:
                    logger.info(f"Found the following test image directories: {potential_image_dirs}")
                    
                    # 复制所有测试图像到测试目录
                    copied_count = 0
                    for image_dir in potential_image_dirs:
                        count = data_prep.copy_files_to_folder(
                            image_dir, 
                            paths.test_images_dir, 
                            file_pattern="*.jpg"
                        )
                        count += data_prep.copy_files_to_folder(
                            image_dir, 
                            paths.test_images_dir, 
                            file_pattern="*.png"
                        )
                        copied_count += count
                        logger.info(f"Copied {count} image files from {image_dir} to {paths.test_images_dir}")
                    
                    if copied_count > 0:
                        logger.info(f"Successfully prepared test data: copied {copied_count} image files in total")
                        logger.info("Cleaning up temporary test data files...")
                        data_prep.cleanup_temp_files(force=True)
                    else:
                        logger.warning(f"Could not find any test image files")
                else:
                    logger.error("No test image directory found in the extracted directory")
            else:
                logger.error("Could not find any test dataset files")
    
    # 根据命令行参数覆盖配置
    # Basic training parameters
    if hasattr(args, 'epochs') and args.epochs:
        config.epoch = args.epochs
        logger.info(f"Setting epochs to {config.epoch}")
    
    if hasattr(args, 'model') and args.model:
        config.model_name = args.model
        logger.info(f"Setting model to {config.model_name}")
    
    if hasattr(args, 'batch_size') and args.batch_size:
        config.train_batch_size = args.batch_size
        logger.info(f"Setting batch size to {config.train_batch_size}")
        
    if hasattr(args, 'lr') and args.lr:
        config.lr = args.lr
        logger.info(f"Setting learning rate to {config.lr}")
        
    # 设置数据集路径
    if hasattr(args, 'dataset_path') and args.dataset_path:
        config.dataset_path = args.dataset_path
        config.use_custom_dataset_path = True
        logger.info(f"Using custom dataset path: {config.dataset_path}")
    
    # 配置是否合并增强数据
    if hasattr(args, 'merge_augmented') and hasattr(args, 'no_merge_augmented'):
        if args.merge_augmented and args.no_merge_augmented:
            logger.warning("Both --merge-augmented and --no-merge-augmented specified, using --merge-augmented")
            config.merge_augmented_data = True
        elif args.merge_augmented:
            config.merge_augmented_data = True
            logger.info("Will merge augmented data with original training data")
        elif args.no_merge_augmented:
            config.merge_augmented_data = False
            logger.info("Will not merge augmented data with original training data")
    
    # Optimizer options
    if hasattr(args, 'optimizer') and args.optimizer:
        config.optimizer = args.optimizer
        logger.info(f"Setting optimizer to {config.optimizer}")
    
    if hasattr(args, 'weight_decay') and args.weight_decay:
        config.weight_decay = args.weight_decay
        logger.info(f"Setting weight decay to {config.weight_decay}")
    
    if hasattr(args, 'no_lookahead') and args.no_lookahead:
        config.use_lookahead = False
        logger.info("Disabling Lookahead optimizer wrapper")
    
    # Learning rate scheduler options
    if hasattr(args, 'scheduler') and args.scheduler:
        config.scheduler = args.scheduler
        logger.info(f"Setting LR scheduler to {config.scheduler}")
    
    if hasattr(args, 'warmup_epochs') and args.warmup_epochs:
        config.warmup_epochs = args.warmup_epochs
        logger.info(f"Setting warmup epochs to {config.warmup_epochs}")
    
    if hasattr(args, 'warmup_factor') and args.warmup_factor:
        config.warmup_factor = args.warmup_factor
        logger.info(f"Setting warmup factor to {config.warmup_factor}")
    
    # Mixup and CutMix options
    if hasattr(args, 'no_mixup') and args.no_mixup:
        config.use_mixup = False
        logger.info("Disabling Mixup data augmentation")
    
    if hasattr(args, 'mixup_alpha') and args.mixup_alpha:
        config.mixup_alpha = args.mixup_alpha
        logger.info(f"Setting Mixup alpha to {config.mixup_alpha}")
    
    if hasattr(args, 'cutmix_prob') and args.cutmix_prob:
        config.cutmix_prob = args.cutmix_prob
        logger.info(f"Setting CutMix probability to {config.cutmix_prob}")
    
    if hasattr(args, 'no_random_erasing') and args.no_random_erasing:
        config.use_random_erasing = False
        logger.info("Disabling random erasing augmentation")
    
    # Early stopping parameters
    if hasattr(args, 'no_early_stopping') and args.no_early_stopping:
        config.use_early_stopping = False
        logger.info("Disabling early stopping")
    
    if hasattr(args, 'patience') and args.patience:
        config.early_stopping_patience = args.patience
        logger.info(f"Setting early stopping patience to {config.early_stopping_patience}")
    
    # Mixed precision options
    if hasattr(args, 'no_amp') and args.no_amp:
        config.use_amp = False
        logger.info("Disabling automatic mixed precision training")
    
    # Gradient clipping options
    if hasattr(args, 'gradient_clip_val') and args.gradient_clip_val:
        config.gradient_clip_val = args.gradient_clip_val
        logger.info(f"Setting gradient clipping value to {config.gradient_clip_val}")
    
    # EMA options
    if hasattr(args, 'no_ema') and args.no_ema:
        config.use_ema = False
        logger.info("Disabling Exponential Moving Average (EMA)")
    
    if hasattr(args, 'ema_decay') and args.ema_decay:
        config.ema_decay = args.ema_decay
        logger.info(f"Setting EMA decay rate to {config.ema_decay}")
    
    # Device options
    if hasattr(args, 'device') and args.device:
        config.device = args.device
        logger.info(f"Setting device to {config.device}")
    
    if hasattr(args, 'gpus') and args.gpus:
        config.gpus = args.gpus
        logger.info(f"Setting GPUs to {config.gpus}")
    
    # Advanced options
    if hasattr(args, 'seed') and args.seed:
        config.seed = args.seed
        logger.info(f"Setting random seed to {config.seed}")
    
    if hasattr(args, 'label_smoothing') and args.label_smoothing:
        config.label_smoothing = args.label_smoothing
        logger.info(f"Setting label smoothing to {config.label_smoothing}")
    
    if hasattr(args, 'no_gradient_checkpointing') and args.no_gradient_checkpointing:
        config.use_gradient_checkpointing = False
        logger.info("Disabling gradient checkpointing")
        
    # 训练模型
    from libs.training import train_model
    train_model(config)
    
    # 训练完成后，如果参数中指定了清理临时文件，则执行清理
    if getattr(args, 'cleanup', False):
        logger.info("Training completed, cleaning up temporary files...")
        from dataset.data_prep import DataPreparation
        data_prep = DataPreparation()
        # 使用force_cleanup参数决定是否强制清理
        force_cleanup = getattr(args, 'force_cleanup', False)
        data_prep.cleanup_temp_files(force=force_cleanup)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Training pipeline completed in {elapsed_time:.2f} seconds")

def run_inference(args) -> None:
    """执行模型推理
    
    参数:
        args: 命令行参数
    """
    logger.info("Starting inference")
    
    # 验证模型文件是否存在
    model_path = args.model
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
        
    # 获取输入路径，可以是单个图像或目录
    input_path = args.input
    if not os.path.exists(input_path):
        logger.error(f"Input path not found: {input_path}")
        
        # 检查是否是测试目录
        if input_path == paths.test_images_dir or input_path == paths.test_dir:
            logger.info("Test data preparation needed")
            
            # 直接使用DataPreparation类来处理测试数据
            from dataset.data_prep import DataPreparation
            data_prep = DataPreparation()
            
            # 设置测试集合并标志
            config.merge_test_datasets = True
            logger.info("Test dataset merging enabled")
            
            # 获取data目录下所有包含test的zip文件
            data_dir = normalize_path(paths.data_dir)
            test_files = []
            for ext in config.supported_dataset_formats:
                test_files.extend(glob.glob(os.path.join(data_dir, f"*test*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, f"*TEST*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, "**", f"*test*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, "**", f"*TEST*{ext}")))
            
            if test_files:
                logger.info(f"Found the following test dataset files: {test_files}")
                
                # 确保测试解压目录存在
                extract_to = normalize_path(os.path.join(paths.temp_dataset_dir, "AgriculturalDisease_testset"))
                os.makedirs(extract_to, exist_ok=True)
                
                # 解压所有测试数据集文件
                for test_file in test_files:
                    logger.info(f"Extracting test dataset file: {test_file}")
                    if data_prep.extract_zip_file(test_file, extract_to):
                        logger.info(f"Successfully extracted {test_file}")
                    else:
                        logger.error(f"Failed to extract {test_file}")
                
                # 确保测试目录存在
                os.makedirs(paths.test_images_dir, exist_ok=True)
                
                # 查找所有可能的测试图像目录
                potential_image_dirs = []
                potential_image_dirs_abs = []  # 存储绝对路径，用于检查
                
                # 1. 直接检查标准images目录
                standard_images_dir = normalize_path(os.path.join(extract_to, "images"))
                if os.path.exists(standard_images_dir) and os.path.isdir(standard_images_dir):
                    potential_image_dirs.append(standard_images_dir)
                    potential_image_dirs_abs.append(os.path.abspath(standard_images_dir))
                
                # 2. 检查可能的测试A/B子目录中的images
                test_subdirs = glob.glob(os.path.join(extract_to, "AgriculturalDisease_test*"))
                for subdir in test_subdirs:
                    subdir_images = os.path.join(subdir, "images")
                    if os.path.exists(subdir_images) and os.path.isdir(subdir_images):
                        potential_image_dirs.append(subdir_images)
                        potential_image_dirs_abs.append(os.path.abspath(subdir_images))
                
                # 3. 递归搜索其他可能包含图像的目录
                for root, dirs, files in os.walk(extract_to):
                    root_abs = os.path.abspath(root)
                    # 检查当前目录是否已经在潜在图像目录列表中或是其子目录
                    already_included = False
                    for dir_abs in potential_image_dirs_abs:
                        if root_abs == dir_abs or root_abs.startswith(dir_abs + os.sep):
                            already_included = True
                            break
                    
                    if not already_included:
                        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if image_files and os.path.basename(root) != "labels":
                            potential_image_dirs.append(root)
                            potential_image_dirs_abs.append(root_abs)
                
                if potential_image_dirs:
                    logger.info(f"Found the following test image directories: {potential_image_dirs}")
                    
                    # 复制所有测试图像到测试目录
                    copied_count = 0
                    for image_dir in potential_image_dirs:
                        count = data_prep.copy_files_to_folder(
                            image_dir, 
                            paths.test_images_dir, 
                            file_pattern="*.jpg"
                        )
                        count += data_prep.copy_files_to_folder(
                            image_dir, 
                            paths.test_images_dir, 
                            file_pattern="*.png"
                        )
                        copied_count += count
                        logger.info(f"Copied {count} image files from {image_dir} to {paths.test_images_dir}")
                    
                    if copied_count > 0:
                        logger.info(f"Successfully prepared test data: copied {copied_count} image files in total")
                        logger.info("Cleaning up temporary test data files...")
                        data_prep.cleanup_temp_files(force=True)
                    else:
                        logger.warning(f"Could not find any test image files")
                else:
                    logger.error("No test image directory found in the extracted directory")
            else:
                # 尝试使用标准方法查找测试数据集
                logger.info("Trying to find test dataset files with specific names")
                test_file = data_prep.find_dataset_file(
                    config.test_name_pattern,
                    getattr(args, 'dataset_path', None) if config.use_custom_dataset_path else None
                )
                
                if test_file:
                    logger.info(f"Found test dataset file: {test_file}")
                    extract_to = normalize_path(os.path.join(paths.temp_dataset_dir, "AgriculturalDisease_testset"))
                    os.makedirs(extract_to, exist_ok=True)
                    
                    if data_prep.extract_zip_file(test_file, extract_to):
                        # 确保测试目录存在
                        os.makedirs(paths.test_images_dir, exist_ok=True)
                        
                        # 查找所有可能的测试图像目录
                        potential_image_dirs = []
                        potential_image_dirs_abs = []  # 存储绝对路径，用于检查
                        
                        # 1. 检查多种可能的图像目录路径
                        possible_paths = [
                            normalize_path(os.path.join(extract_to, "images")),
                            normalize_path(os.path.join(extract_to, "AgriculturalDisease_testA", "images")),
                            normalize_path(os.path.join(extract_to, "AgriculturalDisease_testB", "images"))
                        ]
                        
                        for path in possible_paths:
                            if os.path.exists(path) and os.path.isdir(path):
                                potential_image_dirs.append(path)
                                potential_image_dirs_abs.append(os.path.abspath(path))
                        
                        # 2. 递归搜索其他可能包含图像的目录
                        for root, dirs, files in os.walk(extract_to):
                            root_abs = os.path.abspath(root)
                            # 检查当前目录是否已经在潜在图像目录列表中或是其子目录
                            already_included = False
                            for dir_abs in potential_image_dirs_abs:
                                if root_abs == dir_abs or root_abs.startswith(dir_abs + os.sep):
                                    already_included = True
                                    break
                            
                            if not already_included:
                                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                                if image_files and os.path.basename(root) != "labels":
                                    potential_image_dirs.append(root)
                                    potential_image_dirs_abs.append(root_abs)
                        
                        if potential_image_dirs:
                            logger.info(f"Found the following test image directories: {potential_image_dirs}")
                            
                            # 复制所有测试图像到测试目录
                            copied_count = 0
                            for image_dir in potential_image_dirs:
                                count = data_prep.copy_files_to_folder(
                                    image_dir, 
                                    paths.test_images_dir, 
                                    file_pattern="*.jpg"
                                )
                                count += data_prep.copy_files_to_folder(
                                    image_dir, 
                                    paths.test_images_dir, 
                                    file_pattern="*.png"
                                )
                                copied_count += count
                                logger.info(f"Copied {count} image files from {image_dir} to {paths.test_images_dir}")
                            
                            if copied_count > 0:
                                logger.info(f"Successfully prepared test data: copied {copied_count} image files in total")
                                logger.info("Cleaning up temporary test data files...")
                                data_prep.cleanup_temp_files(force=True)
                            else:
                                logger.warning(f"Could not find any test image files")
                        else:
                            logger.error(f"No test image directory found in the extracted directory")
                    else:
                        logger.error(f"Unable to extract test dataset file: {test_file}")
                        return
                else:
                    logger.error("Could not find any test dataset files")
                    return
            
            # 再次检查输入路径
            if not os.path.exists(input_path):
                logger.error(f"Even after attempting to prepare test data, input path still not found: {input_path}")
                return
            else:
                logger.info(f"Successfully prepared test data: {input_path}")
        else:
            return
    
    # 设置输出文件路径
    output_path = args.output if args.output else paths.prediction_file
    
    # 检查输入是文件还是目录
    if os.path.isfile(input_path):
        logger.info(f"Running inference on single image: {input_path}")
        results = predict(model_path, input_path, output_path)
    else:
        logger.info(f"Running inference on directory: {input_path}")
        results = predict(model_path, input_path, output_path, is_dir=True)
    
    # 保存预测结果
    if results:
        logger.info(f"Inference completed successfully, results saved to: {output_path}")
    else:
        logger.error("Inference failed or no results generated")

def main():
    """主函数"""
    args = setup_parser().parse_args()
    
    # 处理未指定命令的情况
    if args.command is None:
        logger.info("No command specified. Running all operations in sequence.")
        
        # 1. 数据准备
        logger.info("Step 1: Data preparation")
        
        # 检查现有数据
        from dataset.data_prep import DataPreparation
        data_prep = DataPreparation()
        # 检查是否需要准备数据
        need_data_preparation = True
        
        # 检查训练数据是否已存在
        train_path = normalize_path(paths.train_dir)
        if os.path.exists(train_path):
            train_files = sum(len(glob.glob(os.path.join(d, "*.*"))) 
                           for d in glob.glob(os.path.join(train_path, "*")) if os.path.isdir(d))
            if train_files > config.min_files_threshold:
                logger.info(f"Training data already exists ({train_files} files), skipping training data preparation")
                need_data_preparation = False
        
        # 检查测试数据是否已存在
        test_path = normalize_path(paths.test_images_dir)
        need_test_data = True
        if os.path.exists(test_path) and len(glob.glob(os.path.join(test_path, "*.*"))) > 0:
            need_test_data = False
            logger.info(f"Test data already exists, skipping test data preparation")
        
        # 根据检查结果运行不同的数据准备步骤
        if need_data_preparation:
            logger.info("Preparing full training and test data")
            prepare_args = argparse.Namespace(
                extract=True,
                process=True,
                augment=True,
                status=True,
                merge="all",
                cleanup=True,  # 启用清理临时文件
                dataset_path=None,
                merge_augmented=True,
                no_merge_augmented=False
            )
            prepare_data(prepare_args)
        elif need_test_data:
            logger.info("Preparing test data only")
            
            # 设置测试集合并标志
            config.merge_test_datasets = True
            logger.info("Test dataset merging enabled")
            
            # 获取data目录下所有包含test的zip文件
            data_dir = normalize_path(paths.data_dir)
            test_files = []
            for ext in config.supported_dataset_formats:
                test_files.extend(glob.glob(os.path.join(data_dir, f"*test*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, f"*TEST*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, "**", f"*test*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, "**", f"*TEST*{ext}")))
            
            if test_files:
                logger.info(f"Found the following test dataset files: {test_files}")
                
                # 确保测试解压目录存在
                extract_to = normalize_path(os.path.join(paths.temp_dataset_dir, "AgriculturalDisease_testset"))
                os.makedirs(extract_to, exist_ok=True)
                
                # 解压所有测试数据集文件
                for test_file in test_files:
                    logger.info(f"Extracting test dataset file: {test_file}")
                    if data_prep.extract_zip_file(test_file, extract_to):
                        logger.info(f"Successfully extracted {test_file}")
                    else:
                        logger.error(f"Failed to extract {test_file}")
                
                # 确保测试目录存在
                os.makedirs(paths.test_images_dir, exist_ok=True)
                
                # 查找所有可能的测试图像目录
                potential_image_dirs = []
                potential_image_dirs_abs = []  # 存储绝对路径，用于检查
                
                # 1. 直接检查标准images目录
                standard_images_dir = normalize_path(os.path.join(extract_to, "images"))
                if os.path.exists(standard_images_dir) and os.path.isdir(standard_images_dir):
                    potential_image_dirs.append(standard_images_dir)
                    potential_image_dirs_abs.append(os.path.abspath(standard_images_dir))
                
                # 2. 检查可能的测试A/B子目录中的images
                test_subdirs = glob.glob(os.path.join(extract_to, "AgriculturalDisease_test*"))
                for subdir in test_subdirs:
                    subdir_images = os.path.join(subdir, "images")
                    if os.path.exists(subdir_images) and os.path.isdir(subdir_images):
                        potential_image_dirs.append(subdir_images)
                        potential_image_dirs_abs.append(os.path.abspath(subdir_images))
                
                # 3. 递归搜索其他可能包含图像的目录
                for root, dirs, files in os.walk(extract_to):
                    root_abs = os.path.abspath(root)
                    # 检查当前目录是否已经在潜在图像目录列表中或是其子目录
                    already_included = False
                    for dir_abs in potential_image_dirs_abs:
                        if root_abs == dir_abs or root_abs.startswith(dir_abs + os.sep):
                            already_included = True
                            break
                    
                    if not already_included:
                        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if image_files and os.path.basename(root) != "labels":
                            potential_image_dirs.append(root)
                            potential_image_dirs_abs.append(root_abs)
                
                if potential_image_dirs:
                    logger.info(f"Found the following test image directories: {potential_image_dirs}")
                    
                    # 复制所有测试图像到测试目录
                    copied_count = 0
                    for image_dir in potential_image_dirs:
                        count = data_prep.copy_files_to_folder(
                            image_dir, 
                            paths.test_images_dir, 
                            file_pattern="*.jpg"
                        )
                        count += data_prep.copy_files_to_folder(
                            image_dir, 
                            paths.test_images_dir, 
                            file_pattern="*.png"
                        )
                        copied_count += count
                        logger.info(f"Copied {count} image files from {image_dir} to {paths.test_images_dir}")
                    
                    if copied_count > 0:
                        logger.info(f"Successfully prepared test data: copied {copied_count} image files in total")
                        logger.info("Cleaning up temporary test data files...")
                        data_prep.cleanup_temp_files(force=True)
                    else:
                        logger.warning(f"Could not find any test image files")
                else:
                    logger.error("No test image directory found in the extracted directory")
            else:
                logger.error("Could not find any test dataset files")
        else:
            logger.info("All data already exists, skipping data preparation")
        
        # 2. 训练模型
        logger.info("Step 2: Model training")
        
        # 创建训练模型的参数对象
        train_args = argparse.Namespace(
            epochs=None,                    # 使用配置文件中的默认轮次
            model=None,                     # 使用配置文件中的默认模型
            batch_size=None,
            lr=None,
            prepare=False,                  # 我们已经在步骤1中准备了数据
            no_prepare=True,
            dataset_path=None,
            merge_augmented=False,
            no_merge_augmented=False,
            optimizer=None,
            weight_decay=None,
            no_lookahead=False,
            scheduler=None,
            warmup_epochs=None,
            warmup_factor=None,
            no_mixup=False,
            mixup_alpha=None,
            cutmix_prob=None,
            no_random_erasing=False,
            no_early_stopping=False,
            patience=None,
            no_amp=False,
            gradient_clip_val=None,
            no_ema=False,
            ema_decay=None,
            device=None,
            gpus=None,
            seed=None,
            label_smoothing=None,
            no_gradient_checkpointing=False,
            cleanup=True,                   # 在训练后清理临时文件
            force_cleanup=True,             # 强制执行清理，不询问用户
            force_train=False               # 不强制重新训练
        )
        
        # 运行训练流程
        train_pipeline(train_args)
        
        # 3. 模型推理
        logger.info("Step 3: Model inference")
        # 查找最佳模型文件
        best_model_path = os.path.join(config.best_weights, config.model_name, "0", "best_model.pth.tar")
        if not os.path.exists(best_model_path):
            best_model_path = os.path.join(config.weights, config.model_name, "0", "_latest_model.pth.tar")
        
        predict_args = argparse.Namespace(
            model=best_model_path,
            input=paths.test_images_dir,
            output=paths.prediction_file,
            merge=True  # 确保测试集合并设置正确
        )
        run_inference(predict_args)
        
        logger.info("All operations completed successfully.")
        return
    
    logger.info(f"Running command: {args.command}")
    
    if args.command == "prepare":
        prepare_data(args)
    elif args.command == "train":
        train_pipeline(args)
    elif args.command == "predict":
        # 确保测试集合并设置正确
        if not hasattr(args, 'merge') or not args.merge:
            # 默认启用测试集合并
            logger.info("Test dataset merging enabled")
            config.merge_test_datasets = True
        run_inference(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main() 