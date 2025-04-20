#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import time
import glob
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
from datetime import datetime

# Add the project root to the path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

# Import configuration
from config.config import config, paths
from dataset.data_prep import setup_data, normalize_path
from libs.inference import predict

# Set up logging
current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
log_file = f"logs/main_{current_time}.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Main')

def add_train_arguments(train_parser: argparse.ArgumentParser) -> None:
    """Add training-related arguments to the parser
    
    Args:
        train_parser: The argument parser for the training command
    """
    # Basic training parameters
    train_parser.add_argument('--epochs', type=int, 
                             help='Number of epochs (default: from config)')
    train_parser.add_argument('--model', type=str, 
                             help='Model name (default: from config)')
    train_parser.add_argument('--batch-size', type=int, 
                             help='Batch size (default: from config)')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    
    # Data preparation flags
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
    
    # Optimizer options
    train_parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd', 'ranger'],
                             help='Optimizer selection (default: from config)')
    train_parser.add_argument('--weight-decay', type=float,
                             help='Weight decay for optimizer (default: from config)')
    train_parser.add_argument('--no-lookahead', action='store_true',
                             help='Disable Lookahead optimizer wrapper')
    
    # Learning rate scheduler options
    train_parser.add_argument('--scheduler', type=str, choices=['step', 'cosine', 'onecycle'],
                             help='Learning rate scheduler (default: from config)')
    train_parser.add_argument('--warmup-epochs', type=int,
                             help='Number of warmup epochs (default: from config)')
    train_parser.add_argument('--warmup-factor', type=float,
                             help='Warmup factor (default: from config)')
    
    # Mixup and CutMix options
    train_parser.add_argument('--no-mixup', action='store_true',
                             help='Disable Mixup data augmentation')
    train_parser.add_argument('--mixup-alpha', type=float,
                             help='Mixup alpha parameter (default: from config)')
    train_parser.add_argument('--cutmix-prob', type=float,
                             help='CutMix probability (default: from config)')
    train_parser.add_argument('--no-random-erasing', action='store_true',
                             help='Disable random erasing augmentation')
    
    # Early stopping parameters
    train_parser.add_argument('--no-early-stopping', action='store_true',
                             help='Disable early stopping')
    train_parser.add_argument('--patience', type=int,
                             help='Early stopping patience (default: from config)')
    
    # Mixed precision options
    train_parser.add_argument('--no-amp', action='store_true',
                             help='Disable automatic mixed precision training')
    
    # Gradient clipping options
    train_parser.add_argument('--gradient-clip-val', type=float,
                             help='Gradient clipping value (default: from config)')
    
    # EMA options
    train_parser.add_argument('--no-ema', action='store_true', 
                             help='Disable Exponential Moving Average (EMA)')
    train_parser.add_argument('--ema-decay', type=float,
                             help='EMA decay rate (default: from config)')
    
    # Device options
    train_parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'],
                             help='Device to use for training (default: from config)')
    train_parser.add_argument('--gpus', type=str,
                             help='GPU device IDs to use, comma separated (e.g., "0,1")')
    
    # Advanced options
    train_parser.add_argument('--seed', type=int,
                             help='Random seed for reproducibility (default: from config)')
    train_parser.add_argument('--label-smoothing', type=float,
                             help='Label smoothing coefficient (default: from config)')
    train_parser.add_argument('--no-gradient-checkpointing', action='store_true',
                             help='Disable gradient checkpointing')

def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Plant Disease Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Data preparation command
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
    
    # Training command
    train_parser = subparsers.add_parser('train', 
                                         help='Train a model')
    # Use the dedicated function to add training arguments
    add_train_arguments(train_parser)
    
    # Inference command
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
    
    # 如果--all参数指定，则运行所有数据准备步骤
    if getattr(args, 'all', False):
        extract = process = augment = status = True
        merge = "all"
    else:
        extract = getattr(args, 'extract', False)
        process = getattr(args, 'process', False)
        augment = getattr(args, 'augment', False)
        status = getattr(args, 'status', False)
        merge = getattr(args, 'merge', None)
    
    # 获取自定义数据集路径
    custom_dataset_path = getattr(args, 'dataset_path', None)
    
    # 处理增强数据合并标志
    merge_augmented = None
    if getattr(args, 'merge_augmented', False) and getattr(args, 'no_merge_augmented', False):
        logger.warning("Both --merge-augmented and --no-merge-augmented specified, using --merge-augmented")
        merge_augmented = True
    elif getattr(args, 'merge_augmented', False):
        merge_augmented = True
    elif getattr(args, 'no_merge_augmented', False):
        merge_augmented = False
    
    # 运行数据准备
    result = setup_data(
        extract=extract,
        process=process, 
        augment=augment,
        status=status,
        merge=merge,
        cleanup_temp=getattr(args, 'clean', False),
        custom_dataset_path=custom_dataset_path,
        merge_augmented=merge_augmented
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
            clean=False
        )
        prepare_data(prepare_args)
    else:
        # 检查测试数据是否存在，不存在时仅处理测试数据
        test_images_path = normalize_path(paths.test_images_dir)
        if not os.path.exists(test_images_path) or len(glob.glob(os.path.join(test_images_path, "*.*"))) == 0:
            logger.warning(f"测试图像目录不存在或为空: {test_images_path}")
            logger.info("仅准备测试数据")
            
            # 初始化DataPreparation对象并仅提取和处理测试数据
            from dataset.data_prep import DataPreparation
            data_prep = DataPreparation()
            
            # 设置测试集合并标志
            config.merge_test_datasets = True
            logger.info("已启用测试集合并")
            
            # 获取data目录下所有包含test的zip文件
            data_dir = normalize_path(paths.data_dir)
            test_files = []
            for ext in config.supported_dataset_formats:
                test_files.extend(glob.glob(os.path.join(data_dir, f"*test*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, f"*TEST*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, "**", f"*test*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, "**", f"*TEST*{ext}")))
            
            if test_files:
                logger.info(f"找到以下测试数据集文件: {test_files}")
                
                # 确保测试解压目录存在
                extract_to = normalize_path(os.path.join(paths.temp_dataset_dir, "AgriculturalDisease_testset"))
                os.makedirs(extract_to, exist_ok=True)
                
                # 解压所有测试数据集文件
                for test_file in test_files:
                    logger.info(f"解压测试数据集文件: {test_file}")
                    if data_prep.extract_zip_file(test_file, extract_to):
                        logger.info(f"成功解压 {test_file}")
                    else:
                        logger.error(f"解压 {test_file} 失败")
                
                # 确保测试目录存在
                os.makedirs(paths.test_images_dir, exist_ok=True)
                
                # 查找所有可能的测试图像目录
                potential_image_dirs = []
                
                # 1. 直接检查标准images目录
                standard_images_dir = normalize_path(os.path.join(extract_to, "images"))
                if os.path.exists(standard_images_dir) and os.path.isdir(standard_images_dir):
                    potential_image_dirs.append(standard_images_dir)
                
                # 2. 检查可能的测试A/B子目录中的images
                test_subdirs = glob.glob(os.path.join(extract_to, "AgriculturalDisease_test*"))
                for subdir in test_subdirs:
                    subdir_images = os.path.join(subdir, "images")
                    if os.path.exists(subdir_images) and os.path.isdir(subdir_images):
                        potential_image_dirs.append(subdir_images)
                
                # 3. 递归搜索其他可能包含图像的目录
                for root, dirs, files in os.walk(extract_to):
                    if not any(root.startswith(d) for d in potential_image_dirs):
                        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if image_files and os.path.basename(root) != "labels":
                            potential_image_dirs.append(root)
                
                if potential_image_dirs:
                    logger.info(f"找到以下测试图像目录: {potential_image_dirs}")
                    
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
                        logger.info(f"从 {image_dir} 复制了 {count} 个图像文件到 {paths.test_images_dir}")
                    
                    if copied_count > 0:
                        logger.info(f"成功准备测试数据: 总共复制了 {copied_count} 个图像文件")
                    else:
                        logger.warning(f"未能找到任何测试图像文件")
                else:
                    logger.error("在解压后的目录中未找到测试图像目录")
            else:
                logger.error("找不到任何测试数据集文件")
    
    # 根据命令行参数覆盖配置
    # Basic training parameters
    if args.epochs:
        config.epoch = args.epochs
        logger.info(f"Setting epochs to {config.epoch}")
    
    if args.model:
        config.model_name = args.model
        logger.info(f"Setting model to {config.model_name}")
    
    if args.batch_size:
        config.train_batch_size = args.batch_size
        logger.info(f"Setting batch size to {config.train_batch_size}")
        
    if args.lr:
        config.lr = args.lr
        logger.info(f"Setting learning rate to {config.lr}")
        
    # 设置数据集路径
    if args.dataset_path:
        config.dataset_path = args.dataset_path
        config.use_custom_dataset_path = True
        logger.info(f"Using custom dataset path: {config.dataset_path}")
    
    # 配置是否合并增强数据
    if getattr(args, 'merge_augmented', False) and getattr(args, 'no_merge_augmented', False):
        logger.warning("Both --merge-augmented and --no-merge-augmented specified, using --merge-augmented")
        config.merge_augmented_data = True
    elif getattr(args, 'merge_augmented', False):
        config.merge_augmented_data = True
        logger.info("Will merge augmented data with original training data")
    elif getattr(args, 'no_merge_augmented', False):
        config.merge_augmented_data = False
        logger.info("Will not merge augmented data with original training data")
    
    # Optimizer options
    if args.optimizer:
        config.optimizer = args.optimizer
        logger.info(f"Setting optimizer to {config.optimizer}")
    
    if args.weight_decay:
        config.weight_decay = args.weight_decay
        logger.info(f"Setting weight decay to {config.weight_decay}")
    
    if args.no_lookahead:
        config.use_lookahead = False
        logger.info("Disabling Lookahead optimizer wrapper")
    
    # Learning rate scheduler options
    if args.scheduler:
        config.scheduler = args.scheduler
        logger.info(f"Setting LR scheduler to {config.scheduler}")
    
    if args.warmup_epochs:
        config.warmup_epochs = args.warmup_epochs
        logger.info(f"Setting warmup epochs to {config.warmup_epochs}")
    
    if args.warmup_factor:
        config.warmup_factor = args.warmup_factor
        logger.info(f"Setting warmup factor to {config.warmup_factor}")
    
    # Mixup and CutMix options
    if args.no_mixup:
        config.use_mixup = False
        logger.info("Disabling Mixup data augmentation")
    
    if args.mixup_alpha:
        config.mixup_alpha = args.mixup_alpha
        logger.info(f"Setting Mixup alpha to {config.mixup_alpha}")
    
    if args.cutmix_prob:
        config.cutmix_prob = args.cutmix_prob
        logger.info(f"Setting CutMix probability to {config.cutmix_prob}")
    
    if args.no_random_erasing:
        config.use_random_erasing = False
        logger.info("Disabling random erasing augmentation")
    
    # Early stopping parameters
    if args.no_early_stopping:
        config.use_early_stopping = False
        logger.info("Disabling early stopping")
    
    if args.patience:
        config.early_stopping_patience = args.patience
        logger.info(f"Setting early stopping patience to {config.early_stopping_patience}")
    
    # Mixed precision options
    if args.no_amp:
        config.use_amp = False
        logger.info("Disabling automatic mixed precision training")
    
    # Gradient clipping options
    if args.gradient_clip_val:
        config.gradient_clip_val = args.gradient_clip_val
        logger.info(f"Setting gradient clipping value to {config.gradient_clip_val}")
    
    # EMA options
    if args.no_ema:
        config.use_ema = False
        logger.info("Disabling Exponential Moving Average (EMA)")
    
    if args.ema_decay:
        config.ema_decay = args.ema_decay
        logger.info(f"Setting EMA decay rate to {config.ema_decay}")
    
    # Device options
    if args.device:
        config.device = args.device
        logger.info(f"Setting device to {config.device}")
    
    if args.gpus:
        config.gpus = args.gpus
        logger.info(f"Setting GPUs to {config.gpus}")
    
    # Advanced options
    if args.seed:
        config.seed = args.seed
        logger.info(f"Setting random seed to {config.seed}")
    
    if args.label_smoothing:
        config.label_smoothing = args.label_smoothing
        logger.info(f"Setting label smoothing to {config.label_smoothing}")
    
    if args.no_gradient_checkpointing:
        config.use_gradient_checkpointing = False
        logger.info("Disabling gradient checkpointing")
        
    # 训练模型
    from libs.training import train_model
    train_model(config)
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f} seconds")

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
            logger.info("需要准备测试数据")
            
            # 直接使用DataPreparation类来处理测试数据
            from dataset.data_prep import DataPreparation
            data_prep = DataPreparation()
            
            # 设置测试集合并标志
            config.merge_test_datasets = True
            logger.info("已启用测试集合并")
            
            # 获取data目录下所有包含test的zip文件
            data_dir = normalize_path(paths.data_dir)
            test_files = []
            for ext in config.supported_dataset_formats:
                test_files.extend(glob.glob(os.path.join(data_dir, f"*test*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, f"*TEST*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, "**", f"*test*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, "**", f"*TEST*{ext}")))
            
            if test_files:
                logger.info(f"找到以下测试数据集文件: {test_files}")
                
                # 确保测试解压目录存在
                extract_to = normalize_path(os.path.join(paths.temp_dataset_dir, "AgriculturalDisease_testset"))
                os.makedirs(extract_to, exist_ok=True)
                
                # 解压所有测试数据集文件
                for test_file in test_files:
                    logger.info(f"解压测试数据集文件: {test_file}")
                    if data_prep.extract_zip_file(test_file, extract_to):
                        logger.info(f"成功解压 {test_file}")
                    else:
                        logger.error(f"解压 {test_file} 失败")
                
                # 确保测试目录存在
                os.makedirs(paths.test_images_dir, exist_ok=True)
                
                # 查找所有可能的测试图像目录
                potential_image_dirs = []
                
                # 1. 直接检查标准images目录
                standard_images_dir = normalize_path(os.path.join(extract_to, "images"))
                if os.path.exists(standard_images_dir) and os.path.isdir(standard_images_dir):
                    potential_image_dirs.append(standard_images_dir)
                
                # 2. 检查可能的测试A/B子目录中的images
                test_subdirs = glob.glob(os.path.join(extract_to, "AgriculturalDisease_test*"))
                for subdir in test_subdirs:
                    subdir_images = os.path.join(subdir, "images")
                    if os.path.exists(subdir_images) and os.path.isdir(subdir_images):
                        potential_image_dirs.append(subdir_images)
                
                # 3. 递归搜索其他可能包含图像的目录
                for root, dirs, files in os.walk(extract_to):
                    if not any(root.startswith(d) for d in potential_image_dirs):
                        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if image_files and os.path.basename(root) != "labels":
                            potential_image_dirs.append(root)
                
                if potential_image_dirs:
                    logger.info(f"找到以下测试图像目录: {potential_image_dirs}")
                    
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
                        logger.info(f"从 {image_dir} 复制了 {count} 个图像文件到 {paths.test_images_dir}")
                    
                    if copied_count > 0:
                        logger.info(f"成功准备测试数据: 总共复制了 {copied_count} 个图像文件")
                    else:
                        logger.warning(f"未能找到任何测试图像文件")
                else:
                    logger.error("在解压后的目录中未找到测试图像目录")
            else:
                # 尝试使用标准方法查找测试数据集
                logger.info("尝试查找特定名称的测试数据集文件")
                test_file = data_prep.find_dataset_file(
                    config.test_name_pattern,
                    getattr(args, 'dataset_path', None) if config.use_custom_dataset_path else None
                )
                
                if test_file:
                    logger.info(f"找到测试数据集文件: {test_file}")
                    extract_to = normalize_path(os.path.join(paths.temp_dataset_dir, "AgriculturalDisease_testset"))
                    os.makedirs(extract_to, exist_ok=True)
                    
                    if data_prep.extract_zip_file(test_file, extract_to):
                        # 确保测试目录存在
                        os.makedirs(paths.test_images_dir, exist_ok=True)
                        
                        # 尝试查找多种可能的图像目录路径
                        potential_image_dirs = [
                            normalize_path(os.path.join(extract_to, "images")),
                            normalize_path(os.path.join(extract_to, "AgriculturalDisease_testA", "images")),
                            normalize_path(os.path.join(extract_to, "AgriculturalDisease_testB", "images"))
                        ]
                        
                        image_dir_found = False
                        for img_dir in potential_image_dirs:
                            if os.path.exists(img_dir) and os.path.isdir(img_dir):
                                logger.info(f"找到测试图像目录: {img_dir}")
                                data_prep.copy_files_to_folder(img_dir, paths.test_images_dir)
                                image_dir_found = True
                                break
                        
                        if not image_dir_found:
                            logger.warning(f"在解压后的目录中未找到测试图像目录，尝试递归搜索")
                            # 递归搜索包含图像的目录
                            for root, dirs, files in os.walk(extract_to):
                                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                                if image_files:
                                    logger.info(f"找到包含图像的目录: {root}")
                                    data_prep.copy_files_to_folder(root, paths.test_images_dir)
                                    image_dir_found = True
                                    break
                            
                            if not image_dir_found:
                                logger.error(f"在解压后的目录中未找到任何测试图像")
                    else:
                        logger.error(f"无法解压测试数据集文件: {test_file}")
                        return
                else:
                    logger.error("找不到任何测试数据集文件")
                    return
            
            # 再次检查输入路径
            if not os.path.exists(input_path):
                logger.error(f"即使尝试准备测试数据后，仍无法找到输入路径: {input_path}")
                return
            else:
                logger.info(f"成功准备测试数据: {input_path}")
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
        data_status = data_prep.get_data_status()
        
        # 检查是否需要准备数据
        need_data_preparation = True
        
        # 检查训练数据是否已存在
        train_path = normalize_path(paths.train_dir)
        if os.path.exists(train_path):
            train_files = sum(len(glob.glob(os.path.join(d, "*.*"))) 
                           for d in glob.glob(os.path.join(train_path, "*")) if os.path.isdir(d))
            if train_files > config.min_files_threshold:
                logger.info(f"训练数据已存在 ({train_files} 个文件)，跳过训练数据准备")
                need_data_preparation = False
        
        # 检查测试数据是否已存在
        test_path = normalize_path(paths.test_images_dir)
        need_test_data = True
        if os.path.exists(test_path) and len(glob.glob(os.path.join(test_path, "*.*"))) > 0:
            need_test_data = False
            logger.info(f"测试数据已存在，跳过测试数据准备")
        
        # 根据检查结果运行不同的数据准备步骤
        if need_data_preparation:
            logger.info("准备完整训练和测试数据")
            prepare_args = argparse.Namespace(
                extract=True,
                process=True,
                augment=True,
                status=True,
                merge="all",
                cleanup=False,
                dataset_path=None,
                merge_augmented=True,
                no_merge_augmented=False
            )
            prepare_data(prepare_args)
        elif need_test_data:
            logger.info("仅准备测试数据")
            
            # 设置测试集合并标志
            config.merge_test_datasets = True
            logger.info("已启用测试集合并")
            
            # 获取data目录下所有包含test的zip文件
            data_dir = normalize_path(paths.data_dir)
            test_files = []
            for ext in config.supported_dataset_formats:
                test_files.extend(glob.glob(os.path.join(data_dir, f"*test*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, f"*TEST*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, "**", f"*test*{ext}")))
                test_files.extend(glob.glob(os.path.join(data_dir, "**", f"*TEST*{ext}")))
            
            if test_files:
                logger.info(f"找到以下测试数据集文件: {test_files}")
                
                # 确保测试解压目录存在
                extract_to = normalize_path(os.path.join(paths.temp_dataset_dir, "AgriculturalDisease_testset"))
                os.makedirs(extract_to, exist_ok=True)
                
                # 解压所有测试数据集文件
                for test_file in test_files:
                    logger.info(f"解压测试数据集文件: {test_file}")
                    if data_prep.extract_zip_file(test_file, extract_to):
                        logger.info(f"成功解压 {test_file}")
                    else:
                        logger.error(f"解压 {test_file} 失败")
                
                # 确保测试目录存在
                os.makedirs(paths.test_images_dir, exist_ok=True)
                
                # 查找所有可能的测试图像目录
                potential_image_dirs = []
                
                # 1. 直接检查标准images目录
                standard_images_dir = normalize_path(os.path.join(extract_to, "images"))
                if os.path.exists(standard_images_dir) and os.path.isdir(standard_images_dir):
                    potential_image_dirs.append(standard_images_dir)
                
                # 2. 检查可能的测试A/B子目录中的images
                test_subdirs = glob.glob(os.path.join(extract_to, "AgriculturalDisease_test*"))
                for subdir in test_subdirs:
                    subdir_images = os.path.join(subdir, "images")
                    if os.path.exists(subdir_images) and os.path.isdir(subdir_images):
                        potential_image_dirs.append(subdir_images)
                
                # 3. 递归搜索其他可能包含图像的目录
                for root, dirs, files in os.walk(extract_to):
                    if not any(root.startswith(d) for d in potential_image_dirs):
                        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if image_files and os.path.basename(root) != "labels":
                            potential_image_dirs.append(root)
                
                if potential_image_dirs:
                    logger.info(f"找到以下测试图像目录: {potential_image_dirs}")
                    
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
                        logger.info(f"从 {image_dir} 复制了 {count} 个图像文件到 {paths.test_images_dir}")
                    
                    if copied_count > 0:
                        logger.info(f"成功准备测试数据: 总共复制了 {copied_count} 个图像文件")
                    else:
                        logger.warning(f"未能找到任何测试图像文件")
                else:
                    logger.error("在解压后的目录中未找到测试图像目录")
            else:
                logger.error("找不到任何测试数据集文件")
        else:
            logger.info("所有数据均已存在，跳过数据准备")
        
        # 2. 训练模型
        logger.info("Step 2: Model training")
        train_args = argparse.Namespace(
            epochs=config.epoch,
            model=config.model_name,
            batch_size=config.train_batch_size,
            lr=config.lr,
            prepare=False,  # 已经在步骤1中准备了数据
            no_prepare=True,
            force_train=True,
            merge_augmented=True,
            no_merge_augmented=False,
            dataset_path=None,
            optimizer=config.optimizer,
            weight_decay=config.weight_decay,
            no_lookahead=False,
            scheduler=config.scheduler,
            warmup_epochs=config.warmup_epochs,
            warmup_factor=config.warmup_factor,
            no_mixup=False,
            mixup_alpha=config.mixup_alpha,
            cutmix_prob=config.cutmix_prob,
            no_random_erasing=False,
            no_early_stopping=False,
            patience=config.early_stopping_patience,
            no_amp=False,
            gradient_clip_val=config.gradient_clip_val,
            no_ema=False,
            ema_decay=config.ema_decay,
            device=config.device,
            gpus=config.gpus,
            seed=config.seed,
            label_smoothing=config.label_smoothing,
            no_gradient_checkpointing=False
        )
        train_pipeline(train_args)
        
        # 3. 模型推理
        logger.info("Step 3: Model inference")
        # 查找最佳模型文件
        best_model_path = os.path.join(config.best_weights, config.model_name, "0", "model_best.pth.tar")
        if not os.path.exists(best_model_path):
            best_model_path = os.path.join(config.weights, config.model_name, "0", "_checkpoint.pth.tar")
        
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
            logger.info("启用测试集合并")
            config.merge_test_datasets = True
        run_inference(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main() 