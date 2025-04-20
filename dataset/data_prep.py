import concurrent.futures
import json
import shutil
import os
import zipfile
import glob
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
from skimage.util import random_noise
from skimage import exposure
import albumentations as A
from typing import List, Optional, Tuple, Union, Dict, Any
from config.config import config, paths
import random
from concurrent.futures import ThreadPoolExecutor
from utils.utils import handle_datasets
import traceback
import tarfile
import time
import re
import pandas as pd
import threading

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(paths.data_proc_log),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DataPreparation')

# 添加路径规范化辅助函数
def normalize_path(path):
    """规范化路径，确保路径分隔符一致性
    
    Args:
        path: 原始路径
        
    Returns:
        规范化后的路径
    """
    if path is None:
        return None
    # 使用os.path.normpath确保路径分隔符的一致性
    normalized = os.path.normpath(path)
    # 始终转换为正斜杠格式，无论操作系统
    normalized = normalized.replace('\\', '/')
    return normalized

def contains_chinese_char(text: str) -> bool:
    """检查文本是否包含中文字符
    
    参数:
        text: 要检查的文本
        
    返回:
        是否包含中文字符
    """
    # 中文字符的Unicode范围
    chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]')
    return bool(chinese_pattern.search(text))

class DataPreparation:
    """统一的数据准备类，结合数据集提取、处理和增强功能"""
    
    def __init__(self, config_obj=None):
        """初始化数据准备
        
        参数:
            config_obj: 配置对象(如果为None则使用默认配置)
        """
        self.config = config_obj or config
        self.paths = paths
        self.error_images = []
        
        # 初始化增强管道
        self.aug_pipeline = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(0.01, 0.05)),
                A.GaussianBlur(blur_limit=3),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.ElasticTransform(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
    
    def setup_directories(self) -> None:
        """创建项目所需的所有目录"""
        try:
            # 训练数据目录
            for i in range(0, 59):
                os.makedirs(os.path.join(self.paths.train_dir, str(i)), exist_ok=True)
            
            # 临时目录
            os.makedirs(self.paths.temp_images_dir, exist_ok=True)
            os.makedirs(self.paths.temp_labels_dir, exist_ok=True)
            
            # 测试图片目录
            os.makedirs(self.paths.test_images_dir, exist_ok=True)
            
            # 输出和日志目录
            os.makedirs(self.paths.submit_dir, exist_ok=True)
            os.makedirs(self.paths.logs_dir, exist_ok=True)
            
            # 模型保存目录 - 更清晰的命名
            os.makedirs(self.paths.weights_dir, exist_ok=True)
            os.makedirs(self.paths.best_weights_dir, exist_ok=True)
            
            # 数据增强目录
            os.makedirs(self.paths.aug_train_dir, exist_ok=True)
            
            # 合并数据集目录
            os.makedirs(self.paths.merged_train_dir, exist_ok=True)
            os.makedirs(self.paths.merged_test_dir, exist_ok=True)
            os.makedirs(self.paths.merged_val_dir, exist_ok=True)
            
            logger.info("All directories created successfully")
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise
    
    def extract_zip_file(self, zip_path: str, extract_to: Optional[str] = None) -> bool:
        """解压ZIP文件
        
        参数:
            zip_path: ZIP文件路径
            extract_to: 解压目标路径，如果为None则解压到同目录
        
        返回:
            布尔值，表示是否成功
        """
        try:
            if extract_to is None:
                extract_to = os.path.dirname(zip_path)
            
            logger.info(f"Extracting {os.path.basename(zip_path)} to {extract_to}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info(f"Successfully extracted {os.path.basename(zip_path)}")
            return True
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {str(e)}")
            return False
    
    def copy_files_to_folder(self, source_folder: str, destination_folder: str, 
                           file_pattern: str = "*") -> int:
        """复制源文件夹中的所有匹配文件到目标文件夹
        
        参数:
            source_folder: 源文件夹路径
            destination_folder: 目标文件夹路径
            file_pattern: 文件匹配模式
        
        返回:
            复制的文件数量
        """
        # 确保目标文件夹存在
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # 获取匹配的文件列表
        files = glob.glob(os.path.join(source_folder, file_pattern))
        
        if not files:
            logger.warning(f"No files matching pattern '{file_pattern}' found in {source_folder}")
            return 0

        files_copied = 0
        for file_path in tqdm(files, desc=f"Copying files to {destination_folder}"):
            try:
                # 拼接目标文件的完整路径
                destination_file_path = os.path.join(destination_folder, os.path.basename(file_path))
                
                # 执行文件复制操作
                shutil.copy2(file_path, destination_file_path)
                files_copied += 1
            except Exception as e:
                logger.error(f"Error copying {file_path}: {str(e)}")
        
        logger.info(f'Copied {files_copied} files from {source_folder} to {destination_folder}')
        return files_copied
    
    def copy_file(self, file: Dict[str, Any]) -> bool:
        """复制单个文件到指定目录
        
        参数:
            file: 包含图像文件信息的字典
            
        返回:
            布尔值，指示是否成功处理
        """
        try:
            filename = file["image_id"]
            origin_path = normalize_path(os.path.join(self.paths.data_dir, "temp", "images", filename))
            
            # 跳过类别44和45
            ids = file["disease_class"]
            if ids == 44 or ids == 45:
                return False
                
            # 调整ID顺序（大于45的ID要减2）
            if ids > 45:
                ids -= 2
                
            # 确保目标目录存在
            save_dir = normalize_path(os.path.join(self.paths.data_dir, "train", str(ids)))
            os.makedirs(save_dir, exist_ok=True)
            
            # 构建目标路径并复制文件
            save_path = normalize_path(os.path.join(save_dir, filename))
            
            # 检查源文件是否存在
            if not os.path.exists(origin_path):
                logger.error(f"Source file does not exist: {origin_path}")
                return False
            
            shutil.copy(origin_path, save_path)
            return True
        except Exception as e:
            logger.error(f"Error processing file {file.get('image_id', 'unknown')}: {str(e)}")
            return False

    def extract_dataset(self, cleanup_temp=False) -> None:
        """解压数据集文件并准备数据
        
        参数:
            cleanup_temp: 处理完成后是否清理临时文件夹
        """
        self.setup_directories()
        
        # 确定数据集源路径
        training_file = self.find_dataset_file(
            self.config.training_dataset_file, 
            self.config.dataset_path if self.config.use_custom_dataset_path else None
        )
        validation_file = self.find_dataset_file(
            self.config.validation_dataset_file,
            self.config.dataset_path if self.config.use_custom_dataset_path else None
        )
        
        # 使用模式匹配查找测试集
        test_file = self.find_dataset_file(
            self.config.test_name_pattern,
            self.config.dataset_path if self.config.use_custom_dataset_path else None
        )
        
        has_training = training_file is not None
        has_validation = validation_file is not None
        has_test = test_file is not None
        
        # 记录找到的数据集文件
        logger.info(f"Found training dataset: {training_file if has_training else 'Not found'}")
        logger.info(f"Found validation dataset: {validation_file if has_validation else 'Not found'}")
        logger.info(f"Found test dataset: {test_file if has_test else 'Not found'}")
        
        # 解压训练集
        if has_training:
            extract_to = normalize_path(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_trainingset"))
            os.makedirs(extract_to, exist_ok=True)
            if self.extract_zip_file(training_file, extract_to):
                # 提取图像和标签
                images_dir = normalize_path(os.path.join(extract_to, "images"))
                if os.path.exists(images_dir):
                    self.copy_files_to_folder(images_dir, self.paths.temp_images_dir)
                
                anno_path = normalize_path(os.path.join(extract_to, "AgriculturalDisease_train_annotations.json"))
                if os.path.exists(anno_path):
                    shutil.copy(anno_path, self.paths.train_annotations)
                    logger.info(f"Copied training annotations to {self.paths.train_annotations}")
                else:
                    logger.warning(f"Training annotations not found at {anno_path}")
        else:
            logger.warning("Training dataset file not found, skipping extraction")
        
        # 解压验证集
        if has_validation:
            extract_to = normalize_path(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_validationset"))
            os.makedirs(extract_to, exist_ok=True)
            if self.extract_zip_file(validation_file, extract_to):
                # 提取图像和标签
                images_dir = normalize_path(os.path.join(extract_to, "images"))
                if os.path.exists(images_dir):
                    self.copy_files_to_folder(images_dir, self.paths.temp_images_dir)
                    # 同时复制到验证图像目录
                    val_images_dir = normalize_path(os.path.join(self.paths.val_dir, "images"))
                    os.makedirs(val_images_dir, exist_ok=True)
                    self.copy_files_to_folder(images_dir, val_images_dir)
                
                anno_path = normalize_path(os.path.join(extract_to, "AgriculturalDisease_validation_annotations.json"))
                if os.path.exists(anno_path):
                    shutil.copy(anno_path, self.paths.val_annotations)
                    logger.info(f"Copied validation annotations to {self.paths.val_annotations}")
                else:
                    logger.warning(f"Validation annotations not found at {anno_path}")
        else:
            logger.warning("Validation dataset file not found, skipping extraction")
        
        # 解压测试集
        if has_test:
            extract_to = normalize_path(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_testset"))
            os.makedirs(extract_to, exist_ok=True)
            
            # 从文件名确定是testA还是testB
            test_filename = os.path.basename(test_file)
            if "testa" in test_filename.lower():
                test_type = "testA"
            elif "testb" in test_filename.lower():
                test_type = "testB"
            else:
                test_type = "test"
                
            if self.extract_zip_file(test_file, extract_to):
                # 提取图像
                images_dir = normalize_path(os.path.join(extract_to, "images"))
                if os.path.exists(images_dir):
                    # 复制到测试图像目录
                    self.copy_files_to_folder(images_dir, self.paths.test_images_dir)
                    logger.info(f"Copied {test_type} images to {self.paths.test_images_dir}")
        else:
            logger.warning("Test dataset file not found, skipping extraction")
        
        # 清理临时文件
        if cleanup_temp:
            self.cleanup_temp_folders()
            logger.info("Cleaned up temporary folders")

    def process_data(self) -> None:
        """处理数据集文件并组织到训练目录"""
        # 确保目录结构存在
        self.setup_directories()
        
        # 检查处理后的数据是否已经存在
        train_processed = self.is_valid_dataset_directory(self.paths.train_dir, 'train')
        val_processed = os.path.exists(self.paths.val_dir) and len(glob.glob(os.path.join(self.paths.val_dir, "images", "*.jpg"))) > 0
        
        if train_processed and val_processed and not self.config.force_data_processing:
            logger.info("Processed data already exists. Set force_data_processing=True to reprocess.")
            return
        
        # 加载标注文件
        try:
            train_json = self.paths.train_annotations
            val_json = self.paths.val_annotations
            
            if not os.path.exists(train_json):
                logger.error(f"Training annotations file not found: {train_json}")
                logger.info("Please run extraction first to prepare the dataset")
                return
                
            if not os.path.exists(val_json):
                logger.error(f"Validation annotations file not found: {val_json}")
                logger.info("Please run extraction first to prepare the dataset")
                return
                
            file_train = json.load(open(train_json, "r", encoding="utf-8"))
            file_val = json.load(open(val_json, "r", encoding="utf-8"))
            file_list = file_train + file_val
        except FileNotFoundError as e:
            logger.error(f"Annotation files not found: {str(e)}")
            logger.info(f"Please ensure data is correctly placed in {self.paths.temp_labels_dir} directory")
            return
        except json.JSONDecodeError as e:
            logger.error(f"Invalid annotation file format: {str(e)}")
            logger.info("Please check JSON file validity")
            return
        
        logger.info(f"Found {len(file_list)} annotation entries")
        
        # 统计每个类别的文件数
        class_counts = {}
        for file in file_list:
            class_id = file["disease_class"]
            if class_id not in class_counts:
                class_counts[class_id] = 0
            class_counts[class_id] += 1
        
        # 显示每个类别的文件数
        logger.info("\nClass distribution:")
        for class_id, count in sorted(class_counts.items()):
            logger.info(f"Class {class_id}: {count} files")
        
        # 检查是否有特殊类别44和45
        if 44 in class_counts or 45 in class_counts:
            logger.warning("\nWarning: Classes 44 and 45 will be ignored, classes >45 will be reduced by 2")
        
        # 创建线程池处理文件
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # 提交任务
            futures = [executor.submit(self.copy_file, file) for file in file_list]
            
            # 使用tqdm跟踪进度
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files"):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error in thread: {str(e)}")
        
        logger.info(f"\nSuccessfully processed {success_count}/{len(file_list)} files")

    def get_data_status(self) -> dict:
        """获取数据准备的状态信息
        
        返回:
            包含数据集状态信息的字典
        """
        status = {
            "status": "success",
            "datasets": {},
            "dataset_extracted": False,
            "data_processed": False,
            "augmentation_completed": False,
            "merged_datasets": False,
            "zip_details": {},
            "processed_details": {},
            "merged_details": {},
            "augmented_image_count": 0
        }
        
        # 检查是否已提取数据集
        try:
            # 检查提取的数据集文件夹是否存在
            status["dataset_extracted"] = os.path.exists(self.paths.temp_dataset_dir) and len(os.listdir(self.paths.temp_dataset_dir)) > 0
            
            # 检查训练、测试和验证目录
            training_dir_exists = os.path.exists(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_trainingset"))
            validation_dir_exists = os.path.exists(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_validationset"))
            
            status["zip_details"] = {
                "training": training_dir_exists,
                "validation": validation_dir_exists
            }
        except Exception as e:
            logger.error(f"Error checking dataset extraction status: {str(e)}")
            status["dataset_extracted"] = False
        
        # 检查数据处理状态
        try:
            # 检查训练和测试目录是否有图像
            train_processed = os.path.exists(self.paths.train_dir) and len(glob.glob(os.path.join(self.paths.train_dir, "**/*.jpg"), recursive=True)) > 0
            test_processed = os.path.exists(self.paths.test_images_dir) and len(glob.glob(os.path.join(self.paths.test_images_dir, "*.jpg"))) > 0
            
            status["data_processed"] = train_processed or test_processed
            status["processed_details"] = {
                "training": train_processed,
                "testing": test_processed
            }
        except Exception as e:
            logger.error(f"Error checking data processing status: {str(e)}")
            status["data_processed"] = False
        
        # 检查数据增强状态
        try:
            # 检查增强数据目录是否有图像
            aug_dir_exists = os.path.exists(self.paths.augmented_images_dir)
            if aug_dir_exists:
                aug_images = glob.glob(os.path.join(self.paths.augmented_images_dir, "**/*.jpg"), recursive=True)
                status["augmentation_completed"] = len(aug_images) > 0
                status["augmented_image_count"] = len(aug_images)
            else:
                status["augmentation_completed"] = False
                status["augmented_image_count"] = 0
        except Exception as e:
            logger.error(f"Error checking data augmentation status: {str(e)}")
            status["augmentation_completed"] = False
        
        # 检查数据集合并状态
        try:
            # 检查合并目录是否存在
            merged_train = os.path.exists(self.paths.merged_train_dir) and len(glob.glob(os.path.join(self.paths.merged_train_dir, "**/*.jpg"), recursive=True)) > 0
            merged_test = os.path.exists(self.paths.merged_test_dir) and len(glob.glob(os.path.join(self.paths.merged_test_dir, "*.jpg"))) > 0
            merged_val = os.path.exists(self.paths.merged_val_dir) and len(glob.glob(os.path.join(self.paths.merged_val_dir, "*.jpg"))) > 0
            
            status["merged_datasets"] = merged_train or merged_test or merged_val
            status["merged_details"] = {
                "training": merged_train,
                "testing": merged_test,
                "validation": merged_val
            }
        except Exception as e:
            logger.error(f"Error checking dataset merging status: {str(e)}")
            status["merged_datasets"] = False
        
        # 检查各目录中的数据情况
        data_dirs = {
            "train": self.paths.train_dir,
            "test": self.paths.test_dir,
            "val": self.paths.val_dir,
            "augmented": self.paths.aug_dir if hasattr(self.paths, "aug_dir") else None,
            "merged_train": self.paths.merged_train_dir if config.merge_datasets else None,
            "merged_test": self.paths.merged_test_dir if config.merge_datasets else None,
            "merged_val": self.paths.merged_val_dir if config.merge_datasets else None
        }
        
        for name, path in data_dirs.items():
            if path and os.path.exists(path):
                # 计算目录中的文件数量
                try:
                    if os.path.isdir(path):
                        if name == "train":
                            # 递归统计训练目录中所有图像
                            img_count = len(glob.glob(os.path.join(path, "**/*.jpg"), recursive=True))
                            # 检查类别数量
                            class_count = len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
                            status["datasets"][name] = {
                                "path": path,
                                "exists": True,
                                "file_count": img_count,
                                "class_count": class_count
                            }
                        else:
                            # 非训练目录，直接统计文件数
                            file_count = len([f for f in glob.glob(os.path.join(path, "*.jpg"))])
                            status["datasets"][name] = {
                                "path": path,
                                "exists": True,
                                "file_count": file_count
                            }
                    else:
                        status["datasets"][name] = {
                            "path": path,
                            "exists": True,
                            "file_count": 0,
                            "error": "Path is not a directory"
                        }
                except Exception as e:
                    status["datasets"][name] = {
                        "path": path,
                        "exists": True,
                        "error": str(e)
                    }
            else:
                status["datasets"][name] = {
                    "path": path,
                    "exists": False,
                    "file_count": 0
                }
        
        # 检查临时目录状态
        temp_dir = self.paths.temp_dir
        if os.path.exists(temp_dir):
            try:
                temp_files = os.listdir(temp_dir)
                status["temp_dir"] = {
                    "path": temp_dir,
                    "exists": True,
                    "file_count": len(temp_files),
                    "files": temp_files[:10]  # 只列出前10个文件以避免过多数据
                }
                if len(temp_files) > 10:
                    status["temp_dir"]["files"].append("... and more")
            except Exception as e:
                status["temp_dir"] = {
                    "path": temp_dir,
                    "exists": True,
                    "error": str(e)
                }
        else:
            status["temp_dir"] = {
                "path": temp_dir,
                "exists": False
            }
        
        # 计算整体的训练图像数量
        training_imgs = 0
        if "train" in status["datasets"] and "file_count" in status["datasets"]["train"]:
            training_imgs += status["datasets"]["train"]["file_count"]
        if "merged_train" in status["datasets"] and "file_count" in status["datasets"]["merged_train"]:
            training_imgs += status["datasets"]["merged_train"]["file_count"]
        
        status["training"] = {
            "directory": self.paths.train_dir,
            "total_images": training_imgs
        }
        
        # 计算整体的测试图像数量
        testing_imgs = 0
        if "test" in status["datasets"] and "file_count" in status["datasets"]["test"]:
            testing_imgs += status["datasets"]["test"]["file_count"]
        if "merged_test" in status["datasets"] and "file_count" in status["datasets"]["merged_test"]:
            testing_imgs += status["datasets"]["merged_test"]["file_count"]
        
        status["testing"] = {
            "directory": self.paths.test_dir,
            "total_images": testing_imgs
        }
        
        # 计算整体的验证图像数量
        validation_imgs = 0
        if "val" in status["datasets"] and "file_count" in status["datasets"]["val"]:
            validation_imgs += status["datasets"]["val"]["file_count"]
        if "merged_val" in status["datasets"] and "file_count" in status["datasets"]["merged_val"]:
            validation_imgs += status["datasets"]["merged_val"]["file_count"]
        
        status["validation"] = {
            "directory": self.paths.val_dir,
            "total_images": validation_imgs
        }
        
        # 检查配置信息
        status["config"] = {
            "use_data_aug": config.use_data_aug,
            "merge_datasets": config.merge_datasets,
            "custom_dataset_path": config.dataset_path if config.use_custom_dataset_path else None
        }
        
        return status

    def check_data_status(self) -> None:
        """检查数据准备状态并输出信息"""
        status = self.get_data_status()
        
        logger.info("=" * 50)
        logger.info("DATA PREPARATION STATUS")
        logger.info("=" * 50)
        
        # 检查数据集提取状态
        logger.info("\nDataset Extraction:")
        if status["dataset_extracted"]:
            logger.info("[OK] Dataset extraction completed")
        else:
            logger.info("[MISSING] Dataset extraction not completed")
            
        for zip_file, extracted in status["zip_details"].items():
            if extracted:
                logger.info(f"  [OK] {os.path.basename(zip_file)} extracted")
            else:
                logger.info(f"  [MISSING] {os.path.basename(zip_file)} not extracted")
        
        # 检查数据处理状态
        logger.info("\nData Processing:")
        if status["data_processed"]:
            logger.info("[OK] Data processing completed")
        else:
            logger.info("[MISSING] Data processing not completed")
            
        for data_type, processed in status["processed_details"].items():
            if processed:
                logger.info(f"  [OK] {data_type} data processed")
            else:
                logger.info(f"  [MISSING] {data_type} data not processed")
        
        # 检查数据增强状态
        logger.info("\nData Augmentation:")
        if status["augmentation_completed"]:
            logger.info("[OK] Data augmentation completed")
            logger.info(f"  Total augmented images: {status['augmented_image_count']}")
        else:
            logger.info("[MISSING] Data augmentation not completed")
        
        # 检查数据集合并状态
        logger.info("\nDataset Merging:")
        if status["merged_datasets"]:
            logger.info("[OK] Datasets merged")
            for data_type, merged in status["merged_details"].items():
                if merged:
                    logger.info(f"  [OK] {data_type} datasets merged")
                else:
                    logger.info(f"  [MISSING] {data_type} datasets not merged")
        else:
            logger.info("[MISSING] Datasets not merged")
        
        logger.info("\nNext Recommended Steps:")
        if not status["dataset_extracted"]:
            logger.info("1. Extract dataset archives")
        elif not status["data_processed"]:
            logger.info("2. Process extracted data")
        elif not status["augmentation_completed"] and self.config.use_data_aug:
            logger.info("3. Run data augmentation")
        elif not status["merged_datasets"] and self.config.merge_datasets:
            logger.info("4. Merge datasets")
        else:
            logger.info("All data preparation steps completed!")
        
        logger.info("=" * 50)

    def add_noise(self, img: np.ndarray) -> Optional[np.ndarray]:
        """添加高斯噪声到图像
        
        参数:
            img: 输入图像
            
        返回:
            添加噪声后的图像，如果失败则返回None
        """
        if img is None:
            logger.warning("Input image is None")
            return None
        try:
            img_float = img.astype(np.float32) / 255.0
            noisy = random_noise(img_float, mode='gaussian', 
                               var=self.config.aug_noise_var, 
                               clip=True)
            noisy = (noisy * 255).astype(np.uint8)
            return noisy
        except Exception as e:
            logger.error(f"Error adding noise: {str(e)}")
            return None

    def change_brightness(self, img: np.ndarray) -> Optional[np.ndarray]:
        """调整图像亮度
        
        参数:
            img: 输入图像
            
        返回:
            调整亮度后的图像，如果失败则返回None
        """
        if img is None:
            logger.warning("Input image is None")
            return None
        try:
            rate = random.uniform(*self.config.aug_brightness_range)
            img_float = img.astype(np.float32) / 255.0
            adjusted = exposure.adjust_gamma(img_float, rate)
            adjusted = (adjusted * 255).astype(np.uint8)
            return adjusted
        except Exception as e:
            logger.error(f"Error adjusting brightness: {str(e)}")
            return None

    def apply_advanced_augmentation(self, img: np.ndarray) -> Optional[np.ndarray]:
        """应用高级数据增强
        
        参数:
            img: 输入图像
            
        返回:
            增强后的图像，如果失败则返回None
        """
        if img is None:
            logger.warning("Input image is None")
            return None
        try:
            augmented = self.aug_pipeline(image=img)['image']
            return augmented
        except Exception as e:
            logger.error(f"Error applying advanced augmentation: {str(e)}")
            return None

    def is_valid_image(self, image_path: str) -> bool:
        """检查图像是否有效
        
        参数:
            image_path: 图像路径
            
        返回:
            图像是否有效
        """
        # 规范化路径
        image_path = normalize_path(image_path)
        
        try:
            # 确保文件存在
            if not os.path.exists(image_path):
                logger.debug(f"Image file does not exist: {image_path}")
                return False
                
            # 检查文件大小
            file_size = os.path.getsize(image_path)
            if file_size < 100:  # 小于100字节的文件几乎不可能是有效图像
                logger.debug(f"Image file too small ({file_size} bytes): {image_path}")
                return False
            
            # 使用PIL快速检查，只验证文件头
            try:
                with Image.open(image_path) as img:
                    # 仅验证图像头信息，不加载完整数据
                    img.verify()  
                    return True
            except Exception as e:
                logger.debug(f"PIL validation failed for {image_path}: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error validating image {image_path}: {str(e)}")
            return False
    
    def remove_error_images(self, image_dir: str) -> None:
        """移除错误图像文件
        
        参数:
            image_dir: 图像目录
        """
        # 规范化路径
        image_dir = normalize_path(image_dir)
        
        if not os.path.exists(image_dir):
            logger.warning(f"Image directory does not exist: {image_dir}")
            return
        
        logger.info(f"Checking for invalid images in {image_dir}")
        
        all_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            # 使用正规化的路径模式，强制使用正斜杠
            pattern = normalize_path(os.path.join(image_dir, f"**/*{ext}"))
            all_images.extend(glob.glob(pattern, recursive=True))
        
        # 确保所有图片路径都使用正确的格式
        all_images = [normalize_path(img) for img in all_images]
        
        logger.info(f"Found {len(all_images)} images to check")
        
        # 如果没有图片，直接返回
        if not all_images:
            logger.info("No images found to check")
            return
        
        # 初始化计数器
        invalid_images = []
        chinese_char_images = []  # 记录包含中文字符的图像
        error_count = 0
        
        # 设置合理的工作线程数 - 限制线程数以避免资源耗尽
        max_workers = min(self.config.aug_max_workers, 8)  # 最多使用8个线程，避免过度并行
        
        # 使用更小的批次尺寸来降低内存使用
        batch_size = 50
        
        # 设置超时时间（秒）
        # timeout = 0.5
        
        try:
            # 使用多线程加速验证
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                logger.info(f"Checking images with {max_workers} workers in batches of {batch_size}")
                
                def check_image(img_path):
                    """检查单个图像是否有效"""
                    try:
                        img_path = normalize_path(img_path)
                        if not os.path.exists(img_path):
                            logger.debug(f"Image file does not exist: {img_path}")
                            return {"path": img_path, "invalid": True, "chinese": False}
                        
                        result = {"path": img_path, "invalid": False, "chinese": False}
                        
                        # 检查文件名是否包含中文字符
                        if contains_chinese_char(os.path.basename(img_path)):
                            result["chinese"] = True
                        
                        # 快速验证 - 首先检查文件是否存在且可以打开
                        if not self.is_valid_image(img_path):
                            result["invalid"] = True
                        
                        return result
                    except Exception as e:
                        logger.error(f"Error checking image {img_path}: {str(e)}")
                        return {"path": img_path, "invalid": True, "chinese": False}  # 出错时也认为图像无效
                
                # 分批处理图像以降低内存使用
                # total_batches = (len(all_images) + batch_size - 1) // batch_size
                
                # 使用tqdm显示总体进度
                with tqdm(total=len(all_images), desc="Checking images") as pbar:
                    for i in range(0, len(all_images), batch_size):
                        # 获取当前批次
                        batch = all_images[i:i+batch_size]
                        
                        # try:
                            # 提交批次任务并设置超时
                        futures = [executor.submit(check_image, img) for img in batch]
                        
                        # 处理完成的任务结果
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                result = future.result()
                                if result["invalid"]:  # 如果图像无效
                                    invalid_images.append(result["path"])
                                if result["chinese"]:  # 如果文件名包含中文
                                    chinese_char_images.append(result["path"])
                                pbar.update(1)
                            except Exception as e:
                                error_count += 1
                                pbar.update(1)
                                logger.error(f"Error processing task result: {str(e)}")
    
        except KeyboardInterrupt:
            logger.warning("Image validation interrupted by user")
            # 即使被中断也尽量删除已发现的无效图像
        
        # 删除无效图像
        if invalid_images:
            logger.info(f"Found {len(invalid_images)} invalid images, removing...")
            removed_count = 0
            
            for invalid_path in tqdm(invalid_images, desc="Removing invalid images", unit="img"):
                try:
                    invalid_path = normalize_path(invalid_path)
                    # 确保路径存在后再删除
                    if os.path.exists(invalid_path):
                        os.remove(invalid_path)
                        removed_count += 1
                    else:
                        logger.warning(f"Invalid image path does not exist: {invalid_path}")
                except Exception as e:
                    error_count += 1
                    logger.error(f"Failed to remove {invalid_path}: {str(e)}")
            
            logger.info(f"Removed {removed_count}/{len(invalid_images)} invalid images")
        else:
            logger.info("No invalid images found")
        
        # 处理包含中文字符的图像
        if chinese_char_images:
            logger.warning(f"Found {len(chinese_char_images)} images with Chinese characters in filenames")
            # 显示前10个示例
            for i, path in enumerate(chinese_char_images[:10]):
                logger.warning(f"  {i+1}. {os.path.basename(path)}")
            if len(chinese_char_images) > 10:
                logger.warning(f"  ... and {len(chinese_char_images) - 10} more files")
            
            logger.warning("Chinese characters in filenames may cause errors during processing.")
            
            # 询问用户是否要删除这些文件
            try:
                user_input = input("Do you want to remove these files with Chinese characters? (yes/no): ").strip().lower()
                if user_input in ['yes', 'y']:
                    removed_count = 0
                    for chinese_path in tqdm(chinese_char_images, desc="Removing files with Chinese characters", unit="img"):
                        try:
                            chinese_path = normalize_path(chinese_path)
                            if os.path.exists(chinese_path):
                                os.remove(chinese_path)
                                removed_count += 1
                            else:
                                logger.warning(f"Path does not exist: {chinese_path}")
                        except Exception as e:
                            error_count += 1
                            logger.error(f"Failed to remove {chinese_path}: {str(e)}")
                    
                    logger.info(f"Removed {removed_count}/{len(chinese_char_images)} files with Chinese characters")
                else:
                    logger.warning("Files with Chinese characters were kept. This may cause errors during processing.")
            except Exception as e:
                logger.error(f"Error processing user input: {str(e)}")
                logger.warning("Files with Chinese characters were kept. This may cause errors during processing.")
        
        if error_count > 0:
            logger.warning(f"Failed to process {error_count} images")

    def augment_image(self, image_path: str, save_dir: str) -> List[str]:
        """对单张图像进行数据增强
        
        参数:
            image_path: 输入图像路径
            save_dir: 保存目录
            
        返回:
            增强后的图像路径列表
        """
        try:
            # 规范化输入路径
            image_path = normalize_path(image_path)
            save_dir = normalize_path(save_dir)
            
            # 确保目录存在
            os.makedirs(save_dir, exist_ok=True)
            
            if not os.path.exists(image_path):
                logger.warning(f"Image path does not exist: {image_path}")
                return []

            try:
                img = Image.open(image_path)
                cv_image = cv2.imread(image_path)
                
                if cv_image is None:
                    logger.warning(f"Failed to read image: {image_path}")
                    return []
            except Exception as e:
                logger.error(f"Error opening image {image_path}: {str(e)}")
                return []

            basename = os.path.basename(image_path)
            augmented_images = []

            # 添加高斯噪声
            if self.config.aug_noise:
                gau_image = self.add_noise(cv_image)
                if gau_image is not None:
                    output_path = normalize_path(os.path.join(save_dir, f"gau_{basename}"))
                    try:
                        cv2.imwrite(output_path, gau_image)
                        augmented_images.append(output_path)
                    except Exception as e:
                        logger.error(f"Error saving noised image to {output_path}: {str(e)}")

            # 调整亮度
            if self.config.aug_brightness:
                light_image = self.change_brightness(cv_image)
                if light_image is not None:
                    output_path = normalize_path(os.path.join(save_dir, f"light_{basename}"))
                    try:
                        cv2.imwrite(output_path, light_image)
                        augmented_images.append(output_path)
                    except Exception as e:
                        logger.error(f"Error saving brightness image to {output_path}: {str(e)}")

            # 翻转
            if self.config.aug_flip:
                lr_path = normalize_path(os.path.join(save_dir, f"left_right_{basename}"))
                tb_path = normalize_path(os.path.join(save_dir, f"top_bottom_{basename}"))
                
                try:
                    img_flip_left_right = img.transpose(Image.FLIP_LEFT_RIGHT)
                    img_flip_top_bottom = img.transpose(Image.FLIP_TOP_BOTTOM)
                    
                    img_flip_left_right.save(lr_path)
                    img_flip_top_bottom.save(tb_path)
                    
                    augmented_images.extend([lr_path, tb_path])
                except Exception as e:
                    logger.error(f"Error saving flipped images: {str(e)}")
        
            return augmented_images
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            traceback.print_exc()
            return []

    def augment_directory(self, source_dir: Optional[str] = None, 
                         target_dir: Optional[str] = None) -> List[str]:
        """对整个目录进行数据增强
        
        参数:
            source_dir: 源数据目录
            target_dir: 目标保存目录
            
        返回:
            增强后的图像路径列表
        """
        if not self.config.use_data_aug:
            logger.warning("Data augmentation is disabled in config.py (use_data_aug=False)")
            return []

        # 使用配置文件中的路径，如果没有指定参数
        source_dir = source_dir or self.config.aug_source_path
        target_dir = target_dir or self.config.aug_target_path
        
        # 规范化路径
        source_dir = normalize_path(source_dir)
        target_dir = normalize_path(target_dir)
        
        # 检查源目录是否存在
        if not os.path.exists(source_dir):
            logger.error(f"Source directory does not exist: {source_dir}")
            return []
        
        # 检查目标目录是否已经包含增强数据
        if os.path.exists(target_dir) and not self.config.force_augmentation:
            if self.is_valid_dataset_directory(target_dir, 'train'):
                logger.info(f"Target directory already contains valid augmented data. Set force_augmentation=True to redo.")
                
                # 统计目标目录中的文件数量并返回文件路径列表
                file_paths = []
                for root, _, files in os.walk(target_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            file_paths.append(os.path.join(root, file))
                
                logger.info(f"Found {len(file_paths)} existing augmented files")
                return file_paths
        
        os.makedirs(target_dir, exist_ok=True)
        augmented_files = []
        
        # 首先删除错误图片
        if self.config.remove_error_images:
            self.remove_error_images(source_dir)
        
        # 获取所有图片文件
        image_files = []
        try:
            for image_path in Path(source_dir).rglob('*'):
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # 确保路径格式正确
                    image_files.append(normalize_path(str(image_path)))
        except Exception as e:
            logger.error(f"Error finding image files in {source_dir}: {str(e)}")
            return []

        logger.info(f"Found {len(image_files)} images for augmentation")
        
        error_count = 0
        with ThreadPoolExecutor(max_workers=self.config.aug_max_workers) as executor:
            futures = []
            for image_path in image_files:
                try:
                    # 规范化相对路径计算
                    rel_path = os.path.relpath(image_path, source_dir)
                    rel_path = normalize_path(rel_path)
                    save_dir = normalize_path(os.path.join(target_dir, os.path.dirname(rel_path)))
                    os.makedirs(save_dir, exist_ok=True)
                    futures.append(executor.submit(self.augment_image, image_path, save_dir))
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error setting up augmentation for {image_path}: {str(e)}")

        for future in tqdm(concurrent.futures.as_completed(futures), 
                         total=len(futures),
                         desc="Processing images"):
            try:
                result = future.result()
                if result:
                    augmented_files.extend(result)
            except Exception as e:
                error_count += 1
                logger.error(f"Error in augmentation thread: {str(e)}")

        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during augmentation")
            
        logger.info(f"Created {len(augmented_files)} augmented images")
        return augmented_files
    
    def merge_datasets(self, mode: str = "train", force: bool = False) -> str:
        """合并多个数据集到一个目录
        
        参数:
            mode: 数据集类型 'train', 'test' 或 'val'
            force: 是否强制覆盖现有的合并目录
            
        返回:
            合并后的数据集路径
        """
        logger.info(f"Merging {mode} datasets")
        
        # 定义合并目标路径
        merged_path = None
        if mode == "train":
            merged_path = normalize_path(self.paths.merged_train_dir)
        elif mode == "test":
            merged_path = normalize_path(self.paths.merged_test_dir)
        elif mode == "val":
            merged_path = normalize_path(self.paths.merged_val_dir)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'test', or 'val'")
        
        os.makedirs(merged_path, exist_ok=True)
        
        # 如果强制模式，清空目标目录
        force = force or self.config.force_merge
        if force and os.path.exists(merged_path):
            logger.info(f"Force flag set. Removing existing merged directory: {merged_path}")
            shutil.rmtree(merged_path)
            os.makedirs(merged_path, exist_ok=True)
        
        # 检查目标目录是否已经有文件，如果有则跳过合并
        if not force and os.path.exists(merged_path):
            if self.is_valid_dataset_directory(merged_path, mode):
                logger.info(f"Merged {mode} directory already contains valid data. Set force_merge=True to overwrite.")
                return merged_path
        
        # 查找需要合并的数据集
        datasets = handle_datasets(data_type=mode, list_only=True)
        
        # 如果是测试集和特定的主测试集
        if mode == "test" and not self.config.use_all_test_datasets:
            primary_test = self.config.primary_test_dataset
            filtered_datasets = [d for d in datasets if primary_test in d]
            
            if filtered_datasets:
                datasets = filtered_datasets
                logger.info(f"Using only primary test dataset: {self.config.primary_test_dataset}")
            else:
                logger.warning(f"Primary test dataset {self.config.primary_test_dataset} not found, using all available")
                
        if not datasets:
            logger.warning(f"No {mode} datasets found to merge")
            
            # 使用默认路径
            if mode == "train":
                return normalize_path(self.config.train_data)
            elif mode == "test":
                return normalize_path(self.config.test_data)
            else:
                return normalize_path(self.config.val_data)
        
        # 添加增强数据目录(仅训练集)
        if mode == "train" and self.config.use_data_aug and self.config.merge_augmented_data:
            aug_dir = normalize_path(self.paths.aug_train_dir)
            if os.path.exists(aug_dir):
                # 检查增强目录是否有内容
                aug_files = glob.glob(os.path.join(aug_dir, "**/*.*"), recursive=True)
                if aug_files:
                    logger.info(f"Adding augmented data directory to merge: {aug_dir}")
                    # 确保不重复添加
                    if aug_dir not in datasets:
                        datasets.append(aug_dir)
        
        logger.info(f"Found {len(datasets)} {mode} datasets to merge")
        
        if len(datasets) == 1 and not force:
            logger.info(f"Only one {mode} dataset found, no merge needed: {datasets[0]}")
            return normalize_path(datasets[0])
        
        # 确保目标目录存在
        os.makedirs(merged_path, exist_ok=True)
        
        # 执行合并
        logger.info(f"Merging {len(datasets)} {mode} datasets to {merged_path}")
        
        # 详细统计信息初始化
        dataset_stats = {}
        total_merged_files = 0
        
        # 创建集合用于跟踪文件名，防止冲突
        existing_filenames = set()
        
        try:
            if mode == "train":
                # 合并训练集：将所有类别目录下的图像复制到对应的merged目录
                for source_dir in datasets:
                    source_dir = normalize_path(source_dir)
                    logger.info(f"Processing training dataset: {source_dir}")
                    source_name = os.path.basename(source_dir)
                    dataset_stats[source_name] = {"files": 0, "classes": {}}
                    
                    # 获取源目录中的所有类别目录
                    class_dirs = []
                    if os.path.isdir(source_dir):
                        class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
                    
                    for class_dir in class_dirs:
                        class_path = normalize_path(os.path.join(source_dir, class_dir))
                        if os.path.isdir(class_path):
                            # 创建目标类别目录
                            target_class_dir = normalize_path(os.path.join(merged_path, class_dir))
                            os.makedirs(target_class_dir, exist_ok=True)
                            
                            # 复制图像文件
                            img_count = 0
                            img_paths = glob.glob(os.path.join(class_path, "*.jpg")) + glob.glob(os.path.join(class_path, "*.png"))
                            
                            for img in img_paths:
                                img = normalize_path(img)
                                # 创建唯一的目标文件名
                                base_name = os.path.basename(img)
                                img_ext = os.path.splitext(base_name)[1]
                                base_filename = os.path.splitext(base_name)[0]
                                
                                # 使用数据集名称和原始文件名创建新文件名
                                img_name = f"{source_name}_{base_filename}{img_ext}"
                                
                                # 确保文件名唯一
                                counter = 1
                                while img_name in existing_filenames:
                                    img_name = f"{source_name}_{base_filename}_{counter}{img_ext}"
                                    counter += 1
                                
                                existing_filenames.add(img_name)
                                target_path = normalize_path(os.path.join(target_class_dir, img_name))
                                
                                try:
                                    # 复制文件
                                    if not os.path.exists(target_path):
                                        shutil.copy2(img, target_path)
                                        img_count += 1
                                except Exception as e:
                                    logger.error(f"Error copying {img} to {target_path}: {str(e)}")
                                    continue
                            
                            if img_count > 0:
                                # 更新统计信息
                                dataset_stats[source_name]["files"] += img_count
                                dataset_stats[source_name]["classes"][class_dir] = img_count
                                total_merged_files += img_count
                                logger.info(f"  Copied {img_count} images from class {class_dir}")
            else:
                # 合并测试集或验证集：直接将所有图像复制到merged目录
                for source_dir in datasets:
                    source_dir = normalize_path(source_dir)
                    logger.info(f"Processing {mode} dataset: {source_dir}")
                    source_name = os.path.basename(os.path.dirname(source_dir))
                    dataset_stats[source_name] = {"files": 0}
                    
                    img_count = 0
                    img_paths = glob.glob(os.path.join(source_dir, "*.jpg")) + glob.glob(os.path.join(source_dir, "*.png"))
                    
                    for img in img_paths:
                        img = normalize_path(img)
                        # 创建唯一的目标文件名
                        base_name = os.path.basename(img)
                        img_ext = os.path.splitext(base_name)[1]
                        base_filename = os.path.splitext(base_name)[0]
                        
                        # 使用数据集名称和原始文件名创建新文件名
                        img_name = f"{source_name}_{base_filename}{img_ext}"
                        
                        # 确保文件名唯一
                        counter = 1
                        while img_name in existing_filenames:
                            img_name = f"{source_name}_{base_filename}_{counter}{img_ext}"
                            counter += 1
                        
                        existing_filenames.add(img_name)
                        target_path = normalize_path(os.path.join(merged_path, img_name))
                        
                        try:
                            # 复制文件
                            if not os.path.exists(target_path):
                                shutil.copy2(img, target_path)
                                img_count += 1
                        except Exception as e:
                            logger.error(f"Error copying {img} to {target_path}: {str(e)}")
                            continue
                    
                    if img_count > 0:
                        # 更新统计信息
                        dataset_stats[source_name]["files"] = img_count
                        total_merged_files += img_count
                        logger.info(f"  Copied {img_count} images")
            
            # 计算合并后的文件数并验证
            # 统计目标目录中的实际文件数
            actual_file_count = 0
            if mode == "train":
                for root, _, files in os.walk(merged_path):
                    actual_file_count += len([f for f in files if f.endswith(('.jpg', '.png'))])
            else:
                actual_file_count = len([f for f in os.listdir(merged_path) if f.endswith(('.jpg', '.png'))])
            
            # 验证合并后的文件数量
            if actual_file_count == 0:
                logger.error(f"ERROR: Merged directory is empty: {merged_path}")
                logger.error(f"Expected to merge {total_merged_files} files from {len(datasets)} datasets")
                logger.error(f"Source datasets: {datasets}")
                # 尝试查看目录是否真的创建了
                if not os.path.exists(merged_path):
                    logger.error(f"Merged directory does not exist: {merged_path}")
                return merged_path
            
            if actual_file_count != total_merged_files:
                logger.warning(f"Warning: File count mismatch - copied {total_merged_files} files, but found {actual_file_count} files in merged directory")
            
            # 详细的合并汇总
            logger.info(f"\n=== {mode.capitalize()} Dataset Merge Summary ===")
            logger.info(f"Merged directory: {merged_path}")
            logger.info(f"Total merged files: {actual_file_count}")
            logger.info(f"Source datasets: {len(datasets)}")
            for dataset_name, stats in dataset_stats.items():
                logger.info(f"  - {dataset_name}: {stats['files']} files")
                if 'classes' in stats and stats['classes']:
                    for class_name, count in stats['classes'].items():
                        if count > 0:
                            logger.info(f"    - Class {class_name}: {count} files")
            
            logger.info(f"{mode.capitalize()} dataset merge completed with {actual_file_count} total files")
            return merged_path
            
        except Exception as e:
            logger.error(f"Error during dataset merging: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # 尝试恢复 - 在严重错误时返回最合适的单一数据集
            if datasets:
                logger.info(f"Falling back to using a single dataset due to merge error")
                largest_dataset = max(datasets, key=lambda d: len(glob.glob(os.path.join(d, "**/*"), recursive=True)))
                largest_dataset = normalize_path(largest_dataset)
                logger.info(f"Selected largest {mode} dataset as fallback: {largest_dataset}")
                return largest_dataset
            else:
                # 如果没有数据集，返回默认路径
                if mode == "train":
                    return normalize_path(self.config.train_data)
                elif mode == "test":
                    return normalize_path(self.config.test_data)
                else:
                    return normalize_path(self.config.val_data)
    
    def list_available_datasets(self) -> Dict[str, List[str]]:
        """列出所有可用的数据集
        
        返回:
            字典，包含各类型的数据集路径列表
        """
        # 保存原始配置
        old_setting = self.config.merge_datasets
        self.config.merge_datasets = False
             
        # 列出数据集但不合并
        logger.info("Listing all available datasets:")
             
        datasets = {
            "train": handle_datasets(data_type="train", list_only=True),
            "test": handle_datasets(data_type="test", list_only=True),
            "val": handle_datasets(data_type="val", list_only=True)
        }
             
        # 恢复配置
        self.config.merge_datasets = old_setting
        
        return datasets

    def is_valid_dataset_directory(self, directory: str, dataset_type: str = 'train',
                          min_files: Optional[int] = None) -> bool:
        """检查目录是否包含足够的文件，可被视为有效的数据集
        
        参数:
            directory: 要检查的目录路径
            dataset_type: 数据集类型，'train', 'test' 或 'val'
            min_files: 最小文件数阈值，如果为None则使用配置中的min_files_threshold
        
        返回:
            是否是有效的数据集目录
        """
        if min_files is None:
            min_files = self.config.min_files_threshold
        
        directory = normalize_path(directory)
        
        if not os.path.exists(directory):
            logger.debug(f"Directory {directory} does not exist")
            return False
        
        file_count = 0
        
        try:
            if dataset_type == 'train':
                # 训练集需要检查类别目录
                class_dirs = [os.path.join(directory, d) for d in os.listdir(directory)
                             if os.path.isdir(os.path.join(directory, d))]
                
                if not class_dirs:
                    logger.debug(f"No class directories found in {directory}")
                    return False
                
                for class_dir in class_dirs:
                    file_count += len(glob.glob(os.path.join(class_dir, "*.*")))
            else:
                # 测试集和验证集直接计算根目录中的文件
                file_count = len(glob.glob(os.path.join(directory, "*.*")))
            
            logger.debug(f"Found {file_count} files in {directory}")
            return file_count >= min_files
        
        except Exception as e:
            logger.error(f"Error checking dataset directory {directory}: {str(e)}")
            return False

    def cleanup_temp_folders(self) -> None:
        """清理数据集解压后的临时文件夹，保留处理后的数据
        
        该方法会删除数据处理完成后不再需要的原始解压目录和临时文件夹，但保留已处理的数据
        在删除前会进行安全检查，确保数据已经被正确处理
        """
        # 规范化路径
        train_dir = normalize_path(self.paths.train_dir)
        
        # 安全检查：确保训练数据目录已经准备好
        train_glob_path = normalize_path(os.path.join(train_dir, '*/'))
        if not os.path.exists(train_dir) or len(glob.glob(train_glob_path)) < 10:
            logger.warning("Training data directory does not exist or contains too few class folders. Skipping cleanup for safety.")
            return
        
        # 安全检查：确保至少有一些图像已经被处理
        train_jpg_pattern = normalize_path(os.path.join(train_dir, '**/*.jpg'))
        train_image_count = len(glob.glob(train_jpg_pattern, recursive=True))
        if train_image_count < 100:  # 假设至少应该有100张图像
            logger.warning(f"Only {train_image_count} images found in training directory. Skipping cleanup for safety.")
            return
            
        data_dir = normalize_path(self.paths.data_dir)
        temp_dirs_to_remove = [
            normalize_path(os.path.join(data_dir, "AgriculturalDisease_trainingset")),
            normalize_path(os.path.join(data_dir, "AgriculturalDisease_validationset")),
            # 添加临时文件夹
            normalize_path(self.paths.temp_dir),
            normalize_path(self.paths.temp_images_dir),
            normalize_path(self.paths.temp_labels_dir)
        ]
        
        # 添加所有可能的测试集目录
        test_pattern = normalize_path(os.path.join(data_dir, "AgriculturalDisease_test*"))
        test_dirs = glob.glob(test_pattern)
        temp_dirs_to_remove.extend([normalize_path(d) for d in test_dirs])
        
        # 确保不会删除关键目录
        safe_dirs = [
            normalize_path(self.paths.train_dir),
            normalize_path(self.paths.test_dir),
            normalize_path(self.paths.val_dir),
            normalize_path(self.paths.merged_train_dir),
            normalize_path(self.paths.merged_test_dir),
            normalize_path(self.paths.merged_val_dir),
            normalize_path(self.paths.aug_dir)
        ]
        
        # 从删除列表中移除安全目录
        temp_dirs_to_remove = [normalize_path(dir) for dir in temp_dirs_to_remove if normalize_path(dir) not in safe_dirs]
        
        removed_count = 0
        for temp_dir in temp_dirs_to_remove:
            if os.path.exists(temp_dir):
                try:
                    # 使用shutil.rmtree递归删除目录
                    shutil.rmtree(temp_dir)
                    logger.info(f"Removed temporary directory: {temp_dir}")
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {temp_dir}: {str(e)}")
        
        if removed_count > 0:
            logger.info(f"Cleanup completed: removed {removed_count} temporary directories")
        else:
            logger.info("No temporary directories found to clean up")

    def find_dataset_file(self, filename, custom_path=None):
        """查找数据集文件
        
        参数:
            filename: 要查找的文件名
            custom_path: 自定义查找路径
            
        返回:
            找到的文件路径，未找到则返回None
        """
        found_path = None
        search_paths = []
        
        # 首先检查自定义路径
        if custom_path:
            custom_path = normalize_path(custom_path)
            search_paths.append(custom_path)
        
        # 添加数据目录下的所有可能的子目录
        search_paths.extend([
            normalize_path(self.paths.data_dir),
            normalize_path(os.path.join(self.paths.data_dir, "downloads")),
            normalize_path(os.path.join(self.paths.data_dir, "temp")),
        ])
        
        # 查找指定文件
        for path in search_paths:
            potential_path = normalize_path(os.path.join(path, filename))
            if os.path.exists(potential_path):
                found_path = potential_path
                logger.info(f"Found dataset file at: {found_path}")
                break
        
        # 如果还没找到，尝试通过通配符匹配
        if not found_path and "test" in filename:
            for path in search_paths:
                # 使用精确的通配符模式
                pattern = normalize_path(os.path.join(path, self.config.test_name_pattern))
                matches = glob.glob(pattern)
                if matches:
                    found_path = matches[0]
                    logger.info(f"Found test dataset file at: {found_path}")
                    break
        
        # 如果找到了文件
        if found_path:
            return found_path
        
        # 如果文件没有找到，记录日志并返回None
        logger.warning(f"Dataset file not found: {filename}")
        logger.info(f"Searched in paths: {', '.join(search_paths)}")
        return None

def setup_data(
    extract=True,
    process=True,
    augment=True,
    status=True,
    merge=None,
    cleanup_temp=False,
    custom_dataset_path=None,
    merge_augmented=None
):
    """初始化并运行数据准备任务
    
    参数:
        extract: 是否解压数据集
        process: 是否处理数据
        augment: 是否执行数据增强(仅应用于训练数据，测试数据永远不会被增强)
        status: 是否检查数据状态
        merge: 是否合并数据集 (可以是 'train', 'test', 'val', 'all' 或者包含这些值的列表)
        cleanup_temp: 是否清理临时文件夹
        custom_dataset_path: 自定义数据集路径
        merge_augmented: 是否合并增强数据
    
    返回:
        字典，包含操作结果和状态信息
    """
    # 初始化结果
    result = {
        "success": True,
        "errors": [],
        "warnings": [],
        "data_status": None
    }
    
    steps_completed = []
    
    # 保存当前配置的合并设置
    original_merge_augmented = config.merge_augmented_data
    original_merge_train = config.merge_train_datasets
    original_merge_val = config.merge_val_datasets
    original_merge_test = config.merge_test_datasets
    
    try:
        # 配置自定义数据集路径
        if custom_dataset_path:
            # 确保路径使用正确的格式
            custom_dataset_path = normalize_path(custom_dataset_path)
            config.dataset_path = custom_dataset_path
            config.use_custom_dataset_path = True
            logger.info(f"Using custom dataset path: {custom_dataset_path}")
        
        # 配置是否合并增强数据
        if merge_augmented is not None:
            config.merge_augmented_data = merge_augmented
            logger.info(f"Setting merge_augmented_data to: {merge_augmented}")
        
        # 初始化数据准备对象
        data_prep = DataPreparation()
        
        # 检查测试集数据是否存在，如果不存在则尝试解压
        test_images_path = normalize_path(paths.test_images_dir)
        if not os.path.exists(test_images_path) or len(glob.glob(os.path.join(test_images_path, "*.*"))) == 0:
            logger.warning(f"Test images directory does not exist or is empty: {test_images_path}")
            logger.info("Attempting to extract test dataset")
            
            # 尝试解压测试集，无论extract参数如何设置
            try:
                data_prep.extract_dataset(cleanup_temp=False)
                logger.info("Successfully extracted test dataset")
            except Exception as e:
                logger.error(f"Failed to extract test dataset: {str(e)}")
                result["errors"].append(f"Failed to extract test dataset: {str(e)}")
        else:
            logger.info(f"Test images directory exists with data: {test_images_path}")
        
        # 处理merge参数，支持字符串或列表
        merge_operations = []
        if merge:
            if isinstance(merge, str):
                if merge == "all":
                    merge_operations = ["train", "test", "val"]
                    # 设置所有合并标志
                    config.merge_train_datasets = True
                    config.merge_val_datasets = True
                    config.merge_test_datasets = True
                elif merge in ["train", "test", "val"]:
                    merge_operations = [merge]
                    # 设置对应的合并标志
                    if merge == "train":
                        config.merge_train_datasets = True
                    elif merge == "test":
                        config.merge_test_datasets = True
                    elif merge == "val":
                        config.merge_val_datasets = True
            elif isinstance(merge, (list, tuple)):
                merge_operations = [m for m in merge if m in ["train", "test", "val"]]
                # 设置对应的合并标志
                config.merge_train_datasets = "train" in merge_operations
                config.merge_val_datasets = "val" in merge_operations
                config.merge_test_datasets = "test" in merge_operations
        else:
            # 使用配置中的默认值
            if config.merge_train_datasets:
                merge_operations.append("train")
            if config.merge_val_datasets:
                merge_operations.append("val")
            if config.merge_test_datasets:
                merge_operations.append("test")
        
        # 检查最终数据集路径是否已经存在数据
        # 这将基于各种配置判断最终的数据会位于哪里
        final_paths = {}
        
        # 根据合并设置确定最终会使用的数据路径
        if "train" in merge_operations:
            final_paths["train"] = normalize_path(paths.merged_train_dir)
        else:
            final_paths["train"] = normalize_path(paths.train_dir)
            
        if "val" in merge_operations:
            final_paths["val"] = normalize_path(paths.merged_val_dir)
        else:
            final_paths["val"] = normalize_path(paths.val_dir)
            
        if "test" in merge_operations:
            final_paths["test"] = normalize_path(paths.merged_test_dir)
        else:
            final_paths["test"] = normalize_path(paths.test_dir)
        
        # 检查这些路径中是否有足够的数据
        data_exists = True
        min_file_threshold = config.min_files_threshold  # 使用配置中的阈值
        data_counts = {}
        
        for dataset_type, path in final_paths.items():
            if os.path.exists(path):
                file_count = 0
                # 对于训练集，检查每个类别目录中的文件
                if dataset_type == "train":
                    if os.path.isdir(path):
                        class_dirs = [normalize_path(os.path.join(path, d)) 
                                     for d in os.listdir(path) 
                                     if os.path.isdir(os.path.join(path, d))]
                        for class_dir in class_dirs:
                            file_count += len(glob.glob(os.path.join(class_dir, "*.*")))
                else:
                    # 对于测试集和验证集，直接计算根目录中的文件
                    file_count = len(glob.glob(os.path.join(path, "*.*")))
                
                data_counts[dataset_type] = file_count
                if dataset_type == "train" and file_count < min_file_threshold:
                    data_exists = False
            else:
                data_exists = False
                data_counts[dataset_type] = 0
        
        # 如果启用了数据增强，检查增强数据目录
        if augment and config.use_data_aug:
            aug_path = normalize_path(paths.augmented_images_dir)
            if os.path.exists(aug_path):
                aug_file_count = len(glob.glob(os.path.join(aug_path, "**/*.*"), recursive=True))
                data_counts["augmentation"] = aug_file_count
            else:
                data_counts["augmentation"] = 0
        
        # 确保测试数据路径存在，这对推理很重要
        test_data_path = final_paths["test"]
        if not os.path.exists(test_data_path):
            os.makedirs(test_data_path, exist_ok=True)
            logger.info(f"Created test data directory: {test_data_path}")
        elif not os.path.isdir(test_data_path):
            logger.error(f"Test data path exists but is not a directory: {test_data_path}")
            result["errors"].append(f"Test data path exists but is not a directory: {test_data_path}")
            result["success"] = False
        
        # 记录数据状态
        logger.info("Checking existing data:")
        for dataset_type, count in data_counts.items():
            logger.info(f"  {dataset_type}: {count} files")
        
        # 如果所有必要的数据都已存在，询问是否应该跳过处理
        if data_exists and "train" in data_counts and data_counts["train"] >= min_file_threshold:
            logger.info(f"Found existing dataset with sufficient data (train: {data_counts['train']} files)")
            if not config.force_data_processing:
                logger.info("Skipping data processing as valid datasets already exist. Set force_data_processing=True in config to override.")
                
                # 只执行状态检查
                if status:
                    logger.info("Checking data status...")
                    try:
                        data_status = data_prep.get_data_status()
                        result["data_status"] = data_status
                        data_prep.check_data_status()
                    except Exception as e:
                        logger.error(f"Error checking data status: {str(e)}")
                
                return result
            else:
                logger.info("Force data processing enabled. Proceeding with data preparation despite existing datasets.")
        
        # Extract dataset
        if extract:
            logger.info("Extracting dataset...")
            try:
                data_prep.extract_dataset(cleanup_temp=False)  # 先不清理临时文件，后面可能还需要用
                
                # 验证提取是否成功
                training_data = paths.train_annotations
                validation_data = paths.val_annotations
                
                # 规范化路径
                training_data = normalize_path(training_data)
                validation_data = normalize_path(validation_data)
                
                training_path = None
                validation_path = None
                
                if os.path.exists(training_data):
                    training_path = training_data
                    logger.info(f"Found training annotations: {training_path}")
                else:
                    logger.warning("Training dataset file not found. Skipping extraction.")
                    result["warnings"].append("Training dataset file not found. Skipping extraction.")
                
                if os.path.exists(validation_data):
                    validation_path = validation_data
                    logger.info(f"Found validation annotations: {validation_path}")
                else:
                    logger.warning("Validation dataset file not found. Skipping extraction.")
                    result["warnings"].append("Validation dataset file not found. Skipping extraction.")
                
                if not training_path and not validation_path:
                    # Check if we already have extracted data
                    temp_dataset_dir = normalize_path(paths.temp_dataset_dir)
                    if os.path.exists(temp_dataset_dir) and os.listdir(temp_dataset_dir):
                        logger.info(f"No dataset files found but temp directory exists with data, continuing with existing data")
                    else:
                        logger.error("No dataset files found and no existing extracted data. Cannot proceed with data preparation.")
                        result["errors"].append("No dataset files found and no existing extracted data.")
                        result["success"] = False
                
            except Exception as e:
                logger.error(f"Error during dataset extraction: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                result["errors"].append(f"Error during dataset extraction: {str(e)}")
                result["success"] = False
            
            steps_completed.append("extract")
        
        # Process data
        if process and result["success"]:
            # 检查处理后的数据是否已存在
            train_processed = os.path.exists(paths.train_dir) and len(glob.glob(os.path.join(paths.train_dir, "**/*.*"), recursive=True)) > min_file_threshold
            val_processed = os.path.exists(paths.val_dir) and len(glob.glob(os.path.join(paths.val_dir, "**/*.*"), recursive=True)) > 0
            
            if train_processed and val_processed and not config.force_data_processing:
                logger.info(f"Processed data already exists. Skipping processing step.")
                steps_completed.append("process")
            else:
                logger.info("Processing data...")
                try:
                    data_prep.process_data()
                    steps_completed.append("process")
                except Exception as e:
                    logger.error(f"Error during data processing: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    result["errors"].append(f"Error during data processing: {str(e)}")
                    result["success"] = False
        
        # Augment data (only for training data, NEVER for test data)
        if augment and result["success"] and config.use_data_aug:
            # 检查增强数据是否已存在
            aug_dir = normalize_path(paths.augmented_images_dir)
            aug_exists = os.path.exists(aug_dir) and len(glob.glob(os.path.join(aug_dir, "**/*.*"), recursive=True)) > min_file_threshold
            
            if aug_exists and not config.force_augmentation:
                logger.info(f"Augmented data already exists with {data_counts.get('augmentation', 0)} files. Skipping augmentation step.")
                steps_completed.append("augment")
            else:
                logger.info("Augmenting training data (test data will NOT be augmented)...")
                try:
                    # Check if we have enough training data first
                    data_status = data_prep.get_data_status()
                    
                    if data_status.get("training", {}).get("total_images", 0) > 0:
                        train_path = normalize_path(os.path.join(paths.train_dir))
                        augmented_dir = normalize_path(paths.augmented_images_dir)
                        
                        # Create directory if it doesn't exist
                        if not os.path.exists(augmented_dir):
                            os.makedirs(augmented_dir, exist_ok=True)
                        
                        logger.info(f"Running data augmentation on {train_path}...")
                        augmented_files = data_prep.augment_directory(train_path, augmented_dir)
                        logger.info(f"Generated {len(augmented_files)} augmented images")
                        
                        steps_completed.append("augment")
                    else:
                        logger.warning("No training images found, skipping augmentation")
                        result["warnings"].append("No training images found, skipping augmentation")
                except KeyboardInterrupt:
                    logger.warning("Data augmentation interrupted by user. Continuing with partial results.")
                    result["warnings"].append("Data augmentation was interrupted.")
                    steps_completed.append("augment")  # 仍然标记为已完成，因为它可能已经生成了一些有效的增强图像
                except Exception as e:
                    logger.error(f"Error during data augmentation: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    result["warnings"].append(f"Error during data augmentation: {str(e)}")
        
        # 如果需要合并增强数据并且已经完成了增强，确保训练数据在合并列表中
        if config.merge_augmented_data and "augment" in steps_completed:
            if "train" not in merge_operations:
                merge_operations.append("train")
                config.merge_train_datasets = True
                logger.info("Added training data to merge operations because augmented data needs to be merged")
        
        # 执行合并操作
        if merge_operations and result["success"]:
            merged_paths = {}
            for mode in merge_operations:
                merged_path = None
                if mode == "train":
                    merged_path = normalize_path(paths.merged_train_dir)
                elif mode == "test":
                    merged_path = normalize_path(paths.merged_test_dir)
                elif mode == "val":
                    merged_path = normalize_path(paths.merged_val_dir)
                
                # 检查合并目录是否已经包含足够的数据
                if os.path.exists(merged_path):
                    file_count = 0
                    if mode == "train":
                        # 训练集需要检查类别目录
                        class_dirs = [os.path.join(merged_path, d) for d in os.listdir(merged_path) 
                                     if os.path.isdir(os.path.join(merged_path, d))]
                        for class_dir in class_dirs:
                            file_count += len(glob.glob(os.path.join(class_dir, "*.*")))
                    else:
                        # 测试集和验证集直接计算
                        file_count = len(glob.glob(os.path.join(merged_path, "*.*")))
                    
                    merged_paths[mode] = (merged_path, file_count)
                else:
                    merged_paths[mode] = (merged_path, 0)
            
            # 判断哪些路径需要合并
            need_merge = []
            for mode, (path, count) in merged_paths.items():
                if count < min_file_threshold or config.force_merge:
                    need_merge.append(mode)
                else:
                    logger.info(f"Merged {mode} data already exists with {count} files. Skipping merge.")
            
            if need_merge:
                logger.info(f"Merging datasets: {', '.join(need_merge)}")
                try:
                    for mode in need_merge:
                        # 对于训练集，先处理是否需要合并增强数据
                        if mode == "train" and config.merge_augmented_data and "augment" in steps_completed:
                            logger.info("Merging augmented data with training data...")
                        
                        # 执行合并操作
                        try:
                            merged_path = data_prep.merge_datasets(mode, force=config.force_merge)
                            logger.info(f"Merged {mode} datasets to: {merged_path}")
                        except KeyboardInterrupt:
                            logger.warning(f"Merging {mode} datasets interrupted by user")
                            result["warnings"].append(f"Merging {mode} datasets was interrupted")
                            continue
                    
                    steps_completed.append("merge")
                except Exception as e:
                    logger.error(f"Error during dataset merging: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    result["warnings"].append(f"Error during dataset merging: {str(e)}")
            elif merge_operations:
                logger.info("All merged datasets already exist with sufficient data. Skipping merge operations.")
                steps_completed.append("merge")
        
        # Check data status if requested
        if status:
            logger.info("Checking data status...")
            try:
                data_status = data_prep.get_data_status()
                result["data_status"] = data_status
                
                # Check for potential issues
                if data_status.get("training", {}).get("total_images", 0) < 100:
                    logger.warning("Very few training images detected (<100). This may lead to poor model performance.")
                    result["warnings"].append("Very few training images detected (<100). This may lead to poor model performance.")
                
                if data_status.get("training", {}).get("total_images", 0) == 0:
                    logger.error("No training images found. Cannot train the model.")
                    result["errors"].append("No training images found. Cannot train the model.")
                    result["success"] = False
                
                # Print the status details
                data_prep.check_data_status()
                
                # 确保测试数据可用
                testing_status = data_status.get("testing", {})
                if testing_status.get("total_images", 0) == 0:
                    logger.warning("No test images found. Inference will not be possible without test data.")
                    result["warnings"].append("No test images found. Inference will not be possible without test data.")
                else:
                    logger.info(f"Found {testing_status.get('total_images', 0)} test images ready for inference.")
                
                # 限制检查无效图像的范围，避免处理过多图像
                logger.info("Checking for corrupted images...")
                
                # 收集哪些目录需要检查
                dirs_to_check = []
                for dataset_type in ["training", "validation", "testing"]:
                    if dataset_type in data_status and "directory" in data_status[dataset_type]:
                        directory = normalize_path(data_status[dataset_type]["directory"])
                        if os.path.exists(directory):
                            # 获取目录中的图像数量
                            img_count = data_status[dataset_type].get("total_images", 0)
                            if img_count > 0:
                                dirs_to_check.append((directory, img_count, dataset_type))
                
                # 按照图像数量从小到大排序，优先处理小目录
                dirs_to_check.sort(key=lambda x: x[1])
                
                # 只检查实际会用到的目录
                # 根据合并设置和是否使用增强数据来确定最终会使用的目录
                final_dirs = set()
                if "merge" in steps_completed:
                    for mode in merge_operations:
                        if mode == "train":
                            final_dirs.add(normalize_path(paths.merged_train_dir))
                        elif mode == "test":
                            final_dirs.add(normalize_path(paths.merged_test_dir))
                        elif mode == "val":
                            final_dirs.add(normalize_path(paths.merged_val_dir))
                else:
                    final_dirs.add(normalize_path(paths.train_dir))
                    final_dirs.add(normalize_path(paths.val_dir))
                    if os.path.exists(normalize_path(paths.test_dir)):
                        final_dirs.add(normalize_path(paths.test_dir))
                
                # 只检查最终会使用的目录中的图像
                dirs_to_actually_check = []
                for directory, img_count, dataset_type in dirs_to_check:
                    if directory in final_dirs:
                        logger.info(f"Checking {dataset_type} directory with {img_count} images")
                        dirs_to_actually_check.append(directory)
                
                for directory in dirs_to_actually_check:
                    data_prep.remove_error_images(directory)
                
            except Exception as e:
                logger.error(f"Error during data status check: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                result["warnings"].append(f"Error during data status check: {str(e)}")
        
        # 清理临时文件夹
        if cleanup_temp and result["success"]:
            logger.info("Cleaning up temporary folders...")
            try:
                data_prep.cleanup_temp_folders()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                result["warnings"].append(f"Error during cleanup: {str(e)}")
        
        # 恢复原始合并设置
        config.merge_augmented_data = original_merge_augmented
        config.merge_train_datasets = original_merge_train
        config.merge_val_datasets = original_merge_val
        config.merge_test_datasets = original_merge_test
        
        return result
    except Exception as e:
        logger.error(f"Unexpected error during data preparation: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        result["errors"].append(f"Unexpected error: {str(e)}")
        result["success"] = False
        
        # 恢复原始合并设置
        config.merge_augmented_data = original_merge_augmented
        config.merge_train_datasets = original_merge_train
        config.merge_val_datasets = original_merge_val
        config.merge_test_datasets = original_merge_test
        
        return result

def extract_dataset(dataset_path, extract_to):
    """提取数据集到指定目录
    
    参数:
        dataset_path: 数据集文件路径
        extract_to: 提取目标路径
        
    返回:
        是否成功提取
    """
        # 规范化路径
    dataset_path = normalize_path(dataset_path)
    extract_to = normalize_path(extract_to)
    
    logger.info(f"Extracting dataset from {dataset_path} to {extract_to}")
    
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file does not exist: {dataset_path}")
        return False
    
    # 确保目标目录存在
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        # 确定文件类型并使用相应的提取器
        if dataset_path.endswith('.zip'):
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                # 获取压缩包内的文件列表
                file_count = len(zip_ref.namelist())
                logger.info(f"Found {file_count} files in zip archive")
                
                # 使用tqdm显示提取进度
                with tqdm(total=file_count, desc="Extracting zip", unit="files") as pbar:
                    for file in zip_ref.namelist():
                        zip_ref.extract(file, extract_to)
                        pbar.update(1)
        
        elif dataset_path.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(dataset_path, 'r:*') as tar_ref:
                members = tar_ref.getmembers()
                file_count = len(members)
                logger.info(f"Found {file_count} files in tar archive")
                
                # 使用tqdm显示提取进度
                with tqdm(total=file_count, desc="Extracting tar", unit="files") as pbar:
                    for member in members:
                        tar_ref.extract(member, extract_to)
                        pbar.update(1)
        
        else:
            logger.error(f"Unsupported file format: {dataset_path}")
            return False
        
        logger.info(f"Successfully extracted dataset to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"Error extracting dataset: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def merge_datasets_cli():
    """数据集合并命令行界面，提供与原merge_datasets.py相同的功能"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset merging tool")
    parser.add_argument('--train', action='store_true', help='Merge training datasets')
    parser.add_argument('--test', action='store_true', help='Merge test datasets')
    parser.add_argument('--val', action='store_true', help='Merge validation datasets')
    parser.add_argument('--all', action='store_true', help='Merge all datasets')
    parser.add_argument('--list', action='store_true', help='Only list available datasets without merging')
    parser.add_argument('--force', action='store_true', help='Force create new merged directory, overwriting existing one')
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，显示帮助信息
    if not (args.train or args.test or args.val or args.all or args.list):
        parser.print_help()
        return
    
    data_prep = DataPreparation()
    
    # 处理不同类型的请求
    if args.list:
        # 只列出数据集
        datasets = data_prep.list_available_datasets()
        for mode, dataset_list in datasets.items():
            logger.info(f"\n{mode.upper()} datasets:")
            for i, dataset in enumerate(dataset_list, 1):
                logger.info(f"  {i}. {dataset}")
    else:
        # 合并数据集
        if args.train or args.all:
            train_path = data_prep.merge_datasets("train", force=args.force)
            logger.info(f"Training dataset merge completed: {train_path}")
        
        if args.test or args.all:
            test_path = data_prep.merge_datasets("test", force=args.force)
            logger.info(f"Test dataset merge completed: {test_path}")
        
        if args.val or args.all:
            val_path = data_prep.merge_datasets("val", force=args.force)
            logger.info(f"Validation dataset merge completed: {val_path}")
        
        logger.info("Dataset merging process completed successfully.")

if __name__ == "__main__":
    import sys
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Data preparation utility for Plant Disease Detection")
    parser.add_argument('--extract', action='store_true', help='Extract dataset archives')
    parser.add_argument('--process', action='store_true', help='Process extracted data')
    parser.add_argument('--augment', action='store_true', help='Run data augmentation')
    parser.add_argument('--status', action='store_true', help='Check data preparation status')
    parser.add_argument('--merge', action='store_true', help='Merge available datasets')
    parser.add_argument('--force', action='store_true', help='Force overwrite existing merged directories')
    parser.add_argument('--test', action='store_true', help='Test data preparation functionality')
    parser.add_argument('--disable-merge', action='store_true', help='Disable dataset merging')
    parser.add_argument('--all', action='store_true', help='Run all data preparation steps')
    parser.add_argument('--custom-dataset', type=str, help='Specify custom dataset path')
    
    args = parser.parse_args()
    
    # 如果是测试模式
    if args.test:
        logger.info("Running in test mode")
        # 临时保存配置
        old_force = config.merge_force
        old_merge = config.merge_datasets
        
        # 更新配置
        if args.force:
            config.merge_force = True
            logger.info("Force merge enabled for testing")
        if args.disable_merge:
            config.merge_datasets = False
            logger.info("Dataset merging disabled for testing")
        elif args.merge:
            config.merge_datasets = True
            logger.info("Dataset merging enabled for testing")
            
        # 创建数据准备对象并测试状态
        data_prep = DataPreparation()
        data_prep.check_data_status()
        
        # 如果启用了合并，测试合并功能
        if config.merge_datasets:
            logger.info("\nTesting dataset merging...")
            train_path = data_prep.merge_datasets("train", force=args.force)
            logger.info(f"Merged training dataset path: {train_path}")
            
            test_path = data_prep.merge_datasets("test", force=args.force)
            logger.info(f"Merged test dataset path: {test_path}")
            
            if os.path.exists(paths.val_dir):
                val_path = data_prep.merge_datasets("val", force=args.force)
                logger.info(f"Merged validation dataset path: {val_path}")
        
        # 恢复配置
        config.merge_force = old_force
        config.merge_datasets = old_merge
        logger.info("Test completed")
    
    # 如果指定了特定任务
    elif args.extract or args.process or args.augment or args.status or args.merge or args.all:
        # 确定要执行的任务
        do_extract = args.extract or args.all
        do_process = args.process or args.all
        do_augment = args.augment or args.all
        do_status = args.status or args.all or True  # 默认始终执行状态检查
        do_merge = args.merge or args.all
        
        # 如果禁用合并
        if args.disable_merge:
            config.merge_datasets = False
            do_merge = False
            logger.info("Dataset merging disabled")
        
        # 如果强制合并
        if args.force:
            config.merge_force = True
            logger.info("Force merge enabled")
        
        # 执行数据准备
        setup_data(
            extract=do_extract,
            process=do_process,
            augment=do_augment, 
            status=do_status,
            merge=do_merge,
            custom_dataset_path=args.custom_dataset
        )
    
    # 如果存在dataset_merge命令行参数
    elif len(sys.argv) > 1 and sys.argv[1] in ["--train", "--test", "--val", "--all", "--list", "--force"]:
        merge_datasets_cli()
    
    # 默认情况下，执行完整流程
    else:
        custom_dataset_path = args.custom_dataset if hasattr(args, 'custom_dataset') else None
        setup_data(extract=True, process=True, augment=True, status=True, merge=None, custom_dataset_path=custom_dataset_path) 