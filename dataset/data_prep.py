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
            origin_path = os.path.join(self.paths.data_dir, "temp", "images", filename)
            
            # 跳过类别44和45
            ids = file["disease_class"]
            if ids == 44 or ids == 45:
                return False
                
            # 调整ID顺序（大于45的ID要减2）
            if ids > 45:
                ids -= 2
                
            # 确保目标目录存在
            save_dir = os.path.join(self.paths.data_dir, "train", str(ids))
            os.makedirs(save_dir, exist_ok=True)
            
            # 构建目标路径并复制文件
            save_path = os.path.join(save_dir, filename)
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
        data_dir = self.paths.data_dir
        custom_dataset_path = self.config.dataset_path if self.config.use_custom_dataset_path else None
        
        # 如果配置了自定义数据集路径，使用自定义路径
        if custom_dataset_path:
            source_dir = custom_dataset_path
            logger.info(f"Using custom dataset path: {source_dir}")
            
            # 检查自定义路径是否存在
            if not os.path.exists(source_dir):
                logger.error(f"Custom dataset path does not exist: {source_dir}")
                return
                
            # 从自定义路径查找数据集文件
            train_zip = None
            val_zip = None
            test_zips = []
            
            # 递归搜索支持的文件格式
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_lower = file.lower()
                    
                    # 检查是否是支持的数据集格式
                    is_supported = any(file_lower.endswith(ext) for ext in self.config.supported_dataset_formats)
                    if not is_supported:
                        continue
                        
                    # 识别数据集类型
                    if self.config.training_dataset_file in file:
                        train_zip = file_path
                        logger.info(f"Found training dataset: {file_path}")
                    elif self.config.validation_dataset_file in file:
                        val_zip = file_path
                        logger.info(f"Found validation dataset: {file_path}")
                    elif "test" in file_lower:
                        test_zips.append(file_path)
                        logger.info(f"Found test dataset: {file_path}")
            
            # 如果没有找到数据集文件，尝试复制其他格式文件
            if not (train_zip or val_zip or test_zips):
                logger.warning(f"No dataset files found in custom path: {source_dir}")
                logger.info(f"Looking for other supported formats: {self.config.supported_dataset_formats}")
                
                # 复制其他格式的数据文件
                copied_files = 0
                for root, _, files in os.walk(source_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_ext = os.path.splitext(file)[1].lower()
                        
                        # 检查是否是图像文件
                        if file_ext in ['.jpg', '.jpeg', '.png']:
                            if "train" in root.lower():
                                # 提取类别信息
                                try:
                                    rel_path = os.path.relpath(root, source_dir)
                                    class_parts = [p for p in rel_path.split(os.sep) if p.isdigit()]
                                    if class_parts:
                                        class_id = int(class_parts[0])
                                        target_dir = os.path.join(self.paths.train_dir, str(class_id))
                                        os.makedirs(target_dir, exist_ok=True)
                                        shutil.copy2(file_path, os.path.join(target_dir, file))
                                        copied_files += 1
                                except Exception as e:
                                    logger.error(f"Error copying training image: {str(e)}")
                            elif "test" in root.lower():
                                # 复制到测试目录
                                try:
                                    shutil.copy2(file_path, os.path.join(self.paths.test_images_dir, file))
                                    copied_files += 1
                                except Exception as e:
                                    logger.error(f"Error copying test image: {str(e)}")
                
                if copied_files > 0:
                    logger.info(f"Copied {copied_files} image files directly from custom dataset path")
                    return
                else:
                    logger.error("No valid dataset files or images found in custom path")
                    return
        else:
            # 使用默认路径
            source_dir = data_dir
            train_zip = os.path.join(data_dir, self.config.training_dataset_file)
            val_zip = os.path.join(data_dir, self.config.validation_dataset_file)
            test_zips = glob.glob(os.path.join(data_dir, self.config.test_name_pattern))
            logger.info(f"Using default dataset path: {source_dir}")
        
        # 检查并解压训练集
        if train_zip and os.path.exists(train_zip):
            self.extract_zip_file(train_zip, data_dir)
            train_dir = os.path.join(data_dir, "AgriculturalDisease_trainingset")
            
            # 复制训练图片到临时目录
            images_dir = os.path.join(train_dir, "images")
            if os.path.exists(images_dir):
                self.copy_files_to_folder(images_dir, self.paths.temp_images_dir, "*.jpg")
            
            # 复制训练标注文件到标签目录
            annot_file = os.path.join(train_dir, "AgriculturalDisease_train_annotations.json")
            if os.path.exists(annot_file):
                shutil.copy2(annot_file, self.paths.temp_labels_dir)
                logger.info(f"Copied training annotations file")
        else:
            logger.warning(f"Training dataset zip not found: {train_zip if train_zip else 'No file specified'}")
        
        # 检查并解压验证集
        if val_zip and os.path.exists(val_zip):
            self.extract_zip_file(val_zip, data_dir)
            val_dir = os.path.join(data_dir, "AgriculturalDisease_validationset")
            
            # 复制验证图片到临时目录
            images_dir = os.path.join(val_dir, "images")
            if os.path.exists(images_dir):
                self.copy_files_to_folder(images_dir, self.paths.temp_images_dir, "*.jpg")
            
            # 复制验证标注文件到标签目录
            annot_file = os.path.join(val_dir, "AgriculturalDisease_validation_annotations.json")
            if os.path.exists(annot_file):
                shutil.copy2(annot_file, self.paths.temp_labels_dir)
                logger.info(f"Copied validation annotations file")
        else:
            logger.warning(f"Validation dataset zip not found: {val_zip if val_zip else 'No file specified'}")
        
        # 检查并解压所有测试集
        if test_zips:
            logger.info(f"Found {len(test_zips)} test dataset zip files")
            
            for test_zip in test_zips:
                # 创建文件名的唯一目录名称
                zip_basename = os.path.basename(test_zip)
                
                # 尝试提取test数据集标识符
                try:
                    test_dir_suffix = zip_basename.split('_')[3].split('.')[0]  # 从文件名提取 testa 或 testb 部分
                except:
                    # 如果无法提取，使用文件名作为标识符
                    test_dir_suffix = os.path.splitext(zip_basename)[0]
                    
                extracted_dir = os.path.join(data_dir, f"AgriculturalDisease_{test_dir_suffix}")
                
                # 解压测试集
                self.extract_zip_file(test_zip, data_dir)
                
                # 为每个测试集创建单独的目录
                test_specific_dir = os.path.join(self.paths.test_dir, test_dir_suffix)
                os.makedirs(test_specific_dir, exist_ok=True)
                
                # 复制测试图片到对应的测试目录
                images_dir = os.path.join(extracted_dir, "images")
                if os.path.exists(images_dir):
                    self.copy_files_to_folder(images_dir, test_specific_dir, "*.jpg")
                    
                    # 同时复制到通用测试目录，以保持向后兼容性
                    if self.config.duplicate_test_to_common:
                        self.copy_files_to_folder(images_dir, self.paths.test_images_dir, "*.jpg")
            
            # 准备合并测试集，如果配置中启用
            if self.config.merge_datasets:
                logger.info("Merging test datasets as per configuration")
                self.merge_datasets(mode="test", force=True)
        else:
            logger.warning(f"No test dataset zip files found")
            
        # 确保测试目录存在
        os.makedirs(self.paths.test_images_dir, exist_ok=True)
        
        # 如果启用了清理选项，删除临时解压目录
        if cleanup_temp:
            self.cleanup_temp_folders()

    def process_data(self) -> None:
        """处理数据集文件并组织到训练目录"""
        # 确保目录结构存在
        self.setup_directories()
        
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
            布尔值，指示图像是否有效
        """
        try:
            # 尝试用PIL和OpenCV打开图像
            img = Image.open(image_path)
            img.verify()
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                raise Exception("OpenCV cannot read image")
                
            # 检查图像尺寸
            if cv_image.shape[0] < 10 or cv_image.shape[1] < 10:
                raise Exception("Image is too small")
                
            # 检查是否为空图像
            if cv_image.size == 0:
                raise Exception("Empty image")
                
            # 检查是否为全黑或全白图像
            if np.mean(cv_image) < 1 or np.mean(cv_image) > 254:
                raise Exception("Image is mostly black or white")
                
            return True
        except Exception as e:
            logger.error(f"Invalid image {image_path}: {str(e)}")
            self.error_images.append(image_path)
            return False

    def remove_error_images(self, image_dir: str) -> None:
        """删除错误的图像文件
        
        参数:
            image_dir: 图像目录路径
        """
        if not self.config.remove_error_images:
            logger.info("Skip removing error images (disabled in config)")
            return

        logger.info("Checking for invalid images...")
        
        # 收集需要检查的图像文件
        image_files = []
        for image_path in Path(image_dir).rglob('*'):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_files.append(str(image_path))
        
        if not image_files:
            logger.info("No images found to check")
            return
        
        # 初始化计数器和结果列表
        total_files = len(image_files)
        invalid_images = []
        error_count = 0
        
        # 定义进度条更新函数
        def check_image_batch(batch):
            batch_invalid = []
            for image_path in batch:
                if not self.is_valid_image(image_path):
                    batch_invalid.append(image_path)
            return batch_invalid
        
        # 使用多线程并行验证图像
        batch_size = 20  # 每个线程处理的批次大小
        batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.config.aug_max_workers) as executor:
            futures = []
            for batch in batches:
                futures.append(executor.submit(check_image_batch, batch))
            
            # 使用tqdm显示进度条
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(futures), 
                              desc="Validating images", 
                              unit="batch"):
                batch_invalid = future.result()
                invalid_images.extend(batch_invalid)
        
        # 删除无效图像
        if invalid_images:
            logger.info(f"Found {len(invalid_images)} invalid images, removing...")
            removed_count = 0
            
            for invalid_path in tqdm(invalid_images, desc="Removing invalid images", unit="img"):
                try:
                    os.remove(invalid_path)
                    removed_count += 1
                except Exception as e:
                    error_count += 1
                    logger.error(f"Failed to remove {invalid_path}: {str(e)}")
            
            logger.info(f"Removed {removed_count}/{len(invalid_images)} invalid images")
        else:
            logger.info("No invalid images found")
        
        if error_count > 0:
            logger.warning(f"Failed to remove {error_count} invalid images")

    def augment_image(self, image_path: str, save_dir: str) -> List[str]:
        """对单张图像进行数据增强
        
        参数:
            image_path: 输入图像路径
            save_dir: 保存目录
            
        返回:
            增强后的图像路径列表
        """
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image path does not exist: {image_path}")
                return []

            img = Image.open(image_path)
            cv_image = cv2.imread(image_path)
            
            if cv_image is None:
                logger.warning(f"Failed to read image: {image_path}")
                return []

            basename = os.path.basename(image_path)
            augmented_images = []

            # 添加高斯噪声
            if self.config.aug_noise:
                gau_image = self.add_noise(cv_image)
                if gau_image is not None:
                    output_path = os.path.join(save_dir, f"gau_{basename}")
                    cv2.imwrite(output_path, gau_image)
                    augmented_images.append(output_path)

            # 调整亮度
            if self.config.aug_brightness:
                light_image = self.change_brightness(cv_image)
                if light_image is not None:
                    output_path = os.path.join(save_dir, f"light_{basename}")
                    cv2.imwrite(output_path, light_image)
                    augmented_images.append(output_path)

            # 翻转
            if self.config.aug_flip:
                img_flip_left_right = img.transpose(Image.FLIP_LEFT_RIGHT)
                img_flip_top_bottom = img.transpose(Image.FLIP_TOP_BOTTOM)
                
                lr_path = os.path.join(save_dir, f"left_right_{basename}")
                tb_path = os.path.join(save_dir, f"top_bottom_{basename}")
                
                img_flip_left_right.save(lr_path)
                img_flip_top_bottom.save(tb_path)
                
                augmented_images.extend([lr_path, tb_path])

            # 对比度增强
            if self.config.aug_contrast:
                enh_con = ImageEnhance.Contrast(img)
                image_contrasted = enh_con.enhance(self.config.aug_contrast_factor)
                output_path = os.path.join(save_dir, f"contrasted_{basename}")
                image_contrasted.save(output_path)
                augmented_images.append(output_path)

            # 应用高级增强
            advanced_image = self.apply_advanced_augmentation(cv_image)
            if advanced_image is not None:
                output_path = os.path.join(save_dir, f"advanced_{basename}")
                cv2.imwrite(output_path, advanced_image)
                augmented_images.append(output_path)

            return augmented_images
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
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
        
        os.makedirs(target_dir, exist_ok=True)
        augmented_files = []
        
        # 首先删除错误图片
        if self.config.remove_error_images:
            self.remove_error_images(source_dir)
        
        # 获取所有图片文件
        image_files = []
        for image_path in Path(source_dir).rglob('*'):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_files.append(str(image_path))

        logger.info(f"Found {len(image_files)} images for augmentation")
        
        with ThreadPoolExecutor(max_workers=self.config.aug_max_workers) as executor:
            futures = []
            for image_path in image_files:
                relative_path = os.path.relpath(image_path, source_dir)
                save_dir = os.path.join(target_dir, os.path.dirname(relative_path))
                os.makedirs(save_dir, exist_ok=True)
                futures.append(executor.submit(self.augment_image, image_path, save_dir))

            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures),
                             desc="Processing images"):
                try:
                    result = future.result()
                    if result:
                        augmented_files.extend(result)
                except Exception as e:
                    logger.error(f"Error in augmentation: {str(e)}")

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
            merged_path = self.paths.merged_train_dir
        elif mode == "test":
            merged_path = self.paths.merged_test_dir
        elif mode == "val":
            merged_path = self.paths.merged_val_dir
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'test', or 'val'")
        
        os.makedirs(merged_path, exist_ok=True)
        
        # 如果强制模式，清空目标目录
        if force and os.path.exists(merged_path):
            logger.info(f"Force flag set. Removing existing merged directory: {merged_path}")
            shutil.rmtree(merged_path)
            os.makedirs(merged_path, exist_ok=True)
        
        # 检查目标目录是否已经有文件，如果有则跳过合并
        if not force and os.path.exists(merged_path):
            file_count = 0
            # 对于训练集，检查每个类别目录中的文件
            if mode == "train":
                class_dirs = [os.path.join(merged_path, d) for d in os.listdir(merged_path) if os.path.isdir(os.path.join(merged_path, d))]
                for class_dir in class_dirs:
                    file_count += len(glob.glob(os.path.join(class_dir, "*.*")))
            else:
                # 对于测试集和验证集，直接计算根目录中的文件
                file_count = len(glob.glob(os.path.join(merged_path, "*.*")))
            
            if file_count > 0:
                logger.info(f"Merged {mode} directory already contains {file_count} files. Use force=True to overwrite.")
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
                return self.config.train_data
            elif mode == "test":
                return self.config.test_data
            else:
                return self.config.val_data
        
        # 添加增强数据目录(仅训练集)
        if mode == "train" and self.config.use_data_aug and self.config.merge_augmented_data:
            aug_dir = self.paths.aug_train_dir
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
            return datasets[0]
        
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
                    logger.info(f"Processing training dataset: {source_dir}")
                    source_name = os.path.basename(source_dir)
                    dataset_stats[source_name] = {"files": 0, "classes": {}}
                    
                    # 获取源目录中的所有类别目录
                    class_dirs = []
                    if os.path.isdir(source_dir):
                        class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
                    
                    for class_dir in class_dirs:
                        class_path = os.path.join(source_dir, class_dir)
                        if os.path.isdir(class_path):
                            # 创建目标类别目录
                            target_class_dir = os.path.join(merged_path, class_dir)
                            os.makedirs(target_class_dir, exist_ok=True)
                            
                            # 复制图像文件
                            img_count = 0
                            img_paths = glob.glob(os.path.join(class_path, "*.jpg")) + glob.glob(os.path.join(class_path, "*.png"))
                            
                            for img in img_paths:
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
                                target_path = os.path.join(target_class_dir, img_name)
                                
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
                    logger.info(f"Processing {mode} dataset: {source_dir}")
                    source_name = os.path.basename(os.path.dirname(source_dir))
                    dataset_stats[source_name] = {"files": 0}
                    
                    img_count = 0
                    img_paths = glob.glob(os.path.join(source_dir, "*.jpg")) + glob.glob(os.path.join(source_dir, "*.png"))
                    
                    for img in img_paths:
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
                        target_path = os.path.join(merged_path, img_name)
                        
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
                logger.info(f"Selected largest {mode} dataset as fallback: {largest_dataset}")
                return largest_dataset
            else:
                # 如果没有数据集，返回默认路径
                if mode == "train":
                    return self.config.train_data
                elif mode == "test":
                    return self.config.test_data
                else:
                    return self.config.val_data
    
    def list_available_datasets(self) -> Dict[str, List[str]]:
        """列出所有可用的数据集
        
        返回:
            包含不同模式可用数据集的字典
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

    def cleanup_temp_folders(self) -> None:
        """清理数据集解压后的临时文件夹，保留处理后的数据
        
        该方法会删除数据处理完成后不再需要的原始解压目录和临时文件夹，但保留已处理的数据
        在删除前会进行安全检查，确保数据已经被正确处理
        """
        # 安全检查：确保训练数据目录已经准备好
        if not os.path.exists(self.paths.train_dir) or len(glob.glob(os.path.join(self.paths.train_dir, '*/'))) < 10:
            logger.warning("Training data directory does not exist or contains too few class folders. Skipping cleanup for safety.")
            return
        
        # 安全检查：确保至少有一些图像已经被处理
        train_image_count = len(glob.glob(os.path.join(self.paths.train_dir, '**/*.jpg'), recursive=True))
        if train_image_count < 100:  # 假设至少应该有100张图像
            logger.warning(f"Only {train_image_count} images found in training directory. Skipping cleanup for safety.")
            return
            
        data_dir = self.paths.data_dir
        temp_dirs_to_remove = [
            os.path.join(data_dir, "AgriculturalDisease_trainingset"),
            os.path.join(data_dir, "AgriculturalDisease_validationset"),
            # 添加临时文件夹
            self.paths.temp_dir,
            self.paths.temp_images_dir,
            self.paths.temp_labels_dir
        ]
        
        # 添加所有可能的测试集目录
        test_dirs = glob.glob(os.path.join(data_dir, "AgriculturalDisease_test*"))
        temp_dirs_to_remove.extend(test_dirs)
        
        # 确保不会删除关键目录
        safe_dirs = [
            self.paths.train_dir,
            self.paths.test_dir,
            self.paths.val_dir,
            self.paths.merged_train_dir,
            self.paths.merged_test_dir,
            self.paths.merged_val_dir,
            self.paths.aug_dir
        ]
        
        # 从删除列表中移除安全目录
        temp_dirs_to_remove = [dir for dir in temp_dirs_to_remove if dir not in safe_dirs]
        
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
        """
        Find a dataset file in the custom path or the data directory.
        
        Args:
            filename (str): The name of the dataset file to find
            custom_path (str, optional): A custom path to look for the dataset file
            
        Returns:
            str: The full path to the dataset file if found, otherwise None
        """
        logger = logging.getLogger('data_prep')
        
        # Check if a filename was provided
        if not filename:
            logger.warning("No filename provided to find_dataset_file")
            return None
        
        found_in_custom_path = False
        custom_file_path = None
        
        # First try the custom path if provided
        if custom_path:
            # Handle directory path or direct file path
            if os.path.isdir(custom_path):
                # Check if the file exists in the directory
                potential_path = os.path.join(custom_path, filename)
                if os.path.exists(potential_path):
                    logger.info(f"Found dataset file {filename} in custom directory: {custom_path}")
                    found_in_custom_path = True
                    custom_file_path = potential_path
                    return custom_file_path
                    
                # Also check if any file in the directory matches the supported formats
                found_alternative = False
                for file in os.listdir(custom_path):
                    if any(file.endswith(ext) for ext in config.supported_dataset_formats):
                        full_path = os.path.join(custom_path, file)
                        logger.info(f"Found alternative dataset file in custom directory: {full_path}")
                        found_in_custom_path = True
                        custom_file_path = full_path
                        found_alternative = True
                        return custom_file_path
                
                if not found_alternative:
                    logger.warning(f"Could not find {filename} or any supported dataset file in custom path: {custom_path}")
        
            # If custom_path is a direct file path
            elif os.path.isfile(custom_path):
                # Check if the file has a supported extension
                if any(custom_path.endswith(ext) for ext in config.supported_dataset_formats):
                    logger.info(f"Using custom dataset file: {custom_path}")
                    found_in_custom_path = True
                    custom_file_path = custom_path
                    return custom_file_path
                else:
                    logger.warning(f"Custom path file exists but is not a supported format: {custom_path}")
                    logger.info(f"Supported formats: {', '.join(config.supported_dataset_formats)}")
            
            # If custom_path doesn't exist
            else:
                logger.warning(f"Custom dataset path does not exist: {custom_path}")
        
        # If no custom path or nothing found in custom path, check default data directory
        default_path = os.path.join(config.paths.data_dir, filename)
        if os.path.exists(default_path):
            if custom_path and not found_in_custom_path:
                logger.warning(f"No data found in custom path. Falling back to default path: {default_path}")
            else:
                logger.info(f"Found dataset file in default data directory: {default_path}")
            return default_path
        
        # If still not found, check if any file in the data directory matches the supported formats
        if os.path.isdir(config.paths.data_dir):
            for file in os.listdir(config.paths.data_dir):
                if any(file.endswith(ext) for ext in config.supported_dataset_formats):
                    full_path = os.path.join(config.paths.data_dir, file)
                    if custom_path and not found_in_custom_path:
                        logger.warning(f"No data found in custom path. Falling back to alternative file in default path: {full_path}")
                    else:
                        logger.info(f"Found alternative dataset file in data directory: {full_path}")
                    return full_path
        
        # No dataset found anywhere
        if custom_path:
            logger.error(f"Could not find dataset file {filename} in custom path or default location")
            logger.warning("Please ensure your dataset files are in the specified path or in the default data directory")
        else:
            logger.error(f"Could not find dataset file {filename} in any location")
            logger.warning("Please download the dataset and place it in the data directory")
        
        return None

# 创建一个模块级函数来初始化和运行数据准备
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
    """
    Initialize and run data preparation tasks.
    
    Args:
        extract (bool): Whether to extract the dataset
        process (bool): Whether to process the images
        augment (bool): Whether to augment the images
        status (bool): Whether to check data status
        merge (str): Dataset merge mode ('train', 'test', 'val', 'all', or None)
        cleanup_temp (bool): Whether to clean up temporary folders
        custom_dataset_path (str, optional): Path to custom dataset files
        merge_augmented (bool, optional): Whether to merge augmented data with original data
          
    Returns:
        dict: Dictionary containing results of data preparation
    """
    logger = logging.getLogger('DataPrep')
    logger.info("Setting up data preparation...")
    
    result = {
        "success": True,
        "warnings": [],
        "errors": [],
        "data_status": {}
    }
    
    # 为了跟踪进度
    steps_completed = []
    
    # Override merge_augmented in config if specified
    original_merge_augmented = config.merge_augmented_data
    if merge_augmented is not None:
        config.merge_augmented_data = merge_augmented
        logger.info(f"Setting merge_augmented to {merge_augmented}")
    
    try:
        # Create data preparation object
        data_prep = DataPreparation()
        
        # Extract dataset
        if extract:
            logger.info("Extracting dataset...")
            training_file = config.training_dataset_file
            validation_file = config.validation_dataset_file
            
            try:
                # Look for the training dataset
                training_path = data_prep.find_dataset_file(training_file, custom_dataset_path)
                if training_path:
                    logger.info(f"Found training dataset at: {training_path}")
                    success = extract_dataset(training_path, paths.temp_dataset_dir)
                    if not success:
                        logger.error(f"Failed to extract training dataset from {training_path}")
                        result["errors"].append(f"Failed to extract training dataset from {training_path}")
                        result["success"] = False
                else:
                    logger.warning("Training dataset file not found. Skipping extraction.")
                    result["warnings"].append("Training dataset file not found. Skipping extraction.")
                
                # Look for the validation dataset
                validation_path = data_prep.find_dataset_file(validation_file, custom_dataset_path)
                if validation_path:
                    logger.info(f"Found validation dataset at: {validation_path}")
                    success = extract_dataset(validation_path, paths.temp_dataset_dir)
                    if not success:
                        logger.error(f"Failed to extract validation dataset from {validation_path}")
                        result["errors"].append(f"Failed to extract validation dataset from {validation_path}")
                        result["success"] = False
                else:
                    logger.warning("Validation dataset file not found. Skipping extraction.")
                    result["warnings"].append("Validation dataset file not found. Skipping extraction.")
                
                if not training_path and not validation_path:
                    # Check if we already have extracted data
                    if os.path.exists(paths.temp_dataset_dir) and os.listdir(paths.temp_dataset_dir):
                        logger.info(f"No dataset files found but temp directory exists with data, continuing with existing data")
                    else:
                        logger.error("No dataset files found and no existing extracted data. Cannot proceed with data preparation.")
                        result["errors"].append("No dataset files found and no existing extracted data.")
                        result["success"] = False
                
            except Exception as e:
                logger.error(f"Error during dataset extraction: {str(e)}")
                result["errors"].append(f"Error during dataset extraction: {str(e)}")
                result["success"] = False
            
            steps_completed.append("extract")
        
        # Process data
        if process and result["success"]:
            logger.info("Processing data...")
            try:
                data_prep.process_data()
                steps_completed.append("process")
            except Exception as e:
                logger.error(f"Error during data processing: {str(e)}")
                result["errors"].append(f"Error during data processing: {str(e)}")
                result["success"] = False
        
        # Augment data
        if augment and result["success"]:
            logger.info("Augmenting data...")
            try:
                # Check if we have enough training data first
                data_status = data_prep.get_data_status()
                
                if data_status.get("training", {}).get("total_images", 0) > 0:
                    train_path = os.path.join(paths.train_images_dir, "train")
                    augmented_dir = paths.augmented_images_dir
                    
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
            except Exception as e:
                logger.error(f"Error during data augmentation: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                result["warnings"].append(f"Error during data augmentation: {str(e)}")
        
        # 统一处理合并逻辑
        merge_operations = []
        
        # 添加对应的合并操作
        if merge:
            if merge == "all":
                merge_operations.extend(["train", "test", "val"])
            elif merge in ["train", "test", "val"]:
                merge_operations.append(merge)
        
        # 如果需要合并增强数据，添加到操作列表中
        if config.merge_augmented_data and "augment" in steps_completed:
            if "train" not in merge_operations:
                merge_operations.append("train")
        
        # 执行合并操作
        if merge_operations and result["success"]:
            logger.info(f"Merging datasets: {', '.join(merge_operations)}")
            try:
                for mode in merge_operations:
                    # 对于训练集，先处理是否需要合并增强数据
                    if mode == "train" and config.merge_augmented_data and "augment" in steps_completed:
                        logger.info("Merging augmented data with training data...")
                        # 现在不直接复制文件，而是确保在 merge_datasets 函数中处理增强数据
                        # 通过配置参数 merge_augmented_data 来控制是否包括增强数据
                    
                    # 执行合并操作
                    merged_path = data_prep.merge_datasets(mode, force=config.merge_force)
                    logger.info(f"Merged {mode} datasets to: {merged_path}")
                
                steps_completed.append("merge")
            except Exception as e:
                logger.error(f"Error during dataset merging: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                result["warnings"].append(f"Error during dataset merging: {str(e)}")
        
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
                
                # Check for corrupted images
                logger.info("Checking for corrupted images...")
                for dataset_type in ["training", "validation", "testing"]:
                    if dataset_type in data_status and "directory" in data_status[dataset_type]:
                        directory = data_status[dataset_type]["directory"]
                        if os.path.exists(directory):
                            data_prep.remove_error_images(directory)
                
                steps_completed.append("status")
            except Exception as e:
                logger.error(f"Error checking data status: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                result["errors"].append(f"Error checking data status: {str(e)}")
                result["success"] = False
        
        # Clean up temporary folders if requested
        if cleanup_temp and result["success"]:
            logger.info("Cleaning up temporary folders...")
            try:
                data_prep.cleanup_temp_folders()
                steps_completed.append("cleanup")
            except Exception as e:
                logger.error(f"Error cleaning up temporary folders: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                result["warnings"].append(f"Error cleaning up temporary folders: {str(e)}")
        
    except Exception as e:
        logger.error(f"Unexpected error in data preparation: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        result["errors"].append(f"Unexpected error in data preparation: {str(e)}")
        result["success"] = False
    
    # 恢复原始配置
    if merge_augmented is not None:
        config.merge_augmented_data = original_merge_augmented
    
    # 总结
    logger.info(f"Data preparation completed. Steps: {', '.join(steps_completed)}")
    if result["warnings"]:
        logger.warning(f"Warnings: {len(result['warnings'])}")
    if result["errors"]:
        logger.error(f"Errors: {len(result['errors'])}")
    
    return result

def extract_dataset(dataset_path, extract_to):
    """
    Extract dataset from zip, rar, tar, gz, or tgz file
    
    Args:
        dataset_path (str): Path to the dataset file
        extract_to (str): Directory to extract to
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger("DataPrep")
    
    try:
        if not os.path.exists(extract_to):
            os.makedirs(extract_to, exist_ok=True)
        
        logger.info(f"Extracting {dataset_path} to {extract_to}")
        
        # Extract based on file extension
        if dataset_path.endswith('.zip'):
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                
        elif dataset_path.endswith('.rar'):
            try:
                import rarfile
                with rarfile.RarFile(dataset_path) as rf:
                    rf.extractall(extract_to)
            except ImportError:
                logger.error("rarfile module not installed. Please install it to extract RAR files.")
                return False
                
        elif dataset_path.endswith('.tar'):
            with tarfile.open(dataset_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
                
        elif dataset_path.endswith('.gz') or dataset_path.endswith('.tgz'):
            with tarfile.open(dataset_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
                
        else:
            supported_formats = ', '.join(config.supported_dataset_formats)
            logger.error(f"Unsupported file format. Supported formats: {supported_formats}")
            return False
            
        logger.info(f"Successfully extracted {dataset_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting dataset {dataset_path}: {str(e)}")
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