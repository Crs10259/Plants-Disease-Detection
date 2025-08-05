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
from config import config, paths
import random
from concurrent.futures import ThreadPoolExecutor
from utils.utils import handle_datasets
import traceback
import re
import threading
import torch

# 设置日志
os.makedirs(os.path.dirname(paths.data_proc_log), exist_ok=True)

# 删除root logger的已有处理器以避免重复
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.FileHandler) and handler.baseFilename.endswith('data_processing.log'):
        root_logger.removeHandler(handler)

# 设置基本日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(paths.data_proc_log, encoding="utf-8", mode="a"),
        logging.StreamHandler()
    ]
)

# 获取数据准备日志记录器
logger = logging.getLogger('DataPreparation')

# 确保日志记录器级别正确设置
logger.setLevel(logging.INFO)

# 确保处理器的编码和级别正确
for handler in logger.handlers:
    if isinstance(handler, logging.FileHandler):
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        handler.setLevel(logging.INFO)

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

def setup_data(extract=False, process=False, augment=False, status=False, 
               merge=None, cleanup_temp=False, custom_dataset_path=None, 
               merge_augmented=None, config_obj=None, force_cleanup=False):
    """设置和准备数据集
    
    参数:
        extract: 是否解压数据集文件
        process: 是否处理数据
        augment: 是否执行数据增强
        status: 是否检查状态
        merge: 要合并的数据集类型('train', 'test', 'val', 'all')
        cleanup_temp: 处理后是否清理临时文件
        custom_dataset_path: 自定义数据集路径
        merge_augmented: 是否合并增强数据
        config_obj: 配置对象(如果为None则使用默认配置)
        force_cleanup: 是否强制清理而不询问用户
        
    返回:
        数据准备结果字典
    """
    try:
        # 创建数据准备类的实例
        data_prep = DataPreparation(config_obj)
        
        # 如果提供了自定义数据集路径，更新配置
        if custom_dataset_path:
            config.dataset_path = custom_dataset_path
            config.use_custom_dataset_path = True
            logger.info(f"Using custom dataset path: {custom_dataset_path}")
        
        # 如果指定了合并增强数据选项，更新配置
        if merge_augmented is not None:
            config.merge_augmented_data = merge_augmented
            logger.info(f"Merge augmented data set to: {merge_augmented}")
        
        # 创建所需目录
        data_prep.setup_directories()
        
        # 检查数据状态
        if status or (not extract and not process and not augment and not merge):
            # 获取数据状态但只显示必要的信息
            data_status = data_prep.get_data_status()
            data_prep.check_data_status()
            
            # 检查只需要的数据是否准备好
            train_ready = data_status["processed_details"].get("training", False)
            test_ready = data_status["processed_details"].get("testing", False)
            aug_ready = not config.use_data_aug or data_status["augmentation_completed"]
            merge_ready = not (config.merge_datasets or config.merge_train_datasets or 
                               config.merge_test_datasets or config.merge_val_datasets) or data_status["merged_datasets"]
            
            if train_ready and test_ready and aug_ready and merge_ready:
                logger.info("All required data is ready based on current configuration.")
                # 检查是否有可清理的数据
                data_prep.check_for_cleanable_data(force=force_cleanup)
                return {"status": "success", "message": "Data already prepared"}
        
        # 根据指定的操作执行相应的数据准备步骤
        
        # 1. 提取数据 - 这一步总是需要的
        if extract:
            logger.info("Extracting dataset archives...")
            # 调用数据集提取功能
            data_prep.extract_datasets()
        
        # 2. 处理数据 - 这一步总是需要的
        if process:
            logger.info("Processing dataset...")
            data_prep.process_data()
        
        # 3. 执行数据增强 - 只有在config.use_data_aug=True时才需要
        if augment and config.use_data_aug:
            logger.info("Augmenting data...")
            data_prep.augment_directory()
        elif augment and not config.use_data_aug:
            logger.info("Data augmentation is disabled (use_data_aug=False). Skipping augmentation step.")
        
        # 4. 合并数据集 - 只在相应的合并标志设置为True时才需要
        need_merge = False
        if merge:
            if merge == 'train' and (config.merge_datasets or config.merge_train_datasets):
                need_merge = True
            elif merge == 'test' and (config.merge_datasets or config.merge_test_datasets):
                need_merge = True
            elif merge == 'val' and (config.merge_datasets or config.merge_val_datasets):
                need_merge = True
            elif merge == 'all' and (config.merge_datasets or config.merge_train_datasets or 
                                     config.merge_test_datasets or config.merge_val_datasets):
                need_merge = True
                
            if need_merge:
                logger.info(f"Merging datasets: {merge}")
                # 调用数据集合并功能
                data_prep.merge_datasets(merge_type=merge)
            else:
                logger.info(f"Dataset merging is not enabled for {merge}. Skipping merge step.")
        
        # 检查处理后的数据状态
        if extract or process or augment or (merge and need_merge):
            logger.info("Checking data status after processing...")
            data_status = data_prep.get_data_status()
            data_prep.check_data_status()
            
            # 检查只需要的数据是否准备好
            train_ready = data_status["processed_details"].get("training", False)
            test_ready = data_status["processed_details"].get("testing", False)
            
            if train_ready and test_ready:
                logger.info("Dataset processing is complete.")
                # 检查是否有可清理的数据
                data_prep.check_for_cleanable_data(force=force_cleanup)
        
        # 清理临时文件
        if cleanup_temp:
            logger.info("Cleaning up temporary files...")
            data_prep.cleanup_temp_files(force=force_cleanup)
        
        # 返回数据准备结果
        result = {
            "status": "success",
            "operations": {
                "extract": extract,
                "process": process,
                "augment": augment and config.use_data_aug,
                "status": status,
                "merge": merge and need_merge,
                "cleanup_temp": cleanup_temp
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error setting up data: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

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
            os.makedirs(self.paths.log_dir, exist_ok=True)
            
            # 模型保存目录 - 更清晰的命名
            os.makedirs(self.paths.weight_dir, exist_ok=True)
            os.makedirs(self.paths.best_weight_dir, exist_ok=True)
            
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
            
            # 尝试多个可能的图像路径
            possible_paths = [
                # 标准临时图像目录
                normalize_path(os.path.join(self.paths.temp_images_dir, filename)),
                # 训练集提取目录
                normalize_path(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_trainingset", "images", filename)),
                # 嵌套训练集目录
                normalize_path(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_trainingset", "AgriculturalDisease_trainingset", "images", filename)),
                # 验证集提取目录
                normalize_path(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_validationset", "images", filename)),
                # 嵌套验证集目录
                normalize_path(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_validationset", "AgriculturalDisease_validationset", "images", filename)),
                # 数据目录下的图像
                normalize_path(os.path.join(self.paths.data_dir, "images", filename)),
                # 数据目录下的图像（旧结构）
                normalize_path(os.path.join(self.paths.data_dir, "temp", "images", filename))
            ]
            
            # 找到第一个存在的图像路径
            origin_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    origin_path = path
                    break
            
            # 如果找不到图像，记录错误并返回
            if origin_path is None:
                logger.error(f"Image file not found: {filename}. Tried paths: {', '.join(possible_paths)}")
                return False
                
            # 跳过类别44和45
            ids = file["disease_class"]
            if ids == 44 or ids == 45:
                return False
                
            # 调整ID顺序（大于45的ID要减2）
            if ids > 45:
                ids -= 2
                
            # 确定是训练集还是验证集图像
            is_validation = "validationset" in origin_path.lower()
            
            # 根据图像来源选择目标目录
            if is_validation:
                # 验证集图像保存到val目录
                save_dir = normalize_path(os.path.join(self.paths.val_dir, str(ids)))
            else:
                # 训练集图像保存到train目录
                save_dir = normalize_path(os.path.join(self.paths.train_dir, str(ids)))
            
            os.makedirs(save_dir, exist_ok=True)
            
            # 构建目标路径并复制文件
            save_path = normalize_path(os.path.join(save_dir, filename))
            
            # 检查源文件是否存在
            if not os.path.exists(origin_path):
                logger.error(f"Source file does not exist: {origin_path}")
                return False
            
            # 检查目标文件是否已存在
            if os.path.exists(save_path):
                # 文件已存在，视为成功
                return True
            
            shutil.copy(origin_path, save_path)
            return True
        except Exception as e:
            logger.error(f"Error processing file {file.get('image_id', 'unknown')}: {str(e)}")
            return False

    def find_image_path(self, filename):
        """查找图像文件路径并缓存结果"""
        # 尝试多个可能的图像路径
        possible_paths = [
            # 标准临时图像目录
            normalize_path(os.path.join(self.paths.temp_images_dir, filename)),
            # 训练集提取目录
            normalize_path(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_trainingset", "images", filename)),
            # 嵌套训练集目录
            normalize_path(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_trainingset", "AgriculturalDisease_trainingset", "images", filename)),
            # 验证集提取目录
            normalize_path(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_validationset", "images", filename)),
            # 嵌套验证集目录
            normalize_path(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_validationset", "AgriculturalDisease_validationset", "images", filename)),
            # 数据目录下的图像
            normalize_path(os.path.join(self.paths.data_dir, "images", filename)),
            # 数据目录下的图像（旧结构）
            normalize_path(os.path.join(self.paths.data_dir, "temp", "images", filename))
        ]
        
        # 找到第一个存在的图像路径
        origin_path = None
        for path in possible_paths:
            if os.path.exists(path):
                origin_path = path
                break
        
        # 如果找不到图像，记录错误并返回
        if origin_path is None:
            logger.error(f"Image file not found: {filename}. Tried paths: {', '.join(possible_paths)}")
            return None
        
        return origin_path

    def batch_copy_files(self, files: List[Dict[str, Any]], max_workers: int = None):
        """批量复制多个文件到对应目录，使用多线程提高性能
        
        参数:
            files: 包含图像文件信息的字典列表
            max_workers: 最大线程数，如果为None则自动设置
            
        返回:
            成功复制的文件数量
        """
        if not files:
            return 0
            
        # 按类别和数据集类型将文件分组
        train_files_by_class = {}
        val_files_by_class = {}
        
        for file in files:
            class_id = file["disease_class"]
            
            # 跳过类别44和45
            if class_id == 44 or class_id == 45:
                continue
                
            # 调整类别ID（大于45的减2）
            adjusted_id = class_id
            if adjusted_id > 45:
                adjusted_id -= 2
            
            # 确定是否为验证集文件
            filename = file["image_id"]
            origin_path = self.find_image_path(filename)
            if origin_path and "validationset" in origin_path.lower():
                if adjusted_id not in val_files_by_class:
                    val_files_by_class[adjusted_id] = []
                val_files_by_class[adjusted_id].append(file)
            else:
                if adjusted_id not in train_files_by_class:
                    train_files_by_class[adjusted_id] = []
                train_files_by_class[adjusted_id].append(file)
            
        # 预先创建所有目标目录
        for class_id in train_files_by_class.keys():
            save_dir = normalize_path(os.path.join(self.paths.train_dir, str(class_id)))
            os.makedirs(save_dir, exist_ok=True)
            
        for class_id in val_files_by_class.keys():
            save_dir = normalize_path(os.path.join(self.paths.val_dir, str(class_id)))
            os.makedirs(save_dir, exist_ok=True)
            
        # 缓存图像文件路径以提高性能
        image_paths_cache = {}
        image_paths_lock = threading.Lock()
        
        # 使用线程池并行处理多个类别的图像提取
        success_count = 0
        error_count = 0
        not_found_count = 0
        
        # 在ThreadPoolExecutor内部处理文件
        with tqdm(total=len(files), desc="Processing images") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                # 提交所有文件处理任务
                for file in files:
                    # 跳过类别44和45
                    class_id = file["disease_class"]
                    if class_id == 44 or class_id == 45:
                        pbar.update(1)
                        continue
                        
                    # 调整类别ID
                    adjusted_id = class_id
                    if adjusted_id > 45:
                        adjusted_id -= 2
                    
                    filename = file["image_id"]
                    
                    # 使用缓存查找图像路径
                    with image_paths_lock:
                        if filename in image_paths_cache:
                            origin_path = image_paths_cache[filename]
                        else:
                            # 使用类方法查找图像路径
                            origin_path = self.find_image_path(filename)
                            if origin_path:
                                image_paths_cache[filename] = origin_path
                            else:
                                not_found_count += 1
                                pbar.update(1)
                                continue
                    
                    # 确定目标目录（训练集或验证集）
                    is_validation = "validationset" in origin_path.lower()
                    if is_validation:
                        save_dir = normalize_path(os.path.join(self.paths.val_dir, str(adjusted_id)))
                    else:
                        save_dir = normalize_path(os.path.join(self.paths.train_dir, str(adjusted_id)))
                    
                    save_path = normalize_path(os.path.join(save_dir, filename))
                    
                    # 如果文件已存在，则跳过
                    if os.path.exists(save_path):
                        success_count += 1
                        pbar.update(1)
                        continue
                        
                    # 提交复制任务
                    future = executor.submit(shutil.copy, origin_path, save_path)
                    futures.append((future, pbar))
                
                # 等待所有任务完成
                for future, pbar in futures:
                    try:
                        future.result()
                        success_count += 1
                    except Exception as e:
                        logger.error(f"Error copying file: {str(e)}")
                        error_count += 1
                    finally:
                        pbar.update(1)
        
        if error_count > 0 or not_found_count > 0:
            logger.warning(f"Completed with {error_count} errors and {not_found_count} files not found")
            
        return success_count

    def get_data_status(self) -> dict:
        """获取数据准备的状态信息，基于当前配置只检查必要的目录
        
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
        
        # 检查数据处理状态 - 根据配置只检查必要的目录
        try:
            # 检查训练目录是否有图像
            train_processed = os.path.exists(self.paths.train_dir) and len(glob.glob(os.path.join(self.paths.train_dir, "**/*.jpg"), recursive=True)) > 0
            
            # 检查测试目录是否有图像
            test_processed = os.path.exists(self.paths.test_images_dir) and len(glob.glob(os.path.join(self.paths.test_images_dir, "*.jpg"))) > 0
            
            status["data_processed"] = train_processed or test_processed
            status["processed_details"] = {
                "training": train_processed,
                "testing": test_processed
            }
        except Exception as e:
            logger.error(f"Error checking data processing status: {str(e)}")
            status["data_processed"] = False
        
        # 检查数据增强状态 - 只有当启用数据增强时才检查
        if self.config.use_data_aug:
            try:
                # 检查增强数据目录是否有图像
                aug_dir_exists = os.path.exists(self.paths.aug_train_dir)
                if aug_dir_exists:
                    aug_images = glob.glob(os.path.join(self.paths.aug_train_dir, "**/*.jpg"), recursive=True)
                    status["augmentation_completed"] = len(aug_images) > 0
                    status["augmented_image_count"] = len(aug_images)
                else:
                    status["augmentation_completed"] = False
                    status["augmented_image_count"] = 0
            except Exception as e:
                logger.error(f"Error checking data augmentation status: {str(e)}")
                status["augmentation_completed"] = False
        else:
            # 如果未启用数据增强，则标记为已完成
            status["augmentation_completed"] = True
            status["augmented_image_count"] = 0
        
        # 检查数据集合并状态 - 只有当启用数据集合并时才检查
        if self.config.merge_datasets or self.config.merge_train_datasets or self.config.merge_test_datasets or self.config.merge_val_datasets:
            try:
                # 根据具体合并设置检查相应目录
                merged_train = (self.config.merge_datasets or self.config.merge_train_datasets) and \
                                os.path.exists(self.paths.merged_train_dir) and \
                                len(glob.glob(os.path.join(self.paths.merged_train_dir, "**/*.jpg"), recursive=True)) > 0
                
                merged_test = (self.config.merge_datasets or self.config.merge_test_datasets) and \
                                os.path.exists(self.paths.merged_test_dir) and \
                                len(glob.glob(os.path.join(self.paths.merged_test_dir, "*.jpg"))) > 0
                
                merged_val = (self.config.merge_datasets or self.config.merge_val_datasets) and \
                                os.path.exists(self.paths.merged_val_dir) and \
                                len(glob.glob(os.path.join(self.paths.merged_val_dir, "*.jpg"))) > 0
                
                # 只检查需要合并的数据集
                status["merged_datasets"] = (
                    (not (self.config.merge_datasets or self.config.merge_train_datasets) or merged_train) and
                    (not (self.config.merge_datasets or self.config.merge_test_datasets) or merged_test) and
                    (not (self.config.merge_datasets or self.config.merge_val_datasets) or merged_val)
                )
                
                status["merged_details"] = {
                    "training": merged_train if (self.config.merge_datasets or self.config.merge_train_datasets) else None,
                    "testing": merged_test if (self.config.merge_datasets or self.config.merge_test_datasets) else None,
                    "validation": merged_val if (self.config.merge_datasets or self.config.merge_val_datasets) else None
                }
            except Exception as e:
                logger.error(f"Error checking dataset merging status: {str(e)}")
                status["merged_datasets"] = False
        else:
            # 如果未启用数据集合并，则标记为已完成
            status["merged_datasets"] = True
            status["merged_details"] = {"training": None, "testing": None, "validation": None}
        
        # 只检查所需的目录
        data_dirs = {}
        
        # 基本目录 - 总是需要检查
        data_dirs["train"] = self.paths.train_dir
        data_dirs["test"] = self.paths.test_dir
        
        # 视配置添加其他目录
        if self.config.use_data_aug:
            data_dirs["augmented"] = self.paths.aug_dir if hasattr(self.paths, "aug_dir") else None
        
        if self.config.merge_datasets or self.config.merge_train_datasets:
            data_dirs["merged_train"] = self.paths.merged_train_dir
        
        if self.config.merge_datasets or self.config.merge_test_datasets:
            data_dirs["merged_test"] = self.paths.merged_test_dir
        
        if self.config.merge_datasets or self.config.merge_val_datasets:
            data_dirs["merged_val"] = self.paths.merged_val_dir
        
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
        
        # 计算训练图像数量 - 根据配置选择合适的目录
        training_imgs = 0
        if "train" in status["datasets"] and "file_count" in status["datasets"]["train"]:
            training_imgs += status["datasets"]["train"]["file_count"]
        
        if self.config.merge_datasets or self.config.merge_train_datasets:
            if "merged_train" in status["datasets"] and "file_count" in status["datasets"]["merged_train"]:
                training_imgs += status["datasets"]["merged_train"]["file_count"]
        
        status["training"] = {
            "directory": self.paths.train_dir,
            "total_images": training_imgs
        }
        
        # 计算测试图像数量 - 根据配置选择合适的目录
        testing_imgs = 0
        if "test" in status["datasets"] and "file_count" in status["datasets"]["test"]:
            testing_imgs += status["datasets"]["test"]["file_count"]
        
        if self.config.merge_datasets or self.config.merge_test_datasets:
            if "merged_test" in status["datasets"] and "file_count" in status["datasets"]["merged_test"]:
                testing_imgs += status["datasets"]["merged_test"]["file_count"]
        
        status["testing"] = {
            "directory": self.paths.test_dir,
            "total_images": testing_imgs
        }
        
        # 检查配置信息
        status["config"] = {
            "use_data_aug": self.config.use_data_aug,
            "merge_datasets": self.config.merge_datasets,
            "merge_train_datasets": self.config.merge_train_datasets,
            "merge_test_datasets": self.config.merge_test_datasets,
            "merge_val_datasets": self.config.merge_val_datasets,
            "custom_dataset_path": self.config.dataset_path if hasattr(self.config, 'use_custom_dataset_path') and self.config.use_custom_dataset_path else None
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
                logger.info(f"  [OK] {zip_file} extracted")
            else:
                logger.info(f"  [MISSING] {zip_file} not extracted")
        
        # 检查数据处理状态
        logger.info("\nData Processing:")
        if status["data_processed"]:
            logger.info("[OK] Data processing completed")
        else:
            logger.info("[MISSING] Data processing not completed")
            
        for data_type, processed in status["processed_details"].items():
            if processed is not None:  # 只显示需要检查的目录
                if processed:
                    logger.info(f"  [OK] {data_type} data processed")
                else:
                    logger.info(f"  [MISSING] {data_type} data not processed")
        
        # 检查数据增强状态 - 只有当启用数据增强时才显示
        if self.config.use_data_aug:
            logger.info("\nData Augmentation:")
            if status["augmentation_completed"]:
                logger.info("[OK] Data augmentation completed")
                logger.info(f"  Total augmented images: {status['augmented_image_count']}")
            else:
                logger.info("[MISSING] Data augmentation not completed")
        
        # 检查数据集合并状态 - 只有当启用数据集合并时才显示
        if self.config.merge_datasets or self.config.merge_train_datasets or self.config.merge_test_datasets or self.config.merge_val_datasets:
            logger.info("\nDataset Merging:")
            if status["merged_datasets"]:
                logger.info("[OK] Datasets merged")
                for data_type, merged in status["merged_details"].items():
                    if merged is not None:  # 只显示需要合并的数据集
                        if merged:
                            logger.info(f"  [OK] {data_type} datasets merged")
                        else:
                            logger.info(f"  [MISSING] {data_type} datasets not merged")
            else:
                logger.info("[MISSING] Datasets not merged")
        
        # 显示目录统计信息 - 根据配置只显示相关目录
        logger.info("\nDirectory Statistics:")
        for name, details in status["datasets"].items():
            if details["exists"]:
                file_count = details.get("file_count", 0)
                if name == "train":
                    class_count = details.get("class_count", 0)
                    logger.info(f"  {name}: {file_count} files in {class_count} classes")
                else:
                    logger.info(f"  {name}: {file_count} files")
        
        logger.info("\nNext Recommended Steps:")
        if not status["dataset_extracted"]:
            logger.info("1. Extract dataset archives")
        elif not status["data_processed"]:
            logger.info("2. Process extracted data")
        elif not status["augmentation_completed"] and self.config.use_data_aug:
            logger.info("3. Run data augmentation")
        elif not status["merged_datasets"] and (self.config.merge_datasets or self.config.merge_train_datasets or self.config.merge_test_datasets or self.config.merge_val_datasets):
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
            # 检查图像尺寸，如果太大则调整大小
            h, w = img.shape[:2]
            max_dimension = 1500  # 设置最大尺寸，根据内存情况调整
            
            if h > max_dimension or w > max_dimension:
                # 计算缩放比例
                scale = max_dimension / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                logger.info(f"Resizing large image from {h}x{w} to {new_h}x{new_w} for noise augmentation")
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 转换为float32而不是float64，减少内存使用
            img_float = img.astype(np.float32) / 255.0
            
            # 使用memory-efficient的方法添加噪声
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
            # 检查图像尺寸，如果太大则调整大小
            h, w = img.shape[:2]
            max_dimension = 1500  # 设置最大尺寸，根据内存情况调整
            
            if h > max_dimension or w > max_dimension:
                # 计算缩放比例
                scale = max_dimension / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                logger.info(f"Resizing large image from {h}x{w} to {new_h}x{new_w} for brightness adjustment")
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
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
            # 检查图像尺寸，如果太大则调整大小
            h, w = img.shape[:2]
            max_dimension = 1500  # 设置最大尺寸，根据内存情况调整
            
            if h > max_dimension or w > max_dimension:
                # 计算缩放比例
                scale = max_dimension / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                logger.info(f"Resizing large image from {h}x{w} to {new_h}x{new_w} for advanced augmentation")
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
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
        max_workers = self.config.aug_max_workers  # 最多使用8个线程，避免过度并行
        
        # 使用更小的批次尺寸来降低内存使用
        batch_size = 50
        
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
            
            # 获取文件大小
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            if file_size_mb > 10:  # 如果文件大于10MB，记录信息
                logger.info(f"Processing large image ({file_size_mb:.2f} MB): {image_path}")

            try:
                # 使用PIL打开图像
                pil_img = Image.open(image_path)
                # 使用OpenCV打开图像
                cv_image = cv2.imread(image_path)
                
                if cv_image is None:
                    logger.warning(f"Failed to read image: {image_path}")
                    return []
                
                # 获取图像尺寸信息
                height, width = cv_image.shape[:2]
                if height > 3000 or width > 3000:
                    logger.info(f"Large image detected: {width}x{height} pixels")
                
            except Exception as e:
                logger.error(f"Error opening image {image_path}: {str(e)}")
                return []

            basename = os.path.basename(image_path)
            augmented_images = []

            # 添加高斯噪声
            if self.config.aug_noise:
                try:
                    gau_image = self.add_noise(cv_image.copy())
                    if gau_image is not None:
                        output_path = normalize_path(os.path.join(save_dir, f"gau_{basename}"))
                        try:
                            cv2.imwrite(output_path, gau_image)
                            augmented_images.append(output_path)
                        except Exception as e:
                            logger.error(f"Error saving noised image to {output_path}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error applying noise to {image_path}: {str(e)}")
                
                # 强制清理内存
                gau_image = None

            # 调整亮度
            if self.config.aug_brightness:
                try:
                    light_image = self.change_brightness(cv_image.copy())
                    if light_image is not None:
                        output_path = normalize_path(os.path.join(save_dir, f"light_{basename}"))
                        try:
                            cv2.imwrite(output_path, light_image)
                            augmented_images.append(output_path)
                        except Exception as e:
                            logger.error(f"Error saving brightness image to {output_path}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error adjusting brightness for {image_path}: {str(e)}")
                
                # 强制清理内存
                light_image = None

            # 翻转
            if self.config.aug_flip:
                try:
                    lr_path = normalize_path(os.path.join(save_dir, f"left_right_{basename}"))
                    tb_path = normalize_path(os.path.join(save_dir, f"top_bottom_{basename}"))
                    
                    # 检查图像是否过大，如果过大则调整大小后再翻转
                    w, h = pil_img.size
                    resized_img = pil_img
                    if w > 3000 or h > 3000:
                        logger.info(f"Resizing large image from {w}x{h} for flip operation")
                        max_dimension = 1500
                        scale = max_dimension / max(w, h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        resized_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                    
                    img_flip_left_right = resized_img.transpose(Image.FLIP_LEFT_RIGHT)
                    img_flip_top_bottom = resized_img.transpose(Image.FLIP_TOP_BOTTOM)
                    
                    img_flip_left_right.save(lr_path)
                    img_flip_top_bottom.save(tb_path)
                    
                    augmented_images.extend([lr_path, tb_path])
                    
                    # 强制清理内存
                    img_flip_left_right = None
                    img_flip_top_bottom = None
                    
                except Exception as e:
                    logger.error(f"Error saving flipped images for {image_path}: {str(e)}")
            
            # 强制清理内存
            pil_img = None
            cv_image = None
            
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
        
        # 限制同时处理的图像数量以减少内存使用
        batch_size = 500  # 每批处理的图像数
        max_workers = max(self.config.aug_max_workers, 4)  
        
        error_count = 0
        total_processed = 0
        
        # 分批处理图像
        for batch_start in range(0, len(image_files), batch_size):
            batch_end = min(batch_start + batch_size, len(image_files))
            batch = image_files[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(image_files)-1)//batch_size + 1} ({len(batch)} images)")
            
            batch_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for image_path in batch:
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
                                desc=f"Processing batch {batch_start//batch_size + 1}"):
                    try:
                        result = future.result()
                        if result:
                            batch_results.extend(result)
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error in augmentation thread: {str(e)}")
            
            # 处理批次结果
            augmented_files.extend(batch_results)
            total_processed += len(batch)
            
            # 显示进度
            logger.info(f"Processed {total_processed}/{len(image_files)} images, created {len(batch_results)} augmented images in this batch")
            
            # 强制进行垃圾回收
            import gc
            gc.collect()
            
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during augmentation")
            
        logger.info(f"Created {len(augmented_files)} augmented images in total")
        return augmented_files

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
            
        # 优先尝试直接提取图像
        logger.info("Attempting direct image extraction from archives...")
        if self.extract_images_directly():
            logger.info("Direct image extraction successful.")
            return
            
        logger.info("Direct extraction failed or incomplete. Falling back to standard processing...")
        
        # 加载标注文件
        try:
            train_json = self.paths.train_annotation
            val_json = self.paths.val_annotation
            
            # 检查标准路径是否存在训练标注文件
            if not os.path.exists(train_json):
                # 尝试在提取的数据集目录中查找训练标注文件 - 扩展搜索范围以包含嵌套目录
                possible_train_anno_paths = [
                    # 标准位置
                    normalize_path(os.path.join(self.paths.temp_dataset_dir, self.paths.train_dataset_folder_name, "AgriculturalDisease_train_annotations.json")),
                    # 嵌套目录位置
                    normalize_path(os.path.join(self.paths.temp_dataset_dir, self.paths.train_dataset_folder_name, self.paths.train_dataset_folder_name, "AgriculturalDisease_train_annotations.json")),
                    # 其他可能的位置
                    normalize_path(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_train_annotations.json")),
                    normalize_path(os.path.join(self.paths.data_dir, "AgriculturalDisease_train_annotations.json"))
                ]
                
                # 查找存在的注释文件
                found_anno_path = None
                for path in possible_train_anno_paths:
                    if os.path.exists(path):
                        found_anno_path = path
                        break
                
                if found_anno_path:
                    logger.info(f"Found training annotations at {found_anno_path}, copying to {train_json}")
                    # 确保目标目录存在
                    os.makedirs(os.path.dirname(train_json), exist_ok=True)
                    # 复制标注文件到标准位置
                    shutil.copy(found_anno_path, train_json)
                else:
                    # 递归搜索整个数据集目录
                    logger.info("Searching recursively for training annotations file...")
                    for root, _, files in os.walk(self.paths.temp_dataset_dir):
                        for file in files:
                            if file == "AgriculturalDisease_train_annotations.json":
                                found_path = os.path.join(root, file)
                                logger.info(f"Found training annotations at {found_path}")
                                # 确保目标目录存在
                                os.makedirs(os.path.dirname(train_json), exist_ok=True)
                                # 复制标注文件到标准位置
                                shutil.copy(found_path, train_json)
                                break
                    
                    # 如果仍未找到，返回错误
                    if not os.path.exists(train_json):
                        logger.error(f"Training annotations file not found: {train_json}")
                        logger.info("Please run extraction first to prepare the dataset")
                        return
                
            # 检查标准路径是否存在验证标注文件
            if not os.path.exists(val_json):
                # 尝试在提取的数据集目录中查找验证标注文件 - 扩展搜索范围以包含嵌套目录
                possible_val_anno_paths = [
                    # 标准位置
                    normalize_path(os.path.join(self.paths.temp_dataset_dir, self.paths.val_dataset_folder_name, "AgriculturalDisease_validation_annotations.json")),
                    # 嵌套目录位置
                    normalize_path(os.path.join(self.paths.temp_dataset_dir, self.paths.val_dataset_folder_name, self.paths.val_dataset_folder_name, "AgriculturalDisease_validation_annotations.json")),
                    # 其他可能的位置
                    normalize_path(os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_validation_annotations.json")),
                    normalize_path(os.path.join(self.paths.data_dir, "AgriculturalDisease_validation_annotations.json"))
                ]
                
                # 查找存在的注释文件
                found_anno_path = None
                for path in possible_val_anno_paths:
                    if os.path.exists(path):
                        found_anno_path = path
                        break
                
                if found_anno_path:
                    logger.info(f"Found validation annotations at {found_anno_path}, copying to {val_json}")
                    # 确保目标目录存在
                    os.makedirs(os.path.dirname(val_json), exist_ok=True)
                    # 复制标注文件到标准位置
                    shutil.copy(found_anno_path, val_json)
                else:
                    # 递归搜索整个数据集目录
                    logger.info("Searching recursively for validation annotations file...")
                    for root, _, files in os.walk(self.paths.temp_dataset_dir):
                        for file in files:
                            if file == "AgriculturalDisease_validation_annotations.json":
                                found_path = os.path.join(root, file)
                                logger.info(f"Found validation annotations at {found_path}")
                                # 确保目标目录存在
                                os.makedirs(os.path.dirname(val_json), exist_ok=True)
                                # 复制标注文件到标准位置
                                shutil.copy(found_path, val_json)
                                break
                    
                    # 如果仍未找到，返回错误
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
        
        # 使用优化后的批处理复制函数
        logger.info(f"Processing {len(file_list)} files using optimized batch processing...")
        success_count = self.batch_copy_files(file_list)
        
        logger.info(f"\nSuccessfully processed {success_count}/{len(file_list)} files")

    def is_valid_dataset_directory(self, directory_path: str, dataset_type: str) -> bool:
        """检查目录是否包含有效的数据集
        
        参数:
            directory_path: 要检查的目录路径
            dataset_type: 数据集类型，'train'或'test'或'val'
            
        返回:
            布尔值，指示目录是否包含有效数据集
        """
        try:
            directory_path = normalize_path(directory_path)
            
            if not os.path.exists(directory_path):
                logger.debug(f"Directory does not exist: {directory_path}")
                return False
                
            # 检查是否为训练目录
            if dataset_type == 'train':
                # 训练目录应该包含子目录（类别目录）
                subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
                if not subdirs:
                    logger.debug(f"Training directory has no class subdirectories: {directory_path}")
                    return False
                    
                # 检查至少一个子目录包含图像
                has_images = False
                for subdir in subdirs:
                    subdir_path = os.path.join(directory_path, subdir)
                    images = glob.glob(os.path.join(subdir_path, "*.jpg")) + \
                             glob.glob(os.path.join(subdir_path, "*.jpeg")) + \
                             glob.glob(os.path.join(subdir_path, "*.png"))
                    if images:
                        has_images = True
                        break
                        
                return has_images
                
            # 检查是否为测试目录
            elif dataset_type == 'test':
                # 测试目录应该直接包含图像
                images = glob.glob(os.path.join(directory_path, "*.jpg")) + \
                         glob.glob(os.path.join(directory_path, "*.jpeg")) + \
                         glob.glob(os.path.join(directory_path, "*.png"))
                return len(images) > 0
                
            # 检查是否为验证目录
            elif dataset_type == 'val':
                # 验证目录可能包含images子目录
                images_dir = os.path.join(directory_path, "images")
                if os.path.exists(images_dir) and os.path.isdir(images_dir):
                    images = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                             glob.glob(os.path.join(images_dir, "*.jpeg")) + \
                             glob.glob(os.path.join(images_dir, "*.png"))
                    return len(images) > 0
                    
                # 或者验证目录也可能直接包含图像
                images = glob.glob(os.path.join(directory_path, "*.jpg")) + \
                         glob.glob(os.path.join(directory_path, "*.jpeg")) + \
                         glob.glob(os.path.join(directory_path, "*.png"))
                return len(images) > 0
                
            else:
                logger.warning(f"Unknown dataset type: {dataset_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking directory {directory_path}: {str(e)}")
            return False
            
    def extract_images_directly(self) -> bool:
        """直接从ZIP文件中提取图像到目标目录结构
        
        返回:
            布尔值，指示直接提取是否成功
        """
        # 实现直接从ZIP文件提取图像的逻辑
        # 这个方法在process_data中被调用
        # 此处仅返回False，表示直接提取未实现或未成功
        logger.info("Direct image extraction not implemented")
        return False
        
    def extract_datasets(self) -> bool:
        """解压和提取数据集文件
        
        返回:
            布尔值，指示提取是否成功
        """
        try:
            # 确保目标目录存在
            os.makedirs(self.paths.temp_dataset_dir, exist_ok=True)
            
            # 查找训练和验证数据集文件
            train_dataset_found = False
            val_dataset_found = False
            
            # 在数据目录下查找训练和验证数据集文件
            data_dir = normalize_path(self.paths.data_dir)
            
            # 训练数据集模式
            train_patterns = ["*train*.zip", "*training*.zip", "*TRAIN*.zip"]
            val_patterns = ["*val*.zip", "*valid*.zip", "*validation*.zip", "*VAL*.zip"]
            
            # 查找训练数据集
            train_files = []
            for pattern in train_patterns:
                train_files.extend(glob.glob(os.path.join(data_dir, pattern)))
                train_files.extend(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))
            
            # 查找验证数据集
            val_files = []
            for pattern in val_patterns:
                val_files.extend(glob.glob(os.path.join(data_dir, pattern)))
                val_files.extend(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))
            
            # 提取训练数据集
            if train_files:
                logger.info(f"找到训练数据集文件: {train_files}")
                train_extract_dir = os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_trainingset")
                os.makedirs(train_extract_dir, exist_ok=True)
                
                for train_file in train_files:
                    logger.info(f"正在解压训练数据集文件: {train_file}")
                    if self.extract_zip_file(train_file, train_extract_dir):
                        logger.info(f"成功解压 {train_file}")
                        train_dataset_found = True
                    else:
                        logger.error(f"解压 {train_file} 失败")
            else:
                logger.warning("未找到训练数据集文件")
            
            # 提取验证数据集
            if val_files:
                logger.info(f"找到验证数据集文件: {val_files}")
                val_extract_dir = os.path.join(self.paths.temp_dataset_dir, "AgriculturalDisease_validationset")
                os.makedirs(val_extract_dir, exist_ok=True)
                
                for val_file in val_files:
                    logger.info(f"正在解压验证数据集文件: {val_file}")
                    if self.extract_zip_file(val_file, val_extract_dir):
                        logger.info(f"成功解压 {val_file}")
                        val_dataset_found = True
                    else:
                        logger.error(f"解压 {val_file} 失败")
            else:
                logger.warning("未找到验证数据集文件")
            
            # 检查是否找到了训练和验证数据集注释文件
            train_anno_path = normalize_path(os.path.join(train_extract_dir, "AgriculturalDisease_train_annotations.json"))
            val_anno_path = normalize_path(os.path.join(val_extract_dir, "AgriculturalDisease_validation_annotations.json"))
            
            # 检查训练数据集注释
            if os.path.exists(train_anno_path):
                # 复制到标准位置
                target_train_anno = normalize_path(os.path.join(self.paths.temp_labels_dir, "AgriculturalDisease_train_annotations.json"))
                os.makedirs(os.path.dirname(target_train_anno), exist_ok=True)
                shutil.copy(train_anno_path, target_train_anno)
                logger.info(f"已复制训练数据集注释到: {target_train_anno}")
            else:
                logger.warning(f"未找到训练数据集注释文件: {train_anno_path}")
            
            # 检查验证数据集注释
            if os.path.exists(val_anno_path):
                # 复制到标准位置
                target_val_anno = normalize_path(os.path.join(self.paths.temp_labels_dir, "AgriculturalDisease_validation_annotations.json"))
                os.makedirs(os.path.dirname(target_val_anno), exist_ok=True)
                shutil.copy(val_anno_path, target_val_anno)
                logger.info(f"已复制验证数据集注释到: {target_val_anno}")
            else:
                logger.warning(f"未找到验证数据集注释文件: {val_anno_path}")
            
            return train_dataset_found or val_dataset_found
            
        except Exception as e:
            logger.error(f"提取数据集时出错: {str(e)}")
            traceback.print_exc()
            return False

    def cleanup_temp_files(self, force=False) -> None:
        """清理临时文件和目录
        
        删除处理完成后不再需要的临时文件和目录，包括：
        - 临时解压目录
        - 临时图像目录
        - 临时标签目录
        
        参数:
            force: 是否强制删除而不询问
        """
        try:
            # 要清理的目录列表 - 确保使用config.py中的路径
            dirs_to_clean = [
                self.paths.temp_dataset_dir,
                self.paths.temp_images_dir,
                self.paths.temp_labels_dir
            ]
            
            # 检查是否存在要清理的目录
            existing_dirs = [dir_path for dir_path in dirs_to_clean if os.path.exists(dir_path)]
            
            if not existing_dirs:
                logger.info("No temporary directories found to clean up")
                return
            
            # 展示要删除的目录
            logger.info(f"The following temporary directories can be cleaned up:")
            for i, dir_path in enumerate(existing_dirs, 1):
                dir_size = self._get_directory_size(dir_path)
                file_count = sum(len(files) for _, _, files in os.walk(dir_path))
                logger.info(f"  {i}. {dir_path} ({self._format_size(dir_size)}, {file_count} files)")
            
            # 如果不是强制模式，询问用户是否删除
            should_delete = True
            if not force:
                try:
                    user_input = input("\nDo you want to delete these temporary files? (y/n): ").strip().lower()
                    should_delete = user_input in ['y', 'yes']
                    if not should_delete:
                        logger.info("Cleanup cancelled by user")
                        return
                except (KeyboardInterrupt, EOFError):
                    logger.info("Cleanup cancelled (interrupted)")
                    return
            
            # 用户确认删除，执行清理
            for dir_path in existing_dirs:
                logger.info(f"Cleaning up directory: {dir_path}")
                try:
                    # 尝试删除整个目录树
                    shutil.rmtree(dir_path)
                    logger.info(f"Successfully removed directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Error removing directory {dir_path}: {str(e)}")
                    
                    # 如果整个目录删除失败，尝试删除其中的文件
                    try:
                        for root, dirs, files in os.walk(dir_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    os.remove(file_path)
                                except Exception as file_e:
                                    logger.error(f"Failed to remove file {file_path}: {str(file_e)}")
                        logger.info(f"Cleaned up files in {dir_path}")
                    except Exception as walk_e:
                        logger.error(f"Error walking directory {dir_path}: {str(walk_e)}")
            
            logger.info("Temporary files cleanup completed")
        except Exception as e:
            logger.error(f"Error during temporary files cleanup: {str(e)}")
            traceback.print_exc()
    
    def check_for_cleanable_data(self, force=False) -> None:
        """检查并清理可清理的数据目录
        
        在数据集预处理完成后检查并提示清理不再需要的数据文件夹
        
        参数:
            force: 是否强制删除而不询问
        """
        try:
            # 可清理的数据目录列表
            cleanable_dirs = []
            
            # 获取数据状态
            data_status = self.get_data_status()
            
            # 如果训练和测试数据已经处理完毕，可以清理临时目录
            # 使用正确的键检查数据状态
            if data_status["processed_details"].get("training", False) and data_status["processed_details"].get("testing", False):
                # 添加临时目录
                cleanable_dirs.extend([
                    self.paths.temp_dataset_dir,
                    self.paths.temp_images_dir,
                    self.paths.temp_labels_dir
                ])
                
                # 如果数据已经合并，原始训练和测试目录也可以考虑清理
                if data_status["merged_details"].get("training", False) and data_status["merged_details"].get("testing", False):
                    if os.path.exists(self.paths.train_dir) and os.path.exists(self.paths.merged_train_dir):
                        cleanable_dirs.append(self.paths.train_dir)
                    if os.path.exists(self.paths.test_dir) and os.path.exists(self.paths.merged_test_dir):
                        cleanable_dirs.append(self.paths.test_dir)
            
            # 如果只有测试数据准备好，至少可以清理测试相关的临时目录
            elif data_status["processed_details"].get("testing", False):
                cleanable_dirs.append(self.paths.temp_dataset_dir)
            
            # 筛选出实际存在且非空的目录
            existing_cleanable_dirs = []
            for dir_path in cleanable_dirs:
                if os.path.exists(dir_path):
                    # 检查目录是否有内容
                    dir_size = self._get_directory_size(dir_path)
                    file_count = sum(len(files) for _, _, files in os.walk(dir_path))
                    if dir_size > 0 or file_count > 0:
                        existing_cleanable_dirs.append(dir_path)
            
            if not existing_cleanable_dirs:
                logger.info("No cleanable data directories found")
                return
            
            # 展示可清理的目录
            logger.info(f"The following data directories can be cleaned up:")
            for i, dir_path in enumerate(existing_cleanable_dirs, 1):
                dir_size = self._get_directory_size(dir_path)
                file_count = sum(len(files) for _, _, files in os.walk(dir_path))
                logger.info(f"  {i}. {dir_path} ({self._format_size(dir_size)}, {file_count} files)")
            
            # 如果不是强制模式，询问用户是否删除
            should_delete = True
            if not force:
                try:
                    user_input = input("\nDo you want to delete these data directories? (y/n): ").strip().lower()
                    should_delete = user_input in ['y', 'yes']
                    if not should_delete:
                        logger.info("Cleanup cancelled by user")
                        return
                except (KeyboardInterrupt, EOFError):
                    logger.info("Cleanup cancelled (interrupted)")
                    return
            
            # 用户确认删除，执行清理
            for dir_path in existing_cleanable_dirs:
                logger.info(f"Cleaning up directory: {dir_path}")
                try:
                    # 尝试删除整个目录树
                    shutil.rmtree(dir_path)
                    logger.info(f"Successfully removed directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Error removing directory {dir_path}: {str(e)}")
            
            logger.info("Data directories cleanup completed")
        except Exception as e:
            logger.error(f"Error during data directories cleanup: {str(e)}")
            traceback.print_exc()
    
    def _get_directory_size(self, dir_path) -> int:
        """获取目录大小
        
        参数:
            dir_path: 目录路径
            
        返回:
            目录大小（字节）
        """
        total_size = 0
        for dirpath, _, filenames in os.walk(dir_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp) and not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size
    
    def _format_size(self, size_bytes) -> str:
        """格式化字节大小为人类可读格式
        
        参数:
            size_bytes: 字节大小
            
        返回:
            格式化后的大小字符串
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def merge_datasets(self, merge_type='all'):
        """合并数据集
        
        参数:
            merge_type: 要合并的数据集类型('train', 'test', 'val', 'all')
            
        返回:
            布尔值，指示合并是否成功
        """
        try:
            success = False
            
            # 根据指定的类型合并数据集
            if merge_type in ['train', 'all']:
                logger.info("Merge the training dataset...")
                # 获取所有可能的训练数据源
                train_sources = []
                if os.path.exists(self.paths.train_dir) and os.path.isdir(self.paths.train_dir):
                    train_sources.append(self.paths.train_dir)
                if config.merge_augmented_data and os.path.exists(self.paths.aug_train_dir) and os.path.isdir(self.paths.aug_train_dir):
                    train_sources.append(self.paths.aug_train_dir)
                
                # 确保目标目录存在
                os.makedirs(self.paths.merged_train_dir, exist_ok=True)
                
                # 合并训练数据
                if train_sources:
                    for source in train_sources:
                        logger.info(f"The training data sources are being merged: {source}")
                        # 对于训练数据，需要保持类别目录结构
                        for class_dir in [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]:
                            source_class_dir = os.path.join(source, class_dir)
                            target_class_dir = os.path.join(self.paths.merged_train_dir, class_dir)
                            os.makedirs(target_class_dir, exist_ok=True)
                            
                            # 复制图像文件
                            for ext in ['*.jpg', '*.jpeg', '*.png']:
                                image_files = glob.glob(os.path.join(source_class_dir, ext))
                                for image_file in image_files:
                                    filename = os.path.basename(image_file)
                                    target_file = os.path.join(target_class_dir, filename)
                                    if not os.path.exists(target_file):
                                        shutil.copy2(image_file, target_file)
                    
                    # 检查合并后的训练数据
                    merged_images = glob.glob(os.path.join(self.paths.merged_train_dir, "**/*.jpg"), recursive=True)
                    merged_images += glob.glob(os.path.join(self.paths.merged_train_dir, "**/*.jpeg"), recursive=True)
                    merged_images += glob.glob(os.path.join(self.paths.merged_train_dir, "**/*.png"), recursive=True)
                    logger.info(f"The merged training dataset contains {len(merged_images)} images")
                    success = True
                else:
                    logger.warning("No merged training data sources were found")
            
            if merge_type in ['test', 'all']:
                logger.info("Merge the test dataset...")
                # 获取所有可能的测试数据源
                test_sources = []
                if os.path.exists(self.paths.test_images_dir) and os.path.isdir(self.paths.test_images_dir):
                    test_sources.append(self.paths.test_images_dir)
                
                # 确保目标目录存在
                os.makedirs(self.paths.merged_test_dir, exist_ok=True)
                
                # 合并测试数据
                if test_sources:
                    for source in test_sources:
                        logger.info(f"正在合并测试数据源: {source}")
                        # 对于测试数据，直接复制图像文件
                        for ext in ['*.jpg', '*.jpeg', '*.png']:
                            image_files = glob.glob(os.path.join(source, ext))
                            for image_file in image_files:
                                filename = os.path.basename(image_file)
                                target_file = os.path.join(self.paths.merged_test_dir, filename)
                                if not os.path.exists(target_file):
                                    shutil.copy2(image_file, target_file)
                    
                    # 检查合并后的测试数据
                    merged_images = glob.glob(os.path.join(self.paths.merged_test_dir, "*.jpg"))
                    merged_images += glob.glob(os.path.join(self.paths.merged_test_dir, "*.jpeg"))
                    merged_images += glob.glob(os.path.join(self.paths.merged_test_dir, "*.png"))
                    logger.info(f"The merged test dataset includes {len(merged_images)} images")
                    success = True
                else:
                    logger.warning("No test data sources that can be merged were found")
            
            if merge_type in ['val', 'all']:
                logger.info("Merge the validation dataset...")
                # 获取所有可能的验证数据源
                val_sources = []
                if os.path.exists(self.paths.val_dir) and os.path.isdir(self.paths.val_dir):
                    val_sources.append(self.paths.val_dir)
                
                # 确保目标目录存在
                os.makedirs(self.paths.merged_val_dir, exist_ok=True)
                
                # 合并验证数据
                if val_sources:
                    for source in val_sources:
                        logger.info(f"The data sources for verification are being merged: {source}")
                        # 对于验证数据，需要保持类别目录结构
                        for class_dir in [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]:
                            source_class_dir = os.path.join(source, class_dir)
                            target_class_dir = os.path.join(self.paths.merged_val_dir, class_dir)
                            os.makedirs(target_class_dir, exist_ok=True)
                            
                            # 复制图像文件
                            for ext in ['*.jpg', '*.jpeg', '*.png']:
                                image_files = glob.glob(os.path.join(source_class_dir, ext))
                                for image_file in image_files:
                                    filename = os.path.basename(image_file)
                                    target_file = os.path.join(target_class_dir, filename)
                                    if not os.path.exists(target_file):
                                        shutil.copy2(image_file, target_file)
                    
                    # 检查合并后的验证数据
                    merged_images = glob.glob(os.path.join(self.paths.merged_val_dir, "**/*.jpg"), recursive=True)
                    merged_images += glob.glob(os.path.join(self.paths.merged_val_dir, "**/*.jpeg"), recursive=True)
                    merged_images += glob.glob(os.path.join(self.paths.merged_val_dir, "**/*.png"), recursive=True)
                    logger.info(f"The merged validation dataset contains {len(merged_images)} images")
                    success = True
                else:
                    logger.warning("No consolidable validation data sources were found")
            
            return success
            
        except Exception as e:
            logger.error(f"An error occurred when merging the dataset: {str(e)}")
            traceback.print_exc()
            return False