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

    def extract_dataset(self) -> None:
        """解压数据集文件并准备数据"""
        self.setup_directories()
        
        # 检查数据集压缩文件是否存在
        data_dir = self.paths.data_dir
        train_zip = os.path.join(data_dir, "ai_challenger_pdr2018_trainingset_20181023.zip")
        val_zip = os.path.join(data_dir, "ai_challenger_pdr2018_validationset_20181023.zip")
        
        # 查找所有测试集ZIP文件
        test_zips = glob.glob(os.path.join(data_dir, "ai_challenger_pdr2018_test*.zip"))
        
        # 检查并解压训练集
        if os.path.exists(train_zip):
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
            logger.warning(f"Training dataset zip not found: {train_zip}")
        
        # 检查并解压验证集
        if os.path.exists(val_zip):
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
            logger.warning(f"Validation dataset zip not found: {val_zip}")
        
        # 检查并解压所有测试集
        if test_zips:
            logger.info(f"Found {len(test_zips)} test dataset zip files")
            
            for test_zip in test_zips:
                # 创建文件名的唯一目录名称
                zip_basename = os.path.basename(test_zip)
                test_dir_suffix = zip_basename.split('_')[3].split('.')[0]  # 从文件名提取 testa 或 testb 部分
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
            logger.warning(f"No test dataset zip files found in {data_dir}")
            
        # 确保测试目录存在
        os.makedirs(self.paths.test_images_dir, exist_ok=True)

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

    def get_data_status(self) -> Dict[str, Any]:
        """获取数据准备状态
        
        返回:
            包含数据准备状态信息的字典
        """
        # 检查所需目录是否存在
        train_dir_exists = os.path.exists(self.paths.train_dir)
        test_dir_exists = os.path.exists(self.paths.test_dir)
        temp_images_dir_exists = os.path.exists(self.paths.temp_images_dir)
        temp_labels_dir_exists = os.path.exists(self.paths.temp_labels_dir)
        aug_dir_exists = os.path.exists(self.paths.aug_dir)
        merged_train_dir_exists = os.path.exists(self.paths.merged_train_dir)
        merged_test_dir_exists = os.path.exists(self.paths.merged_test_dir)
        merged_val_dir_exists = os.path.exists(self.paths.merged_val_dir)
        
        # 检查重要文件是否存在
        train_annotations_exists = os.path.exists(self.paths.train_annotations)
        val_annotations_exists = os.path.exists(self.paths.val_annotations)
        
        # 检查ZIP文件
        zip_details = {}
        zip_files = glob.glob(os.path.join(self.paths.data_dir, "*.zip"))
        dataset_extracted = True
        for zip_file in zip_files:
            zip_name = os.path.basename(zip_file)
            # 判断是否已解压
            if "train" in zip_name and not train_dir_exists:
                zip_details[zip_file] = False
                dataset_extracted = False
            elif "test" in zip_name and not test_dir_exists:
                zip_details[zip_file] = False
                dataset_extracted = False
            elif "val" in zip_name and not os.path.exists(self.paths.val_dir):
                zip_details[zip_file] = False
                dataset_extracted = False
            else:
                zip_details[zip_file] = True
        
        # 计算训练集类别数量和测试集图像数量
        training_class_count = 0
        test_image_count = 0
        augmented_image_count = 0
        merged_train_class_count = 0
        merged_test_image_count = 0
        
        if train_dir_exists:
            training_class_count = len([d for d in os.listdir(self.paths.train_dir) 
                                       if os.path.isdir(os.path.join(self.paths.train_dir, d))])
        
        if test_dir_exists:
            test_patterns = ["*.jpg", "*.JPG", "*.png", "*.PNG"]
            for pattern in test_patterns:
                test_image_count += len(glob.glob(os.path.join(self.paths.test_images_dir, pattern)))
        
        if aug_dir_exists:
            aug_patterns = ["*.jpg", "*.JPG", "*.png", "*.PNG"]
            for pattern in aug_patterns:
                augmented_image_count += len(glob.glob(os.path.join(self.paths.aug_train_dir, "**", pattern), recursive=True))
        
        # 计算合并后的数据集信息
        if merged_train_dir_exists:
            merged_train_class_count = len([d for d in os.listdir(self.paths.merged_train_dir) 
                                           if os.path.isdir(os.path.join(self.paths.merged_train_dir, d))])
        
        if merged_test_dir_exists:
            test_patterns = ["*.jpg", "*.JPG", "*.png", "*.PNG"]
            for pattern in test_patterns:
                merged_test_image_count += len(glob.glob(os.path.join(self.paths.merged_test_dir, pattern)))
        
        # 数据处理状态细节
        processed_details = {
            "train": train_dir_exists and training_class_count > 0,
            "test": test_dir_exists and test_image_count > 0,
            "annotations": train_annotations_exists and val_annotations_exists
        }
        
        # 合并数据集状态
        merged_details = {
            "train": merged_train_dir_exists and merged_train_class_count > 0,
            "test": merged_test_dir_exists and merged_test_image_count > 0,
            "val": merged_val_dir_exists
        }
        
        # 综合处理状态
        data_processed = all(processed_details.values())
        
        # 数据增强状态
        augmentation_completed = aug_dir_exists and augmented_image_count > 0
        
        # 数据集合并状态
        merged_datasets = self.config.merge_datasets and all(list(merged_details.values())[:2])  # 只检查train和test
        
        return {
            "training_dir": train_dir_exists,
            "test_dir": test_dir_exists,
            "temp_images_dir": temp_images_dir_exists,
            "temp_labels_dir": temp_labels_dir_exists,
            "train_annotations": train_annotations_exists,
            "val_annotations": val_annotations_exists,
            "training_class_count": training_class_count,
            "test_image_count": test_image_count,
            "augmented_image_count": augmented_image_count,
            "dataset_extracted": dataset_extracted,
            "data_processed": data_processed,
            "processed_details": processed_details,
            "augmentation_completed": augmentation_completed,
            "aug_dir_exists": aug_dir_exists,
            "zip_details": zip_details,
            "merged_train_dir": merged_train_dir_exists,
            "merged_test_dir": merged_test_dir_exists,
            "merged_val_dir": merged_val_dir_exists,
            "merged_train_class_count": merged_train_class_count,
            "merged_test_image_count": merged_test_image_count,
            "merged_details": merged_details,
            "merged_datasets": merged_datasets
        }

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
        removed_count = 0
        error_count = 0
        
        for image_path in Path(image_dir).rglob('*'):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                if not self.is_valid_image(str(image_path)):
                    try:
                        os.remove(str(image_path))
                        removed_count += 1
                        logger.info(f"Removed invalid image: {image_path}")
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Failed to remove {image_path}: {str(e)}")

        logger.info(f"Removed {removed_count} invalid images")
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
            mode: 数据集模式 ('train', 'test', 或 'val')
            force: 是否强制重新创建合并目录
            
        返回:
            合并后的数据集路径
        """
        # 如果在配置中禁用合并功能，则不执行合并
        if not self.config.merge_datasets and not force:
            logger.info(f"Dataset merging is disabled in configuration for {mode} mode")
            # 根据模式返回默认路径
            if mode == "train":
                return self.config.train_data
            elif mode == "test":
                return self.config.test_data
            else:
                return self.config.val_data
                
        # 根据模式确定目标目录
        if mode == "train":
            merged_path = self.paths.merged_train_dir
        elif mode == "test":
            merged_path = self.paths.merged_test_dir
        elif mode == "val":
            merged_path = self.paths.merged_val_dir
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # 如果请求强制重建，删除现有目录
        force_merge = force or self.config.merge_force
        if force_merge and os.path.exists(merged_path):
            shutil.rmtree(merged_path)
            logger.info(f"Removed existing merged directory: {merged_path}")
        
        # 检查目标目录是否已存在，如果存在且不是强制模式，直接返回
        if os.path.exists(merged_path) and not force_merge:
            logger.info(f"Using existing merged {mode} dataset: {merged_path}")
            return merged_path
        
        # 获取可用数据集列表
        datasets = handle_datasets(mode, list_only=True)
        
        # 测试集特殊处理
        if mode == "test" and not self.config.use_all_test_datasets:
            # 过滤出指定的主要测试集
            filtered_datasets = []
            for dataset in datasets:
                if self.config.primary_test_dataset in dataset:
                    filtered_datasets.append(dataset)
                    break
                    
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
        
        if len(datasets) == 1 and not force_merge:
            logger.info(f"Only one {mode} dataset found, no merge needed: {datasets[0]}")
            return datasets[0]
        
        # 确保目标目录存在
        os.makedirs(merged_path, exist_ok=True)
        
        # 执行合并
        logger.info(f"Merging {len(datasets)} {mode} datasets...")
        
        if mode == "train":
            # 合并训练集：将所有类别目录下的图像复制到对应的merged目录
            for source_dir in datasets:
                logger.info(f"Processing training dataset: {source_dir}")
                for class_dir in os.listdir(source_dir):
                    class_path = os.path.join(source_dir, class_dir)
                    if os.path.isdir(class_path):
                        # 创建目标类别目录
                        target_class_dir = os.path.join(merged_path, class_dir)
                        os.makedirs(target_class_dir, exist_ok=True)
                        
                        # 复制图像文件
                        img_count = 0
                        for img in glob.glob(os.path.join(class_path, "*.jpg")) + glob.glob(os.path.join(class_path, "*.png")):
                            # 创建唯一的目标文件名
                            source_name = os.path.basename(source_dir)
                            img_name = f"{source_name}_{os.path.basename(img)}"
                            target_path = os.path.join(target_class_dir, img_name)
                            
                            # 复制文件
                            if not os.path.exists(target_path):
                                shutil.copy2(img, target_path)
                                img_count += 1
                        
                        logger.info(f"  Copied {img_count} images from class {class_dir}")
        else:
            # 合并测试集或验证集：直接将所有图像复制到merged目录
            for source_dir in datasets:
                logger.info(f"Processing {mode} dataset: {source_dir}")
                img_count = 0
                for img in glob.glob(os.path.join(source_dir, "*.jpg")) + glob.glob(os.path.join(source_dir, "*.png")):
                    # 创建唯一的目标文件名
                    source_name = os.path.basename(os.path.dirname(source_dir))
                    img_name = f"{source_name}_{os.path.basename(img)}"
                    target_path = os.path.join(merged_path, img_name)
                    
                    # 复制文件
                    if not os.path.exists(target_path):
                        shutil.copy2(img, target_path)
                        img_count += 1
                
                logger.info(f"  Copied {img_count} images")
        
        # 计算合并后的文件数
        jpg_count = len(glob.glob(os.path.join(merged_path, "**", "*.jpg"), recursive=True))
        png_count = len(glob.glob(os.path.join(merged_path, "**", "*.png"), recursive=True))
        total_files = jpg_count + png_count
        
        logger.info(f"{mode.capitalize()} dataset merge completed: {merged_path} with {total_files} total files")
        return merged_path
    
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
            "train": handle_datasets("train", list_only=True),
            "test": handle_datasets("test", list_only=True),
            "val": handle_datasets("val", list_only=True)
        }
        
        # 恢复配置
        self.config.merge_datasets = old_setting
        
        return datasets 

# 创建一个模块级函数来初始化和运行数据准备
def setup_data(extract=True, process=True, augment=True, status=True, merge=None):
    """设置并处理数据
    
    参数:
        extract: 是否执行解压缩步骤
        process: 是否执行处理步骤
        augment: 是否执行数据增强步骤
        status: 是否检查状态
        merge: 是否合并数据集，None表示使用配置中的设置
    
    返回:
        包含完成状态的字典
    """
    try:
        data_prep = DataPreparation()
        
        if status:
            # 首先检查数据状态
            data_prep.check_data_status()
        
        if extract:
            # 解压数据集
            data_prep.extract_dataset()
            
        if process:
            # 处理数据集
            data_prep.process_data()
            
        if augment and config.use_data_aug:
            # 如果启用了数据增强，执行数据增强
            data_prep.augment_directory()
            
        # 执行数据集合并
        should_merge = merge if merge is not None else config.merge_on_startup
        if should_merge:
            logger.info("Merging datasets as configured...")
            train_path = data_prep.merge_datasets(mode="train")
            test_path = data_prep.merge_datasets(mode="test")
            if os.path.exists(paths.val_dir):
                val_path = data_prep.merge_datasets(mode="val")
        else:
            logger.info("Dataset merging skipped as per configuration or parameters")
            
        return {"completed": True}
    except Exception as e:
        logger.error(f"Error in setup_data: {str(e)}")
        return {"completed": False, "error": str(e)}

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
            merge=do_merge
        )
    
    # 如果存在dataset_merge命令行参数
    elif len(sys.argv) > 1 and sys.argv[1] in ["--train", "--test", "--val", "--all", "--list", "--force"]:
        merge_datasets_cli()
    
    # 默认情况下，执行完整流程
    else:
        setup_data(extract=True, process=True, augment=True, status=True, merge=None) 