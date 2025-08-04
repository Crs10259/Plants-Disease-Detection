import random 
import numpy as np 
import pandas as pd 
import torch 
import os
import logging
from itertools import chain 
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as T 
from config import config, paths
from PIL import Image 
from concurrent.futures import ThreadPoolExecutor
from utils.utils import handle_datasets
import concurrent.futures

# 设置日志记录器
logger = logging.getLogger('DataLoader')
logger.setLevel(logging.INFO)

# 创建日志处理器
if not logger.handlers:
    # 文件处理器
    if not os.path.exists(paths.log_dir):
        os.makedirs(paths.log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(paths.log_dir, 'dataloader.log'))
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器并添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# 设置随机种子
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

class PlantDiseaseDataset(Dataset):
    """植物病害图像数据集类"""
    def __init__(self, label_list, transforms=None, train=True, test=False):
        """初始化数据集
        
        参数:
            label_list: 包含文件路径和标签的DataFrame
            transforms: 数据增强转换
            train: 是否为训练模式
            test: 是否为测试模式
        """
        self.test = test 
        self.train = train 
        self.transforms = self._get_transforms(transforms, train, test)
        self.imgs = self._load_images(label_list)
        
    def _load_images(self, label_list):
        """加载并验证图像
        
        参数:
            label_list: 包含文件路径和标签的DataFrame
            
        返回:
            有效的图像数据列表
        """
        if self.test:
            return [(row["filename"]) for _, row in label_list.iterrows()]
        
        # 将DataFrame转换为列表以提高处理速度
        imgs = list(zip(label_list["filename"], label_list["label"]))
        
        # 对于大数据集，打印警告并建议采样
        if len(imgs) > 50000:
            print(f"Warning: The dataset is very large ({len(imgs)} images), it may cause memory problems")
            print("Hint: Consider using smaller batch size or reducing the dataset size.")

        
        valid_imgs = []
        invalid_imgs = []
        
        def validate_image_batch(batch):
            """验证一批图像
            
            参数:
                batch: 图像数据(文件名, 标签)元组的列表
                
            返回:
                (有效图像列表, 无效图像列表)元组
            """
            valid = []
            invalid = []
            
            for img_data in batch:
                try:
                    filename = img_data[0]
                    # 检查文件是否存在
                    if not os.path.exists(filename):
                        invalid.append(img_data)
                        continue
                        
                    # 检查文件大小，跳过过小文件
                    file_size = os.path.getsize(filename)
                    if file_size < 100:  # 小于100字节的文件几乎不可能是有效图像
                        invalid.append(img_data)
                        continue
                        
                    # 只验证文件头，不完整加载图像
                    with Image.open(filename) as img:
                        img.verify()
                        valid.append(img_data)
                except Exception:
                    invalid.append(img_data)
            
            return valid, invalid
        
        # 使用多线程并行验证图像
        print("Validating images...")
        
        # 更高效的批处理配置
        batch_size = 100  # 更大的批次以减少线程创建开销
        max_workers = min(4, os.cpu_count())  # 限制线程数以减少内存压力
        
        # 将图像分成批次
        batches = []
        for i in range(0, len(imgs), batch_size):
            batches.append(imgs[i:min(i + batch_size, len(imgs))])
        
        # 使用线程池并行处理所有批次
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用tqdm显示进度
            futures = []
            for batch in batches:
                futures.append(executor.submit(validate_image_batch, batch))
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc="Validating images"):
                try:
                    batch_valid, batch_invalid = future.result()
                    valid_imgs.extend(batch_valid)
                    invalid_imgs.extend(batch_invalid)
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
        
        # 对于极大的数据集，进行采样以减少内存压力
        if config.enable_sampling and len(valid_imgs) > config.sampling_threshold:
            print(f"\nThe dataset is very large ({len(valid_imgs)} effective images), Conduct sampling to reduce memory usage")
            import random
            random.seed(config.seed)  # 保持一致性
            valid_imgs = random.sample(valid_imgs, config.sample_size)
            print(f"The size of the sampled dataset: {len(valid_imgs)} images")
        
        if invalid_imgs:
            print(f"\nFound {len(invalid_imgs)} unreadable images that will be skipped:")
            for _, (filename, _) in enumerate(invalid_imgs[:5], 1):
                print(f"  {filename}")
            if len(invalid_imgs) > 5:
                print(f"  ... and {len(invalid_imgs) - 5} more")
            
        print(f"Successfully loaded {len(valid_imgs)} valid images")
        
        # 强制清理内存
        import gc
        gc.collect()
        
        return valid_imgs
        
    def _get_transforms(self, transforms, train, test):
        """获取数据转换操作
        
        参数:
            transforms: 自定义转换
            train: 是否为训练模式
            test: 是否为测试模式
            
        返回:
            转换操作序列
        """
        if transforms is not None:
            return transforms
            
        # 基础转换
        base_transforms = [
            T.Resize((config.img_weight, config.img_height)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        
        # 训练模式额外的数据增强，同时检查是否全局启用数据增强
        if not test and train and config.use_data_aug:
            train_transforms = [
                T.RandomRotation(30),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(45)
            ]
            base_transforms[1:1] = train_transforms  # 在ToTensor之前插入训练时的转换
            
        return T.Compose(base_transforms)

    def __getitem__(self, index):
        """获取单个数据样本
        
        参数:
            index: 索引
            
        返回:
            (图像张量, 标签)或(图像张量, 文件名)
        """
        try:
            if self.test:
                filename = self.imgs[index]
                img = Image.open(filename)
                img_tensor = self.transforms(img)
                return img_tensor, filename
            else:
                filename, label = self.imgs[index]
                img = Image.open(filename)
                img_tensor = self.transforms(img)
                return img_tensor, label
        except Exception as e:
            print(f"Error loading image at index {index}: {str(e)}")
            # 返回空张量作为错误处理
            return (torch.zeros((3, config.img_height, config.img_weight)), 
                   self.imgs[index][1] if not self.test else self.imgs[index])
                
    def __len__(self):
        """返回数据集大小"""
        return len(self.imgs)

def collate_fn(batch):
    """批次数据收集函数
    
    参数:
        batch: 批次数据
        
    返回:
        (图像张量堆叠, 标签列表)
    """
    imgs, labels = zip(*batch)
    return torch.stack(imgs, 0), list(labels)

def get_files(root, mode):
    """获取数据集文件路径和标签
    
    参数:
        root: 根目录路径
        mode: 'train', 'val' 或 'test' 模式
        
    返回:
        包含文件路径和标签的DataFrame
    """
    # 处理多数据集情况，获取实际使用的数据集路径
    actual_root = handle_datasets(data_type=mode)
    
    if not os.path.exists(actual_root):
        raise FileNotFoundError(f"Directory not found: {actual_root}")
    
    logger.info(f"Loading {mode} dataset from: {actual_root}")
    
    if mode == "test":
        files = [os.path.join(actual_root, img) for img in os.listdir(actual_root) 
                if img.endswith(('.jpg', '.JPG', '.png', '.PNG'))]
        return pd.DataFrame({"filename": files})
        
    elif mode in ["train", "val"]: 
        all_data_path, labels = [], []
        image_folders = [os.path.join(actual_root, x) for x in os.listdir(actual_root) 
                        if os.path.isdir(os.path.join(actual_root, x))]
        
        # 获取所有jpg和png图像路径
        image_patterns = ['/*.jpg', '/*.JPG', '/*.png', '/*.PNG']
        all_images = []
        for folder in image_folders:
            for pattern in image_patterns:
                all_images.extend(glob(folder + pattern))
                
        logger.info(f"Loading {mode} dataset ({len(all_images)} images)")
        for file in tqdm(all_images):
            all_data_path.append(file)
            # 从路径中提取标签
            label = int(os.path.basename(os.path.dirname(file)))
            labels.append(label)
            
        return pd.DataFrame({
            "filename": all_data_path,
            "label": labels
        })
        
    else:
        raise ValueError("Mode must be one of 'train', 'val', or 'test'") 