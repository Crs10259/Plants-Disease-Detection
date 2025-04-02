import random 
import numpy as np 
import pandas as pd 
import torch 
import os
from itertools import chain 
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as T 
from config import config
from PIL import Image 
from concurrent.futures import ThreadPoolExecutor

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
        valid_imgs = []
        invalid_imgs = []
        
        def validate_image(img_data):
            """验证单个图像是否可读
            
            参数:
                img_data: (文件名, 标签)元组
                
            返回:
                (是否有效, 图像数据)元组
            """
            try:
                filename = img_data[0]
                # 只验证文件头，不完整加载图像
                with Image.open(filename) as img:
                    img.verify()
                return True, img_data
            except Exception:
                return False, img_data
        
        # 使用多线程并行验证图像
        print("Validating images...")
        with ThreadPoolExecutor(max_workers=config.aug_num_workers) as executor:
            # 使用tqdm显示进度
            results = list(tqdm(
                executor.map(validate_image, imgs),
                total=len(imgs),
                desc="Validating images"
            ))
        
        # 分离有效和无效图像
        for is_valid, img_data in results:
            if is_valid:
                valid_imgs.append(img_data)
            else:
                invalid_imgs.append(img_data)
        
        if invalid_imgs:
            print(f"\nFound {len(invalid_imgs)} unreadable images that will be skipped:")
            for _, (filename, _) in enumerate(invalid_imgs[:5], 1):
                print(f"  {filename}")
            if len(invalid_imgs) > 5:
                print(f"  ... and {len(invalid_imgs) - 5} more")
            
        print(f"Successfully loaded {len(valid_imgs)} valid images")
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
        
        # 训练模式额外的数据增强
        if not test and train:
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
        mode: 'train'或'test'模式
        
    返回:
        包含文件路径和标签的DataFrame
    """
    if not os.path.exists(root):
        raise FileNotFoundError(f"Directory not found: {root}")
    
    if mode == "test":
        files = [os.path.join(root, img) for img in os.listdir(root)]
        return pd.DataFrame({"filename": files})
        
    elif mode == "train":
        all_data_path, labels = [], []
        image_folders = [os.path.join(root, x) for x in os.listdir(root)]
        
        # 获取所有jpg图像路径
        jpg_patterns = ['/*.jpg', '/*.JPG']
        all_images = []
        for folder in image_folders:
            for pattern in jpg_patterns:
                all_images.extend(glob(folder + pattern))
                
        print("Loading training dataset")
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
        raise ValueError("Mode must be either 'train' or 'test'")
    
