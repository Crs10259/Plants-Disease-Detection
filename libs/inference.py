import os
import io
import json
import time
import numpy as np
import torch
import logging
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict
from utils.utils import MyEncoder
from config.config import config, paths
from models.model import get_net

class InferenceManager:
    """植物病害检测推理管理器类"""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, logger=None):
        """初始化推理管理器
        
        参数:
            model_path: 模型权重路径
            device: 使用的设备('cuda', 'cpu'或None表示自动检测)
            logger: 可选的日志记录器实例
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.logger = logger or self._setup_logger()
        
    def _setup_logger(self):
        """设置并返回推理日志记录器"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(paths.inference_log),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('Inference')
    
    def _get_device(self, device_str: Optional[str] = None) -> torch.device:
        """获取适合推理的设备
        
        参数:
            device_str: 设备指定('cuda', 'cpu'或None表示自动)
            
        返回:
            torch.device对象
        """
        if device_str == "cuda":
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        elif device_str == "cpu":
            return torch.device('cpu')
        else:
            # 自动模式
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    def load_model(self, model_path: Optional[str] = None) -> None:
        """从检查点加载模型
        
        参数:
            model_path: 模型权重路径(如果为None则使用self.model_path)
        """
        model_path = model_path or self.model_path
        if model_path is None:
            raise ValueError("No model path provided")
            
        self.logger.info(f"Loading model from {model_path}")
        
        # 初始化模型架构
        model = get_net()
        
        # 加载权重
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同的检查点格式
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
                
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
        # 将模型移至设备并设置为评估模式
        self.model = model.to(self.device)
        self.model.eval()
        
    def predict_single(self, image_path: str) -> np.ndarray:
        """对单张图像进行预测
        
        参数:
            image_path: 图像文件路径
            
        返回:
            类别概率的NumPy数组
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first")
            
        # 准备图像变换
        transform = transforms.Compose([
            transforms.Resize((config.img_height, config.img_weight)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载并变换图像
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
            
        # 进行预测
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.Softmax(dim=1)(output)
            
        return probabilities.cpu().numpy()[0]
    
    def predict_batch(self, image_folder: str, batch_size: int = 16, num_workers: int = 4) -> List[Dict[str, Any]]:
        """对文件夹中的所有图像进行预测
        
        参数:
            image_folder: 包含图像的文件夹路径
            batch_size: 推理的批次大小
            num_workers: 数据加载的工作线程数
            
        返回:
            包含预测结果的字典列表
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first")
            
        # Get image files
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            self.logger.warning(f"No images found in {image_folder}")
            return []
            
        self.logger.info(f"Found {len(image_files)} image files")
        
        # Create dataset and dataloader
        dataset = InferenceDataset(image_files)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        # Make predictions
        results = []
        
        with torch.no_grad():
            for batch_images, batch_paths in tqdm(dataloader, desc="Making predictions"):
                batch_images = batch_images.to(self.device)
                outputs = self.model(batch_images)
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.Softmax(dim=1)(outputs)
                
                # Process predictions
                for i, (probs, path) in enumerate(zip(probabilities, batch_paths)):
                    pred_label = int(torch.argmax(probs).item())
                    
                    # Adjust label index for classes 44 and 45 which are removed
                    if pred_label > 43:
                        pred_label = pred_label + 2
                        
                    # Create result entry
                    results.append({
                        "image_id": os.path.basename(path),
                        "disease_class": pred_label,
                        "probabilities": probs.cpu().numpy().tolist()
                    })
        
        self.logger.info(f"Processed {len(results)} images")
        return results
    
    def save_predictions(self, predictions: List[Dict], output_file: str = paths.prediction_file) -> None:
        """将预测结果保存到JSON文件
        
        参数:
            predictions: 预测字典列表
            output_file: 输出文件路径
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare submission format
        submit_format = []
        for pred in predictions:
            submit_format.append({
                "image_id": pred["image_id"],
                "disease_class": pred["disease_class"]
            })
            
        # Save to file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(submit_format, f, ensure_ascii=False, cls=MyEncoder)
            self.logger.info(f"Predictions saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving predictions: {str(e)}")
            raise

class InferenceDataset(Dataset):
    """推理数据集类"""
    
    def __init__(self, file_paths):
        """初始化数据集
        
        参数:
            file_paths: 图像文件路径列表
        """
        self.file_paths = file_paths
        self.transforms = transforms.Compose([
            transforms.Resize((config.img_height, config.img_weight)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        """获取单个数据样本
        
        参数:
            idx: 索引
            
        返回:
            (图像张量, 图像路径)元组
        """
        img_path = self.file_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transforms(image)
            return image, img_path
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            # Return empty image
            return torch.zeros((3, config.img_height, config.img_weight)), img_path

def predict(model_path: str, input_path: str, output_file: str = paths.prediction_file, is_dir: bool = False) -> List[Dict]:
    """预测函数，支持单张图片或图片目录
    
    参数:
        model_path: 模型权重路径
        input_path: 输入图片或图片目录路径
        output_file: 输出文件路径
        is_dir: 输入是否为目录
        
    返回:
        预测结果字典列表
    """
    logger = logging.getLogger('Inference')
    
    # 初始化推理管理器
    inference = InferenceManager(model_path)
    
    try:
        # 加载模型
        inference.load_model()
        
        # 进行预测
        if is_dir:
            # 检查目录是否存在
            if not os.path.exists(input_path) or not os.path.isdir(input_path):
                logger.error(f"Directory not found or not a directory: {input_path}")
                return []
                
            # 检查目录是否为空
            image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not image_files:
                logger.error(f"No image files found in directory: {input_path}")
                return []
                
            # 目录预测
            logger.info(f"Predicting on {len(image_files)} images in {input_path}")
            predictions = inference.predict_batch(input_path)
        else:
            # 单张图片预测
            if not os.path.exists(input_path) or not os.path.isfile(input_path):
                logger.error(f"File not found or not a file: {input_path}")
                return []
                
            # 对单个图像进行预测并将结果转换为列表格式
            logger.info(f"Predicting on single image: {input_path}")
            pred = inference.predict_single(input_path)
            predictions = [{
                'image_id': os.path.basename(input_path),
                'disease_class': int(np.argmax(pred)),
                'confidence': float(np.max(pred))
            }]
        
        # 保存预测结果
        if output_file and predictions:
            inference.save_predictions(predictions, output_file)
            logger.info(f"Saved predictions to {output_file}")
            
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return [] 