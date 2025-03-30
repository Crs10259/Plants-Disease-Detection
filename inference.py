import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict
from utils import MyEncoder
from config import config
from models.model import get_net

class InferenceDataset(Dataset):
    """推理数据集类"""
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.transforms = transforms.Compose([
            transforms.Resize((config.img_height, config.img_weight)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transforms(image)
            return image, img_path
        except Exception as e:
            print(f"加载图像时出错 {img_path}: {str(e)}")
            # 返回空图像
            return torch.zeros((3, config.img_height, config.img_weight)), img_path

def predict(model_path, image_folder, output_file="./submit/prediction.json"):
    """使用训练好的模型进行预测
    
    参数:
        model_path: 模型路径
        image_folder: 图像文件夹路径
        output_file: 输出文件路径
    """
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = get_net()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    
    # 获取图像文件
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 创建数据集和数据加载器
    dataset = InferenceDataset(image_files)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # 预测并保存结果
    csv_map = OrderedDict({"filename": [], "probability": []})
    
    with torch.no_grad(), open(output_file, "w", encoding="utf-8") as f:
        submit_results = []
        
        for batch_images, batch_paths in tqdm(dataloader, desc="预测中"):
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            
            # 应用softmax得到概率
            probabilities = torch.nn.Softmax(dim=1)(outputs)
            
            # 保存文件名和概率
            batch_paths = [os.path.basename(path) for path in batch_paths]
            csv_map["filename"].extend(batch_paths)
            
            for output in probabilities:
                prob = ";".join([str(i) for i in output.data.tolist()])
                csv_map["probability"].append(prob)
                
        # 处理结果
        result = pd.DataFrame(csv_map)
        result["probability"] = result["probability"].map(lambda x: [float(i) for i in x.split(";")])
        
        # 创建提交结果
        for index, row in result.iterrows():
            pred_label = np.argmax(row['probability'])
            if pred_label > 43:
                pred_label = pred_label + 2
            submit_results.append({"image_id": row['filename'], "disease_class": pred_label})
            
        json.dump(submit_results, f, ensure_ascii=False, cls=MyEncoder)
    
    print(f"预测完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模型推理脚本")
    parser.add_argument('--model', required=True, help='模型路径')
    parser.add_argument('--input', required=True, help='图像文件夹路径')
    parser.add_argument('--output', default='./submit/prediction.json', help='输出文件路径')
    
    args = parser.parse_args()
    
    predict(args.model, args.input, args.output) 