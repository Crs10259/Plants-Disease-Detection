import concurrent.futures
import json
import shutil
import os
from tqdm import tqdm
import argparse

def setup_directories():
    """创建项目所需的所有目录结构"""
    # 创建训练数据目录
    try:
        for i in range(0, 59):
            os.makedirs(os.path.join("./data/train/", str(i)), exist_ok=True)
        
        # 创建其他必要目录
        os.makedirs("./data/test/images/", exist_ok=True)
        os.makedirs("./submit/", exist_ok=True)
        os.makedirs("./logs/", exist_ok=True)
        os.makedirs("./checkpoints/", exist_ok=True)
        os.makedirs("./checkpoints/best_model/", exist_ok=True)
        
        print("All directories created successfully")
    except Exception as e:
        print(f"Error creating directories: {str(e)}")

def copy_file(file):
    """复制单个文件到指定目录
    
    参数:
        file: 包含图像文件信息的字典
        
    返回:
        布尔值，指示是否成功处理
    """
    try:
        filename = file["image_id"]
        origin_path = os.path.join("data", "temp", "images", filename)
        
        # 跳过类别44和45
        ids = file["disease_class"]
        if ids == 44 or ids == 45:
            return False
            
        # 调整ID顺序（大于45的ID要减2）
        if ids > 45:
            ids -= 2
            
        # 确保目标目录存在
        save_dir = os.path.join("data", "train", str(ids))
        os.makedirs(save_dir, exist_ok=True)
        
        # 构建目标路径并复制文件
        save_path = os.path.join(save_dir, filename)
        shutil.copy(origin_path, save_path)
        return True
    except Exception as e:
        print(f"Error processing file {file.get('image_id', 'unknown')}: {str(e)}")
        return False

def move_files_to_folder(source_folder, destination_folder):
    """移动源文件夹中的所有文件到目标文件夹
    
    参数:
        source_folder: 源文件夹路径
        destination_folder: 目标文件夹路径
    """
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹中的所有文件
    files_moved = 0
    for filename in os.listdir(source_folder):
        # 拼接完整的文件路径
        source_file_path = os.path.join(source_folder, filename)
        
        # 检查是否为文件
        if os.path.isfile(source_file_path):
            # 拼接目标文件的完整路径
            destination_file_path = os.path.join(destination_folder, filename)
            
            # 执行文件移动操作
            shutil.move(source_file_path, destination_file_path)
            files_moved += 1
    
    print(f'Moved {files_moved} files from {source_folder} to {destination_folder}')

def process_data():
    """处理数据集文件"""
    # 设置目录
    setup_directories()
    
    # 加载标注文件
    try:
        file_train = json.load(open("./data/temp/labels/AgriculturalDisease_train_annotations.json", "r", encoding="utf-8"))
        file_val = json.load(open("./data/temp/labels/AgriculturalDisease_validation_annotations.json", "r", encoding="utf-8"))
        file_list = file_train + file_val
    except FileNotFoundError:
        print("Annotation files not found, please ensure data is correctly placed in ./data/temp/ directory")
        return
    except json.JSONDecodeError:
        print("Invalid annotation file format, please check JSON file validity")
        return
    
    print(f"Found {len(file_list)} annotation files")
    
    # 统计每个类别的文件数
    class_counts = {}
    for file in file_list:
        class_id = file["disease_class"]
        if class_id not in class_counts:
            class_counts[class_id] = 0
        class_counts[class_id] += 1
    
    # 显示每个类别的文件数
    print("\nClass distribution:")
    for class_id, count in sorted(class_counts.items()):
        print(f"Class {class_id}: {count} files")
    
    # 检查是否有特殊类别44和45
    if 44 in class_counts or 45 in class_counts:
        print("\nWarning: Classes 44 and 45 will be ignored, classes >45 will be reduced by 2")
    
    # 创建线程池
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # 使用tqdm监控进度
        futures = [executor.submit(copy_file, file) for file in file_list]
        
        # 更新进度条并处理异常
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                print(f"Error processing file: {str(e)}")
    
    print(f"\nSuccessfully processed {success_count}/{len(file_list)} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset file processing tool")
    parser.add_argument('--move', help='Move files from source directory to target directory', nargs=2, metavar=('source_dir', 'target_dir'))
    
    args = parser.parse_args()
    
    if args.move:
        move_files_to_folder(args.move[0], args.move[1])
    else:
        process_data()