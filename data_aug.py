from PIL import Image, ImageEnhance
import os
import numpy as np
import cv2
import random
from skimage.util import random_noise
from skimage import exposure
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from config import config

class DataAugmentor:
    """数据增强器类"""
    def __init__(self, config):
        """初始化数据增强器
        
        参数:
            config: 配置对象，包含增强参数
        """
        self.config = config
        self.error_images = []

    def add_noise(self, img):
        """添加高斯噪声到图像"""
        if img is None:
            return None
        try:
            img_float = img.astype(np.float32) / 255.0
            noisy = random_noise(img_float, mode='gaussian', 
                               var=self.config.aug_noise_var, 
                               clip=True)
            noisy = (noisy * 255).astype(np.uint8)
            return noisy
        except Exception as e:
            print(f"Error adding noise: {str(e)}")
            return None

    def change_brightness(self, img):
        """调整图像亮度"""
        if img is None:
            return None
        try:
            rate = random.uniform(*self.config.aug_brightness_range)
            img_float = img.astype(np.float32) / 255.0
            adjusted = exposure.adjust_gamma(img_float, rate)
            adjusted = (adjusted * 255).astype(np.uint8)
            return adjusted
        except Exception as e:
            print(f"Error adjusting brightness: {str(e)}")
            return None

    def is_valid_image(self, image_path):
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
            return True
        except Exception as e:
            print(f"Invalid image {image_path}: {str(e)}")
            self.error_images.append(image_path)
            return False

    def remove_error_images(self, image_dir):
        """删除错误的图像文件
        
        参数:
            image_dir: 图像目录路径
        """
        if not self.config.remove_error_images:
            return

        print("Checking for invalid images...")
        for root, _, files in os.walk(image_dir):
            for file in tqdm(files):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    if not self.is_valid_image(image_path):
                        try:
                            os.remove(image_path)
                            print(f"Removed invalid image: {image_path}")
                        except Exception as e:
                            print(f"Failed to remove {image_path}: {str(e)}")

        if self.error_images:
            print(f"Removed {len(self.error_images)} invalid images")

    def augment_image(self, image_path, save_dir):
        """对单张图像进行数据增强"""
        try:
            if not os.path.exists(image_path):
                return []

            img = Image.open(image_path)
            cv_image = cv2.imread(image_path)
            
            if cv_image is None:
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

            return augmented_images
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return []

    def augment_directory(self, source_dir=None, target_dir=None):
        """对整个目录进行数据增强"""
        if not self.config.use_data_aug:
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
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))

        print(f"Augmenting {len(image_files)} images...")
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
                    print(f"Error in augmentation: {str(e)}")

        print(f"Created {len(augmented_files)} augmented images")
        return augmented_files

def check_augmented_images_exist(image_path, saved_path):
    """检查增强版本的图像是否已存在
    
    参数:
        image_path: 原图像路径
        saved_path: 保存目录
        
    返回:
        布尔值，指示是否所有增强版本都已存在
    """
    basename = os.path.basename(image_path)
    augmented_names = [
        basename,  # 原图
        f"gau_{basename}",  # 高斯噪声
        f"light_{basename}",  # 亮度调整
        f"gau_light_{basename}",  # 亮度+噪声
        f"left_right_{basename}",  # 左右翻转
        f"top_bottom_{basename}",  # 上下翻转
        f"contrasted_{basename}"  # 对比度增强
    ]
    
    # 检查所有增强版本是否都存在
    return all(os.path.exists(os.path.join(saved_path, name)) for name in augmented_names)

def is_chinese(string):
    """检查字符串是否包含中文字符
    
    参数:
        string: 要检查的字符串
        
    返回:
        布尔值，指示是否包含中文字符
    """
    for char in string:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def process_image(image_path, saved_path):
    """处理单张图像进行数据增强
    
    参数:
        image_path: 图像路径
        saved_path: 保存目录
        
    返回:
        布尔值，指示处理是否成功
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return False
            
        # 检查是否已经处理过这张图片
        if check_augmented_images_exist(image_path, saved_path):
            return True
            
        # 读取图片
        img = Image.open(image_path)
        cv_image = cv2.imread(image_path)
        
        if cv_image is None:
            print(f"OpenCV cannot read image: {image_path}")
            return False

        data_augmentor = DataAugmentor(config)
        # 高斯噪声
        gau_image = data_augmentor.add_noise(cv_image)
        if gau_image is not None:
            cv2.imwrite(os.path.join(saved_path, f"gau_{os.path.basename(image_path)}"), gau_image)
        
        # 随机改变亮度
        light_image = data_augmentor.change_brightness(cv_image)
        if light_image is not None:
            cv2.imwrite(os.path.join(saved_path, f"light_{os.path.basename(image_path)}"), light_image)
            
        # 亮度调整后添加噪声
        if light_image is not None:
            light_and_gau = data_augmentor.add_noise(light_image)
            if light_and_gau is not None:
                cv2.imwrite(os.path.join(saved_path, f"gau_light_{os.path.basename(image_path)}"), light_and_gau)
        
        # PIL图像增强
        img_flip_left_right = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_flip_top_bottom = img.transpose(Image.FLIP_TOP_BOTTOM)

        # 对比度增强
        enh_con = ImageEnhance.Contrast(img)
        contrast = 1.5
        image_contrasted = enh_con.enhance(contrast)

        # 保存增强后的图片
        img.save(os.path.join(saved_path, os.path.basename(image_path)))
        img_flip_left_right.save(os.path.join(saved_path, f"left_right_{os.path.basename(image_path)}"))
        img_flip_top_bottom.save(os.path.join(saved_path, f"top_bottom_{os.path.basename(image_path)}"))
        image_contrasted.save(os.path.join(saved_path, f"contrasted_{os.path.basename(image_path)}"))
        
        return True
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return False

def process_data_with_naming_issues(renamed_files):
    """处理命名有问题的文件
    
    参数:
        renamed_files: 包含中文字符的文件路径列表
    """
    if not renamed_files:
        return
        
    print("\nFound files with naming issues (containing Chinese characters):")
    for file in renamed_files:
        print(f"- {file}")
        
    while True:
        response = input("\nDo you want to delete these files? (yes/no): ").lower()
        if response in ['yes', 'y']:
            deleted_count = 0
            for file in renamed_files:
                try:
                    os.remove(file)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file}: {str(e)}")
            print(f"\nSuccessfully deleted {deleted_count}/{len(renamed_files)} files with naming issues")
            break
        elif response in ['no', 'n']:
            print("\nKeeping files with naming issues")
            break
        else:
            print("Please answer 'yes' or 'no'")

def main():
    """主函数，处理所有类别的图像"""
    try:
        # 创建输出目录
        for i in range(59):
            os.makedirs(os.path.join(config.aug_target_path, str(i)), exist_ok=True)
        
        # 处理每个类别的图片
        total_processed = 0
        total_skipped = 0
        renamed_files = []  # 存储所有命名有问题的文件路径
        
        for class_idx in range(59):
            class_dir = str(class_idx)
            raw_dir = os.path.join(config.aug_source_path, class_dir)
            save_dir = os.path.join(config.aug_target_path, class_dir)
            
            if not os.path.exists(raw_dir):
                print(f"Directory not found: {raw_dir}")
                continue
                
            # 获取该类别下所有图片
            image_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"\nProcessing class {class_idx} ({len(image_files)} images)")
            processed = 0
            skipped = 0
            
            # 使用多线程处理图像
            with ThreadPoolExecutor(max_workers=config.aug_num_workers) as executor:
                # 准备要处理的图像路径列表
                process_list = []
                for image_file in image_files:
                    image_path = os.path.join(raw_dir, image_file)
                    
                    if is_chinese(image_file):
                        renamed_files.append(image_path)  # 记录含有中文字符的文件路径
                        continue
                    
                    if check_augmented_images_exist(image_path, save_dir):
                        skipped += 1
                        continue
                        
                    process_list.append((image_path, save_dir))
                
                # 使用多线程处理图像
                results = list(tqdm(
                    executor.map(lambda p: process_image(*p), process_list),
                    total=len(process_list),
                    desc=f"Class {class_idx}"
                ))
                
                # 统计处理结果
                processed = sum(1 for r in results if r)
                    
            total_processed += processed
            total_skipped += skipped
            print(f"Class {class_idx}: Processed {processed} images, skipped {skipped} images.")
        
        print(f"\nTotal: Processed {total_processed} images, skipped {total_skipped} images.")
        
        # 处理命名有问题的文件
        if renamed_files:
            print(f"Found {len(renamed_files)} files with naming issues.")
            process_data_with_naming_issues(renamed_files)
            return True
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return False
    
if __name__ == "__main__":
    main()

 
