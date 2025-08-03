import os
import glob
import json
import shutil
import logging
import random
from typing import Dict, List, Any
import time
import io
import concurrent.futures
import math
from PIL import Image
import ssl
import urllib.request

logger = logging.getLogger('DatasetMaker')

class DatasetMaker:
    """数据集生成器类，支持从本地数据和在线数据生成数据集"""
    
    def __init__(self):
        """初始化数据集生成器"""
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.progress_callback = None
        
        # 设置ChromeDriver缓存和配置
        os.environ["WDM_LOG_LEVEL"] = "0"  # 禁用WDM日志
        os.environ["WDM_PRINT_FIRST_LINE"] = "False"  # 禁用WDM首行打印
        os.environ["WDM_LOCAL"] = "1"  # 强制使用本地缓存
        os.environ["WDM_SSL_VERIFY"] = "0"  # 禁用SSL验证
        
        # 创建ChromeDriver缓存目录
        self.chrome_driver_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromedriver_cache")
        os.makedirs(self.chrome_driver_cache, exist_ok=True)
        os.environ["WDM_CACHE_PATH"] = self.chrome_driver_cache
        
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def _update_progress(self, current: int, total: int, status: str = None):
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(current, total, status)
    
    def make_from_local(self, source_dir: str, output_dir: str, 
                       split_ratio: Dict[str, float] = None,
                       class_mapping: Dict[str, str] = None,
                       generate_labels: bool = True,
                       size_filter: Dict[str, Any] = None) -> bool:
        """从本地数据生成数据集
        
        参数:
            source_dir: 源数据目录
            output_dir: 输出目录
            split_ratio: 数据集分割比例，例如 {"train": 0.8, "val": 0.1, "test": 0.1}
            class_mapping: 类别映射，例如 {"folder_name": "class_name"}
            generate_labels: 是否生成标签文件
            size_filter: 图像尺寸过滤配置，例如 {"enabled": True, "min_width": 224, "min_height": 224}
            
        返回:
            bool: 是否成功
        """
        try:
            # 验证源目录
            if not os.path.isdir(source_dir):
                logger.error(f"源目录不存在: {source_dir}")
                return False
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取源目录中的所有子目录（类别）
            class_dirs = [d for d in os.listdir(source_dir) 
                         if os.path.isdir(os.path.join(source_dir, d)) and not d.startswith('.')]
            
            if not class_dirs:
                logger.error("源目录中未找到类别子目录")
                return False
            
            # 使用默认的分割比例
            if split_ratio is None:
                split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}
            
            # 验证分割比例
            total_ratio = sum(split_ratio.values())
            if not math.isclose(total_ratio, 1.0, rel_tol=1e-9):
                logger.error(f"分割比例之和必须为1.0，当前为: {total_ratio}")
                return False
            
            # 使用默认的类别映射
            if class_mapping is None:
                class_mapping = {d: d.replace("_", " ").title() for d in class_dirs}
            
            # 创建数据集子目录
            for split in ["train", "val", "test"]:
                split_dir = os.path.join(output_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                for class_dir in class_dirs:
                    os.makedirs(os.path.join(split_dir, class_dir), exist_ok=True)
            
            # 处理每个类别
            label_mapping = {}
            for i, class_dir in enumerate(sorted(class_dirs)):
                logger.info(f"处理类别: {class_dir}")
                
                # 获取该类别的所有图像文件
                class_path = os.path.join(source_dir, class_dir)
                image_files = []
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    image_files.extend(glob.glob(os.path.join(class_path, f"*{ext}")))
                    image_files.extend(glob.glob(os.path.join(class_path, f"*{ext.upper()}")))
                
                if not image_files:
                    logger.warning(f"类别 {class_dir} 中未找到图像文件")
                    continue
                
                # 过滤图像
                if size_filter and size_filter.get("enabled", False):
                    image_files = self._filter_by_size(image_files, 
                                                     size_filter.get("min_width", 224),
                                                     size_filter.get("min_height", 224))
                
                # 打乱文件顺序
                random.shuffle(image_files)
                
                # 根据比例分割数据
                n_total = len(image_files)
                n_train = int(n_total * split_ratio["train"])
                n_val = int(n_total * split_ratio["val"])
                n_test = n_total - n_train - n_val
                
                # 分配文件
                train_files = image_files[:n_train]
                val_files = image_files[n_train:n_train+n_val]
                test_files = image_files[n_train+n_val:]
                
                # 复制文件到对应目录
                for files, split in [(train_files, "train"), 
                                   (val_files, "val"), 
                                   (test_files, "test")]:
                    for src_file in files:
                        dst_file = os.path.join(output_dir, split, class_dir, 
                                              os.path.basename(src_file))
                        shutil.copy2(src_file, dst_file)
                
                # 添加到标签映射
                label_mapping[str(i)] = {
                    "id": i,
                    "name": class_mapping.get(class_dir, class_dir)
                }
                
                logger.info(f"类别 {class_dir} 处理完成: "
                          f"训练集 {len(train_files)}, "
                          f"验证集 {len(val_files)}, "
                          f"测试集 {len(test_files)}")
            
            # 生成标签文件
            if generate_labels:
                labels_path = os.path.join(output_dir, "labels.json")
                with open(labels_path, 'w', encoding='utf-8') as f:
                    json.dump(label_mapping, f, ensure_ascii=False, indent=2)
                logger.info(f"标签文件已保存到: {labels_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"生成数据集时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def make_from_web(self, search_terms: List[str], output_dir: str,
                     images_per_class: int = 100, sources: List[str] = None,
                     generate_labels: bool = True, quality_filter: bool = True,
                     deduplicate: bool = True, size_filter: Dict[str, Any] = None) -> bool:
        """从网络收集数据并生成数据集
        
        参数:
            search_terms: 搜索词列表
            output_dir: 输出目录
            images_per_class: 每个类别收集的图像数量
            sources: 数据源列表，例如 ["google", "bing", "baidu"]
            generate_labels: 是否生成标签文件
            quality_filter: 是否过滤低质量图像
            deduplicate: 是否去除重复图像
            size_filter: 图像尺寸过滤配置，例如 {"enabled": True, "min_width": 224, "min_height": 224}
            
        返回:
            bool: 是否成功
        """
        try:
            # 创建临时目录
            temp_dir = os.path.join(output_dir, "_temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 使用默认数据源
            if not sources:
                sources = ["google", "baidu"]
            
            # 创建输出目录结构
            for split in ["train", "val", "test"]:
                split_dir = os.path.join(output_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                for i in range(len(search_terms)):
                    os.makedirs(os.path.join(split_dir, str(i)), exist_ok=True)
            
            # 计算总任务数（每个搜索词的目标图片数）
            total_tasks = len(search_terms) * images_per_class
            current_progress = 0
            
            # 准备标签映射
            label_mapping = {
                str(i): {
                    "id": i,
                    "name": term.strip()
                }
                for i, term in enumerate(search_terms)
            }
            
            # 使用不同的数据源收集图像
            for i, search_term in enumerate(search_terms):
                self._update_progress(current_progress, total_tasks, 
                                   f"正在收集 '{search_term}' 的图像...")
                
                class_temp_dir = os.path.join(temp_dir, str(i))
                os.makedirs(class_temp_dir, exist_ok=True)
                
                # 从各个源收集图像
                collected_images = []
                
                # 计算每个源的图像分配
                images_per_source = images_per_class // len(sources)
                remainder = images_per_class % len(sources)
                
                # 收集图像
                remaining_count = images_per_class
                for source in sources:
                    # 分配每个源收集的图像数量
                    source_count = images_per_source
                    if remainder > 0:
                        source_count += 1
                        remainder -= 1
                    
                    self._update_progress(current_progress, total_tasks,
                                       f"从{source}收集 '{search_term}' 的图像...")
                    
                    if source == "google":
                        images = self._collect_from_google(search_term, source_count, class_temp_dir)
                    elif source == "bing":
                        images = self._collect_from_bing(search_term, source_count, class_temp_dir)
                    elif source == "baidu":
                        images = self._collect_from_baidu(search_term, source_count, class_temp_dir)
                    elif source == "flickr":
                        images = self._collect_from_flickr(search_term, source_count, class_temp_dir)
                    elif source == "custom":
                        images = self._collect_from_custom(search_term, source_count, class_temp_dir)
                    else:
                        images = []
                        
                    collected_images.extend(images)
                    current_progress += len(images)
                    self._update_progress(min(current_progress, total_tasks), total_tasks,
                                       f"已从{source}收集到 {len(images)} 张图像")
                    
                    remaining_count -= len(images)
                    if remaining_count <= 0:
                        break
                
                # 过滤图像
                if quality_filter:
                    self._update_progress(current_progress, total_tasks,
                                      f"正在过滤 '{search_term}' 的低质量图像...")
                    collected_images = self._filter_low_quality_images(collected_images)
                
                if size_filter and size_filter.get("enabled", False):
                    self._update_progress(current_progress, total_tasks,
                                      f"正在过滤 '{search_term}' 的图像尺寸...")
                    collected_images = self._filter_by_size(collected_images, 
                                                          size_filter.get("min_width", 224),
                                                          size_filter.get("min_height", 224))
                
                if deduplicate:
                    self._update_progress(current_progress, total_tasks,
                                      f"正在去除 '{search_term}' 的重复图像...")
                    collected_images = self._remove_duplicate_images(collected_images)
                
                # 分割数据集
                random.shuffle(collected_images)
                n_total = len(collected_images)
                n_train = int(n_total * 0.8)
                n_val = int(n_total * 0.1)
                n_test = n_total - n_train - n_val
                
                # 分配文件
                train_files = collected_images[:n_train]
                val_files = collected_images[n_train:n_train+n_val]
                test_files = collected_images[n_train+n_val:]
                
                # 复制文件
                self._update_progress(current_progress, total_tasks,
                                  f"正在保存 '{search_term}' 的图像...")
                self._copy_files_to_split(train_files, os.path.join(output_dir, "train", str(i)))
                self._copy_files_to_split(val_files, os.path.join(output_dir, "val", str(i)))
                self._copy_files_to_split(test_files, os.path.join(output_dir, "test", str(i)))
            
            # 生成标签文件
            if generate_labels:
                self._update_progress(total_tasks - 1, total_tasks, "正在生成标签文件...")
                labels_path = os.path.join(output_dir, "labels.json")
                with open(labels_path, 'w', encoding='utf-8') as f:
                    json.dump(label_mapping, f, ensure_ascii=False, indent=2)
                logger.info(f"标签文件已保存到: {labels_path}")
            
            # 清理临时目录
            if os.path.exists(temp_dir):
                self._update_progress(total_tasks - 1, total_tasks, "正在清理临时文件...")
                shutil.rmtree(temp_dir)
            
            # 最终更新
            self._update_progress(total_tasks, total_tasks, "数据集生成完成")
            return True
            
        except Exception as e:
            logger.error(f"从网络收集数据时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _copy_files_to_split(self, files: List[str], target_dir: str) -> None:
        """将文件复制到相应的数据集分割目录
        
        参数:
            files: 文件路径列表
            target_dir: 目标目录
        """
        os.makedirs(target_dir, exist_ok=True)
        
        for i, file_path in enumerate(files):
            # 确保文件名唯一
            ext = os.path.splitext(file_path)[1]
            target_path = os.path.join(target_dir, f"{i:05d}{ext}")
            try:
                shutil.copy2(file_path, target_path)
            except Exception as e:
                logger.warning(f"复制文件时出错: {file_path} -> {target_path}: {str(e)}")
    
    def _collect_from_google(self, search_term: str, count: int, output_dir: str) -> List[str]:
        """从Google收集图像
        
        参数:
            search_term: 搜索词
            count: 要收集的图像数量
            output_dir: 输出目录
            
        返回:
            收集到的图像文件路径列表
        """
        logger.info(f"从Google收集图像: {search_term}")
        
        # 使用selenium自动搜索和下载图片
        collected_files = []
        
        try:
            # 为避免反爬措施，使用selenium模拟浏览器
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            import time
            import urllib.request
            import urllib.parse
            from PIL import Image
            import os
            
            # 设置Chrome选项
            options = Options()
            options.add_argument("--headless")  # 无头模式
            options.add_argument("--disable-gpu")  # 禁用GPU
            options.add_argument("--window-size=1920,1080")  # 设置窗口大小
            options.add_argument("--no-sandbox")  # 禁用沙盒
            options.add_argument("--disable-dev-shm-usage")  # 禁用共享内存
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            # 使用ChromeDriverManager安装驱动
            driver_path = ChromeDriverManager().install()
            
            # 初始化Chrome驱动
            driver = webdriver.Chrome(service=Service(driver_path), options=options)
            
            # 构建Google图片搜索URL
            encoded_query = urllib.parse.quote(search_term)
            url = f"https://www.google.com/search?q={encoded_query}&tbm=isch"
            
            # 访问Google图片搜索
            driver.get(url)
            
            # 等待页面加载
            time.sleep(3)
            
            # 滚动页面以加载更多图片
            for _ in range(min(count // 10 + 1, 5)):  # 限制滚动次数，避免过长时间
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            # 查找所有图片元素
            image_elements = driver.find_elements(By.CSS_SELECTOR, "img.rg_i")
            
            # 限制图片数量
            image_elements = image_elements[:min(count, len(image_elements))]
            
            for i, img_element in enumerate(image_elements):
                try:
                    # 点击图片以获取高分辨率版本
                    img_element.click()
                    time.sleep(1)
                    
                    # 获取高分辨率图片URL
                    # Google图片搜索结果中，高分辨率图片通常在特定元素中
                    big_img = driver.find_elements(By.CSS_SELECTOR, "img.n3VNCb")
                    if big_img:
                        img_url = big_img[0].get_attribute("src")
                    else:
                        # 如果没有找到高分辨率图片，使用缩略图
                        img_url = img_element.get_attribute("src")
                    
                    # 跳过数据URI（base64编码的图片）
                    if img_url.startswith("data:"):
                        continue
                        
                    # 创建文件名
                    file_name = f"{i:03d}.jpg"
                    file_path = os.path.join(output_dir, file_name)
                    
                    # 下载图片
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                    req = urllib.request.Request(img_url, headers=headers)
                    
                    with urllib.request.urlopen(req, timeout=10) as response:
                        img_data = response.read()
                        
                        # 验证图片数据是否有效
                        try:
                            img = Image.open(io.BytesIO(img_data))
                            
                            # 保存图片
                            with open(file_path, 'wb') as f:
                                f.write(img_data)
                                
                            # 添加到收集列表
                            collected_files.append(file_path)
                            
                            # 如果收集到足够数量的图片，提前结束
                            if len(collected_files) >= count:
                                break
                        except Exception as e:
                            logger.warning(f"无效的图片数据: {str(e)}")
                    
                    # 避免过快请求
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"下载图片时出错: {str(e)}")
            
            # 关闭浏览器
            driver.quit()
                
        except Exception as e:
            logger.error(f"从Google收集图像时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info(f"从Google收集到 {len(collected_files)} 张图片")
        return collected_files
    
    def _collect_from_bing(self, search_term: str, count: int, output_dir: str) -> List[str]:
        """从Bing收集图像
        
        参数:
            search_term: 搜索词
            count: 要收集的图像数量
            output_dir: 输出目录
            
        返回:
            收集到的图像文件路径列表
        """
        logger.info(f"从Bing收集图像: {search_term}")
        
        # 使用Bing图片搜索API获取图像
        collected_files = []
        
        try:
            # 使用Bing图片搜索
            import urllib.request
            import urllib.parse
            import json
            from PIL import Image
            
            # 构建Bing图片搜索URL (不需要API密钥的方法)
            encoded_query = urllib.parse.quote(search_term)
            url = f"https://www.bing.com/images/search?q={encoded_query}&form=HDRSC2&first=1"
            
            # 配置请求头
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            # 使用selenium获取页面内容
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            import time
            import re
            import os
            
            # 设置Chrome选项
            options = Options()
            options.add_argument("--headless")  # 无头模式
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument(f"user-agent={headers['User-Agent']}")
            
            # 使用ChromeDriverManager安装驱动
            driver_path = ChromeDriverManager().install()
            
            # 初始化Chrome驱动
            driver = webdriver.Chrome(service=Service(driver_path), options=options)
            
            # 访问Bing图片搜索
            driver.get(url)
            
            # 等待页面加载
            time.sleep(3)
            
            # 滚动页面以加载更多图片
            for _ in range(min(count // 20 + 1, 5)):  # 限制滚动次数
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            # 查找所有图片元素
            image_elements = driver.find_elements(By.CSS_SELECTOR, ".mimg")
            
            # 如果找不到图片，尝试其他选择器
            if not image_elements:
                image_elements = driver.find_elements(By.CSS_SELECTOR, "img.imgpt")
            
            if not image_elements:
                image_elements = driver.find_elements(By.CSS_SELECTOR, "img")
                
            # 提取图片URL
            img_urls = []
            for img in image_elements:
                src = img.get_attribute("src")
                if src and not src.startswith("data:"):
                    img_urls.append(src)
            
            # 限制URL数量
            img_urls = img_urls[:min(count, len(img_urls))]
            
            # 关闭浏览器
            driver.quit()
            
            # 下载图片
            for i, img_url in enumerate(img_urls):
                try:
                    # 创建文件名
                    file_name = f"bing_{i:03d}.jpg"
                    file_path = os.path.join(output_dir, file_name)
                    
                    # 下载图片
                    req = urllib.request.Request(img_url, headers=headers)
                    
                    with urllib.request.urlopen(req, timeout=10) as response:
                        img_data = response.read()
                        
                        # 验证图片数据是否有效
                        try:
                            img = Image.open(io.BytesIO(img_data))
                            
                            # 保存图片
                            with open(file_path, 'wb') as f:
                                f.write(img_data)
                                
                            # 添加到收集列表
                            collected_files.append(file_path)
                            
                            # 如果收集到足够数量的图片，提前结束
                            if len(collected_files) >= count:
                                break
                        except Exception as e:
                            logger.warning(f"无效的图片数据: {str(e)}")
                    
                    # 避免过快请求
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"下载图片时出错: {str(e)}")
                
        except Exception as e:
            logger.error(f"从Bing收集图像时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info(f"从Bing收集到 {len(collected_files)} 张图片")
        return collected_files
    
    def _download_image(self, img_url: str, file_path: str, headers: dict, max_retries: int = 3) -> bool:
        """下载单个图片，带重试机制
        
        参数:
            img_url: 图片URL
            file_path: 保存路径
            headers: 请求头
            max_retries: 最大重试次数
            
        返回:
            bool: 是否成功
        """
        for retry in range(max_retries):
            try:
                # 创建请求时禁用SSL验证
                context = ssl._create_unverified_context()
                req = urllib.request.Request(img_url, headers=headers)
                
                with urllib.request.urlopen(req, context=context, timeout=10) as response:
                    img_data = response.read()
                    
                    # 验证图片数据是否有效
                    try:
                        img = Image.open(io.BytesIO(img_data))
                        
                        # 检查图片尺寸是否合理
                        if img.width < 50 or img.height < 50:
                            logger.warning(f"图片太小，跳过: {img.width}x{img.height}")
                            return False
                        
                        # 保存图片
                        with open(file_path, 'wb') as f:
                            f.write(img_data)
                        return True
                        
                    except Exception as e:
                        logger.warning(f"无效的图片数据: {str(e)}")
                        return False
                        
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    # 404错误不需要重试
                    logger.warning(f"图片不存在 (404): {img_url}")
                    return False
                elif retry < max_retries - 1:
                    logger.warning(f"HTTP错误 {e.code}，正在重试 ({retry + 1}/{max_retries})")
                    time.sleep(1)  # 等待一秒后重试
                else:
                    logger.warning(f"下载图片失败 (HTTP {e.code}): {img_url}")
                    return False
                    
            except (urllib.error.URLError, ssl.SSLError) as e:
                if retry < max_retries - 1:
                    logger.warning(f"连接错误，正在重试 ({retry + 1}/{max_retries}): {str(e)}")
                    time.sleep(1)
                else:
                    logger.warning(f"下载图片失败: {str(e)}")
                    return False
                    
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"未知错误，正在重试 ({retry + 1}/{max_retries}): {str(e)}")
                    time.sleep(1)
                else:
                    logger.warning(f"下载图片失败: {str(e)}")
                    return False
        
        return False

    def _collect_from_baidu(self, search_term: str, count: int, output_dir: str) -> List[str]:
        """从百度收集图像
        
        参数:
            search_term: 搜索词
            count: 要收集的图像数量
            output_dir: 输出目录
            
        返回:
            收集到的图像文件路径列表
        """
        logger.info(f"从百度收集图像: {search_term}")
        
        # 使用百度图片搜索收集图像
        collected_files = []
        
        try:
            # 使用selenium自动搜索和下载图片
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            import time
            import urllib.request
            import urllib.parse
            from PIL import Image
            import re
            import os
            
            # 设置Chrome选项
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            # 使用ChromeDriverManager安装驱动
            driver_path = ChromeDriverManager().install()
            
            # 初始化Chrome驱动
            driver = webdriver.Chrome(service=Service(driver_path), options=options)
            
            # 构建百度图片搜索URL
            encoded_query = urllib.parse.quote(search_term)
            url = f"https://image.baidu.com/search/index?tn=baiduimage&word={encoded_query}"
            
            # 访问百度图片搜索
            driver.get(url)
            
            # 等待页面加载
            time.sleep(3)
            
            # 滚动页面以加载更多图片
            for _ in range(min(count // 30 + 1, 5)):  # 百度每页大约30张图片
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            # 查找所有图片元素
            # 百度图片的图片元素通常在特定的容器中
            image_elements = driver.find_elements(By.CSS_SELECTOR, ".main_img")
            
            # 如果没有找到图片，尝试其他选择器
            if not image_elements:
                image_elements = driver.find_elements(By.CSS_SELECTOR, "img.imgitem")
            
            if not image_elements:
                # 使用更通用的选择器
                image_elements = driver.find_elements(By.CSS_SELECTOR, "img[data-imgurl]")
            
            if not image_elements:
                # 最后尝试获取所有图片，但过滤掉太小的图片
                all_images = driver.find_elements(By.TAG_NAME, "img")
                image_elements = [img for img in all_images if 
                                 img.get_attribute("src") and 
                                 not img.get_attribute("src").startswith("data:") and
                                 (img.size['width'] > 100 or img.size['height'] > 100)]
            
            logger.info(f"在百度找到 {len(image_elements)} 个图片元素")
            
            # 限制图片元素数量
            image_elements = image_elements[:min(count * 2, len(image_elements))]  # 获取更多备用
            
            # 获取所有图片URL
            img_urls = []
            for img in image_elements:
                # 尝试多种方式获取图片URL
                img_url = img.get_attribute("data-imgurl") or img.get_attribute("data-src") or img.get_attribute("src")
                
                if img_url and not img_url.startswith("data:"):
                    # 处理可能的相对URL
                    if img_url.startswith("//"):
                        img_url = "https:" + img_url
                    
                    # 检查是否有更高质量的版本
                    objurl = img.get_attribute("data-objurl")
                    if objurl:
                        img_url = objurl
                    
                    img_urls.append(img_url)
            
            logger.info(f"从百度提取了 {len(img_urls)} 个图片URL")
            
            # 关闭浏览器
            driver.quit()
            
            # 限制URL数量并打乱顺序
            import random
            random.shuffle(img_urls)
            img_urls = img_urls[:min(count, len(img_urls))]
            
            # 下载图片
            for i, img_url in enumerate(img_urls):
                try:
                    # 创建文件名
                    file_name = f"baidu_{i:03d}.jpg"
                    file_path = os.path.join(output_dir, file_name)
                    
                    # 设置请求头
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                        'Referer': 'https://image.baidu.com/',
                        'Connection': 'keep-alive'
                    }
                    
                    # 使用改进的下载函数
                    if self._download_image(img_url, file_path, headers):
                        collected_files.append(file_path)
                        
                        # 如果收集到足够数量的图片，提前结束
                        if len(collected_files) >= count:
                            break
                
                    # 避免过快请求
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"处理图片时出错: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"从百度收集图像时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info(f"从百度收集到 {len(collected_files)} 张图片")
        return collected_files
    
    def _collect_from_flickr(self, search_term: str, count: int, output_dir: str) -> List[str]:
        """从Flickr收集图像
        
        参数:
            search_term: 搜索词
            count: 要收集的图像数量
            output_dir: 输出目录
            
        返回:
            收集到的图像文件路径列表
        """
        logger.info(f"从Flickr收集图像: {search_term}")
        
        # 使用Flickr API获取图像
        collected_files = []
        
        try:
            import flickrapi
            import urllib.request
            from PIL import Image
            import time
            import os
            from dotenv import load_dotenv
            
            # 加载环境变量（若存在）
            load_dotenv()
            
            # 获取API密钥
            # 用户需要从 https://www.flickr.com/services/apps/create/ 获取
            api_key = os.environ.get('FLICKR_API_KEY')
            api_secret = os.environ.get('FLICKR_API_SECRET')
            
            if not api_key or not api_secret:
                # 如果没有设置API密钥，提示用户
                logger.warning("未设置Flickr API密钥，请在环境变量或.env文件中设置FLICKR_API_KEY和FLICKR_API_SECRET")
                logger.warning("访问 https://www.flickr.com/services/apps/create/ 创建应用并获取密钥")
                # 创建dotenv文件模板
                env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
                if not os.path.exists(env_file):
                    with open(env_file, 'w') as f:
                        f.write("# Flickr API 设置\n")
                        f.write("FLICKR_API_KEY=your_api_key_here\n")
                        f.write("FLICKR_API_SECRET=your_api_secret_here\n")
                # 尝试使用web scraping方法获取图像
                return self._collect_from_flickr_scrape(search_term, count, output_dir)
            
            # 初始化Flickr API
            flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
            
            # 搜索照片
            search_results = flickr.photos.search(
                text=search_term,           # 搜索文本
                per_page=min(count, 100),   # 每页的图片数量
                media='photos',             # 仅搜索照片
                sort='relevance',           # 按相关性排序
                safe_search=1,              # 安全搜索
                content_type=1,             # 照片内容
                extras='url_c,url_l,url_o'  # 额外获取的图片信息，包括URL
            )
            
            # 获取照片列表
            photos = search_results['photos']['photo']
            
            # 下载照片
            for i, photo in enumerate(photos):
                try:
                    # 选择最佳可用的URL（优先选择大图）
                    if 'url_o' in photo and photo['url_o']:
                        img_url = photo['url_o']
                    elif 'url_l' in photo and photo['url_l']:
                        img_url = photo['url_l']
                    elif 'url_c' in photo and photo['url_c']:
                        img_url = photo['url_c']
                    else:
                        # 如果没有可用的URL，构建一个（可能不准确）
                        img_url = f"https://live.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}.jpg"
                    
                    # 创建文件名
                    file_name = f"flickr_{i:03d}.jpg"
                    file_path = os.path.join(output_dir, file_name)
                    
                    # 下载图片
                    urllib.request.urlretrieve(img_url, file_path)
                    
                    # 验证图片是否有效
                    try:
                        img = Image.open(file_path)
                        # 添加到收集列表
                        collected_files.append(file_path)
                    except Exception as e:
                        logger.warning(f"无效的图片数据: {str(e)}")
                        # 移除无效图片
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    
                    # 避免频繁请求
                    time.sleep(0.2)
                    
                    # 如果收集到足够数量的图片，提前结束
                    if len(collected_files) >= count:
                        break
                        
                except Exception as e:
                    logger.warning(f"下载Flickr图片时出错: {str(e)}")
                
        except Exception as e:
            logger.error(f"从Flickr收集图像时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info(f"从Flickr收集到 {len(collected_files)} 张图片")
        return collected_files
        
    def _collect_from_flickr_scrape(self, search_term: str, count: int, output_dir: str) -> List[str]:
        """从Flickr网站爬取图像（备用方法，不使用API）
        
        参数:
            search_term: 搜索词
            count: 要收集的图像数量
            output_dir: 输出目录
            
        返回:
            收集到的图像文件路径列表
        """
        collected_files = []
        
        try:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            import urllib.request
            import urllib.parse
            import time
            from PIL import Image
            import os
            
            # 设置Chrome选项
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            # 使用ChromeDriverManager安装驱动
            driver_path = ChromeDriverManager().install()
            
            # 初始化Chrome驱动
            driver = webdriver.Chrome(service=Service(driver_path), options=options)
            
            # 构建Flickr搜索URL
            encoded_query = urllib.parse.quote(search_term)
            url = f"https://www.flickr.com/search/?text={encoded_query}"
            
            # 访问Flickr搜索
            driver.get(url)
            
            # 等待页面加载
            time.sleep(3)
            
            # 滚动页面以加载更多图片
            for _ in range(min(count // 20 + 1, 5)):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            # 查找所有图片元素
            # Flickr图片通常在特定的容器中
            # 这可能需要根据Flickr页面的变化进行调整
            image_elements = driver.find_elements(By.CSS_SELECTOR, ".photo-list-photo-view")
            
            # 提取图片URL
            img_urls = []
            for img_element in image_elements:
                try:
                    # 获取背景图片URL
                    style = img_element.get_attribute("style")
                    # 提取URL
                    import re
                    url_match = re.search(r'url\(["\']?(.*?)["\']?\)', style)
                    if url_match:
                        img_url = url_match.group(1)
                        # 处理相对URL
                        if img_url.startswith('//'):
                            img_url = 'https:' + img_url
                        img_urls.append(img_url)
                except Exception as e:
                    logger.warning(f"提取Flickr图片URL时出错: {str(e)}")
            
            # 关闭浏览器
            driver.quit()
            
            # 限制URL数量
            img_urls = img_urls[:min(count, len(img_urls))]
            
            # 下载图片
            for i, img_url in enumerate(img_urls):
                try:
                    # 创建文件名
                    file_name = f"flickr_scrape_{i:03d}.jpg"
                    file_path = os.path.join(output_dir, file_name)
                    
                    # 下载图片
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                    req = urllib.request.Request(img_url, headers=headers)
                    
                    with urllib.request.urlopen(req, timeout=10) as response:
                        img_data = response.read()
                        
                        # 验证图片数据是否有效
                        try:
                            img = Image.open(io.BytesIO(img_data))
                            
                            # 保存图片
                            with open(file_path, 'wb') as f:
                                f.write(img_data)
                                
                            # 添加到收集列表
                            collected_files.append(file_path)
                            
                            # 如果收集到足够数量的图片，提前结束
                            if len(collected_files) >= count:
                                break
                        except Exception as e:
                            logger.warning(f"无效的图片数据: {str(e)}")
                    
                    # 避免过快请求
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"下载图片时出错: {str(e)}")
                    
        except Exception as e:
            logger.error(f"从Flickr网站爬取图像时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
        return collected_files
    
    def _collect_from_custom(self, search_term: str, count: int, output_dir: str) -> List[str]:
        """从自定义源收集图像
        
        参数:
            search_term: 搜索词
            count: 要收集的图像数量
            output_dir: 输出目录
            
        返回:
            收集到的图像文件路径列表
        """
        logger.info(f"从自定义源收集图像: {search_term}")
        
        # 这里为自定义源实现，例如爬取特定网站或使用特定API
        # 这里为了示例，使用模拟实现
        collected_files = []
        
        try:
            # 模拟爬取过程，不实际下载
            time.sleep(1)  # 减慢速度
            
            # 返回模拟的文件列表
            for i in range(min(count, 20)):  # 限制数量以避免长时间等待
                file_path = os.path.join(output_dir, f"custom_{i:03d}.jpg")
                # 创建空文件作为占位符
                with open(file_path, 'w') as f:
                    f.write('')
                collected_files.append(file_path)
                
        except Exception as e:
            logger.error(f"从自定义源收集图像时出错: {str(e)}")
        
        return collected_files
    
    def _filter_low_quality_images(self, image_files: List[str]) -> List[str]:
        """过滤低质量图像
        
        参数:
            image_files: 图像文件路径列表
            
        返回:
            过滤后的图像文件路径列表
        """
        logger.info(f"过滤低质量图像: {len(image_files)} 个文件")
        
        # 在实际实现中，可以检查图像分辨率、文件大小、清晰度等
        # 这里为了示例，不进行实际过滤
        return image_files
    
    def _remove_duplicate_images(self, image_files: List[str]) -> List[str]:
        """去除重复图像
        
        参数:
            image_files: 图像文件路径列表
            
        返回:
            去重后的图像文件路径列表
        """
        logger.info(f"去除重复图像: {len(image_files)} 个文件")
        
        # 在实际实现中，可以计算图像哈希或特征向量，比较相似度
        # 这里为了示例，不进行实际去重
        return image_files
    
    def _filter_by_size(self, image_files: List[str], min_width: int, min_height: int) -> List[str]:
        """根据尺寸过滤图像
        
        参数:
            image_files: 图像文件路径列表
            min_width: 最小宽度
            min_height: 最小高度
            
        返回:
            过滤后的图像文件路径列表
        """
        logger.info(f"根据尺寸过滤图像: 最小尺寸 {min_width}x{min_height}")
        
        filtered_files = []
        for file_path in image_files:
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    if width >= min_width and height >= min_height:
                        filtered_files.append(file_path)
                    else:
                        logger.debug(f"图像尺寸太小，已过滤: {file_path} ({width}x{height})")
            except Exception as e:
                logger.warning(f"检查图像尺寸时出错: {file_path} - {str(e)}")
        
        logger.info(f"尺寸过滤后保留 {len(filtered_files)}/{len(image_files)} 个文件")
        return filtered_files 