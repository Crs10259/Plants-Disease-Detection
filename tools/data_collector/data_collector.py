#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                            QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                            QLineEdit, QFileDialog, QProgressBar, QMessageBox,
                            QComboBox, QSpinBox, QCheckBox, QGroupBox, QFormLayout,
                            QTextEdit, QScrollArea, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap, QFont

# Import local modules - 使用相对导入
try:
    from .importers import ImporterTab
    from .scrapers import ScraperTab
    from .dataset_maker import DatasetMaker
except ImportError:
    # 如果处于直接运行模式
    from importers import ImporterTab
    from scrapers import ScraperTab
    from dataset_maker import DatasetMaker

# 确保日志目录存在
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DataCollector')

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("植物病害数据集制作工具")
        self.setMinimumSize(900, 700)
        
        # 设置应用图标
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建主布局和选项卡控件
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.tab_widget = QTabWidget()
        
        # 创建两个选项卡
        self.importer_tab = ImporterTab(self)
        self.scraper_tab = ScraperTab(self)
        
        # 添加选项卡
        self.tab_widget.addTab(self.importer_tab, "导入用户数据")
        self.tab_widget.addTab(self.scraper_tab, "在线数据收集")
        
        # 添加底部按钮
        buttons_layout = QHBoxLayout()
        self.make_dataset_btn = QPushButton("生成数据集")
        self.make_dataset_btn.setMinimumHeight(40)
        self.make_dataset_btn.setStyleSheet("font-weight: bold;")
        
        self.status_label = QLabel("就绪")
        
        # 将按钮添加到布局
        buttons_layout.addWidget(self.status_label, 1)
        buttons_layout.addWidget(self.make_dataset_btn)
        
        # 将选项卡和按钮添加到主布局
        main_layout.addWidget(self.tab_widget, 1)
        main_layout.addLayout(buttons_layout)
        
        # 设置主窗口小部件
        self.setCentralWidget(main_widget)
        
        # 创建数据集生成器
        self.dataset_maker = DatasetMaker()
    
    def connect_signals(self):
        """连接信号和槽"""
        self.make_dataset_btn.clicked.connect(self.generate_dataset)
    
    def generate_dataset(self):
        """生成数据集"""
        try:
            # 获取当前选项卡
            current_tab_index = self.tab_widget.currentIndex()
            
            if current_tab_index == 0:  # 导入用户数据选项卡
                # 获取导入选项卡的参数
                params = self.importer_tab.get_parameters()
                if not params["source_dir"]:
                    QMessageBox.warning(self, "参数错误", "请选择源数据目录")
                    return
                
                # 生成数据集
                self.status_label.setText("正在生成数据集...")
                success = self.dataset_maker.make_from_local(
                    source_dir=params["source_dir"],
                    output_dir=params["output_dir"],
                    split_ratio=params["split_ratio"],
                    class_mapping=params["class_mapping"],
                    generate_labels=params["generate_labels"],
                    size_filter=params["size_filter"]
                )
                
            else:  # 在线数据收集选项卡
                # 获取在线选项卡的参数
                params = self.scraper_tab.get_parameters()
                if not params["search_terms"]:
                    QMessageBox.warning(self, "参数错误", "请输入至少一个搜索词")
                    return
                
                # 显示进度条
                self.scraper_tab.show_progress(True)
                
                # 设置进度回调
                self.dataset_maker.set_progress_callback(self.scraper_tab.update_progress)
                
                # 生成数据集
                self.status_label.setText("正在收集在线数据并生成数据集...")
                success = self.dataset_maker.make_from_web(
                    search_terms=params["search_terms"],
                    output_dir=params["output_dir"],
                    images_per_class=params["images_per_class"],
                    sources=params["sources"],
                    generate_labels=params["generate_labels"],
                    quality_filter=params["quality_filter"],
                    deduplicate=params["deduplicate"],
                    size_filter=params["size_filter"]
                )
                
                # 隐藏进度条
                self.scraper_tab.show_progress(False)
            
            if success:
                self.status_label.setText("数据集生成完成")
                QMessageBox.information(self, "成功", "数据集已成功生成")
            else:
                self.status_label.setText("数据集生成失败")
                QMessageBox.critical(self, "错误", "数据集生成过程中发生错误")
                
        except Exception as e:
            logger.error(f"生成数据集时发生错误: {str(e)}")
            self.status_label.setText("错误")
            QMessageBox.critical(self, "错误", f"生成数据集时发生错误: {str(e)}")
            # 确保进度条被隐藏
            if current_tab_index == 1:
                self.scraper_tab.show_progress(False)

def main():
    """主函数"""
    try:
        # 创建应用
        app = QApplication(sys.argv)
        app.setStyle('Fusion')  
        
        # 创建主窗口
        window = MainWindow()
        window.show()
        
        # 运行应用
        sys.exit(app.exec())
    except Exception as e:
        print(f"启动应用时发生错误: {str(e)}")
        input("按回车键退出...")

if __name__ == "__main__":
    main() 