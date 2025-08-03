#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                            QLineEdit, QFileDialog, QProgressBar, QMessageBox,
                            QComboBox, QSpinBox, QCheckBox, QGroupBox, QFormLayout,
                            QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

logger = logging.getLogger('Scraper')

class ScraperTab(QWidget):
    """在线数据收集选项卡"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """设置用户界面"""
        main_layout = QVBoxLayout(self)
        
        # 创建搜索词组
        search_group = QGroupBox("定义搜索词")
        search_layout = QVBoxLayout()
        
        # 说明
        search_info = QLabel("为每个类别添加搜索词，每行一个搜索词。例如：'玉米叶斑病', '水稻稻瘟病'")
        
        # 搜索词文本编辑器
        self.search_terms_edit = QTextEdit()
        self.search_terms_edit.setPlaceholderText("输入搜索词，每行一个\n例如:\n玉米叶斑病\n水稻稻瘟病\n黄瓜花叶病")
        
        # 添加到布局
        search_layout.addWidget(search_info)
        search_layout.addWidget(self.search_terms_edit)
        search_group.setLayout(search_layout)
        
        # 创建数据源组
        sources_group = QGroupBox("数据源配置")
        sources_layout = QVBoxLayout()
        
        # 数据源标签
        sources_label = QLabel("选择要从中收集数据的源：")
        
        # 数据源复选框
        sources_checkbox_layout = QHBoxLayout()
        self.google_cb = QCheckBox("Google图片")
        self.google_cb.setChecked(True)
        self.bing_cb = QCheckBox("Bing图片")
        self.bing_cb.setChecked(True)
        self.baidu_cb = QCheckBox("百度图片")
        self.baidu_cb.setChecked(True)
        self.flickr_cb = QCheckBox("Flickr")
        self.flickr_cb.setChecked(False)
        self.custom_cb = QCheckBox("自定义API")
        self.custom_cb.setChecked(False)
        
        sources_checkbox_layout.addWidget(self.google_cb)
        sources_checkbox_layout.addWidget(self.bing_cb)
        sources_checkbox_layout.addWidget(self.baidu_cb)
        sources_checkbox_layout.addWidget(self.flickr_cb)
        sources_checkbox_layout.addWidget(self.custom_cb)
        
        # 每个类别的图片数量
        images_per_class_layout = QHBoxLayout()
        images_per_class_label = QLabel("每个类别的图像数量:")
        self.images_per_class_spin = QSpinBox()
        self.images_per_class_spin.setMinimum(10)
        self.images_per_class_spin.setMaximum(1000)
        self.images_per_class_spin.setValue(100)
        self.images_per_class_spin.setSingleStep(10)
        
        images_per_class_layout.addWidget(images_per_class_label)
        images_per_class_layout.addWidget(self.images_per_class_spin)
        images_per_class_layout.addStretch()
        
        # 添加到布局
        sources_layout.addWidget(sources_label)
        sources_layout.addLayout(sources_checkbox_layout)
        sources_layout.addLayout(images_per_class_layout)
        sources_group.setLayout(sources_layout)
        
        # 创建输出配置组
        output_group = QGroupBox("输出配置")
        output_layout = QFormLayout()
        
        # 输出目录选择控件
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("选择数据集输出目录")
        self.browse_output_btn = QPushButton("浏览...")
        output_dir_layout.addWidget(self.output_dir_edit, 1)
        output_dir_layout.addWidget(self.browse_output_btn)
        
        # 添加标签生成选项
        self.generate_labels_cb = QCheckBox("生成标签文件")
        self.generate_labels_cb.setChecked(True)
        
        # 将控件添加到布局
        output_layout.addRow("输出目录:", output_dir_layout)
        output_layout.addRow("", self.generate_labels_cb)
        output_group.setLayout(output_layout)
        
        # 高级选项组
        advanced_group = QGroupBox("高级选项")
        advanced_layout = QFormLayout()
        
        # 自动分类选项
        self.auto_classify_cb = QCheckBox("使用AI对收集的图像进行自动分类")
        self.auto_classify_cb.setChecked(True)
        
        # 图像质量过滤
        self.quality_filter_cb = QCheckBox("过滤低质量图像")
        self.quality_filter_cb.setChecked(True)
        
        # 去重选项
        self.deduplicate_cb = QCheckBox("去除重复图像")
        self.deduplicate_cb.setChecked(True)

        # 图像尺寸过滤选项
        size_filter_layout = QHBoxLayout()
        self.size_filter_cb = QCheckBox("过滤图像尺寸")
        self.size_filter_cb.setChecked(False)
        
        # 最小尺寸设置
        min_size_layout = QHBoxLayout()
        min_size_layout.addWidget(QLabel("最小尺寸:"))
        self.min_width_spin = QSpinBox()
        self.min_width_spin.setRange(32, 4096)
        self.min_width_spin.setValue(224)
        self.min_width_spin.setSuffix(" px")
        self.min_width_spin.setEnabled(False)
        min_size_layout.addWidget(self.min_width_spin)
        min_size_layout.addWidget(QLabel("×"))
        self.min_height_spin = QSpinBox()
        self.min_height_spin.setRange(32, 4096)
        self.min_height_spin.setValue(224)
        self.min_height_spin.setSuffix(" px")
        self.min_height_spin.setEnabled(False)
        min_size_layout.addWidget(self.min_height_spin)
        
        size_filter_layout.addWidget(self.size_filter_cb)
        size_filter_layout.addLayout(min_size_layout)
        size_filter_layout.addStretch()
        
        # 将控件添加到布局
        advanced_layout.addRow("", self.auto_classify_cb)
        advanced_layout.addRow("", self.quality_filter_cb)
        advanced_layout.addRow("", self.deduplicate_cb)
        advanced_layout.addRow("", size_filter_layout)
        advanced_group.setLayout(advanced_layout)
        
        # 状态栏
        status_layout = QHBoxLayout()
        self.status_label = QLabel("就绪")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% (%v/%m)")
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.progress_bar)
        
        # 添加所有组件到主布局
        main_layout.addWidget(search_group, 2)
        main_layout.addWidget(sources_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(advanced_group)
        main_layout.addLayout(status_layout)
        
    def connect_signals(self):
        """连接信号和槽"""
        self.browse_output_btn.clicked.connect(self.browse_output_directory)
        # 连接尺寸过滤复选框
        self.size_filter_cb.toggled.connect(self.toggle_size_filter)
        
    def browse_output_directory(self):
        """浏览输出目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择数据集输出目录")
        if directory:
            self.output_dir_edit.setText(directory)
    
    def toggle_size_filter(self, enabled):
        """切换尺寸过滤选项的启用状态"""
        self.min_width_spin.setEnabled(enabled)
        self.min_height_spin.setEnabled(enabled)
    
    def get_parameters(self):
        """获取在线数据收集参数"""
        # 获取搜索词
        search_terms_text = self.search_terms_edit.toPlainText().strip()
        search_terms = [term.strip() for term in search_terms_text.split('\n') if term.strip()]
        
        # 获取数据源
        sources = []
        if self.google_cb.isChecked():
            sources.append("google")
        if self.bing_cb.isChecked():
            sources.append("bing")
        if self.baidu_cb.isChecked():
            sources.append("baidu")
        if self.flickr_cb.isChecked():
            sources.append("flickr")
        if self.custom_cb.isChecked():
            sources.append("custom")
        
        # 如果没有选择任何数据源，默认使用Google和百度
        if not sources:
            sources = ["google", "baidu"]
        
        params = {
            "search_terms": search_terms,
            "output_dir": self.output_dir_edit.text() or "./data",
            "images_per_class": self.images_per_class_spin.value(),
            "sources": sources,
            "generate_labels": self.generate_labels_cb.isChecked(),
            "auto_classify": self.auto_classify_cb.isChecked(),
            "quality_filter": self.quality_filter_cb.isChecked(),
            "deduplicate": self.deduplicate_cb.isChecked(),
            "size_filter": {
                "enabled": self.size_filter_cb.isChecked(),
                "min_width": self.min_width_spin.value(),
                "min_height": self.min_height_spin.value()
            }
        }
        
        return params 

    def show_progress(self, visible: bool = True):
        """显示或隐藏进度条"""
        self.progress_bar.setVisible(visible)
        if visible:
            self.progress_bar.setValue(0)
        
    def update_progress(self, current: int, total: int, status: str = None):
        """更新进度条和状态"""
        if status:
            self.status_label.setText(status)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        QApplication.processEvents()  # 确保UI更新 