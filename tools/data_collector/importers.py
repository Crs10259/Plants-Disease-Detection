#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                            QLineEdit, QFileDialog, QProgressBar, QMessageBox,
                            QComboBox, QSpinBox, QCheckBox, QGroupBox, QFormLayout,
                            QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

logger = logging.getLogger('Importer')

class ImporterTab(QWidget):
    """导入用户数据选项卡"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """设置用户界面"""
        main_layout = QVBoxLayout(self)
        
        # 创建源目录选择组
        source_group = QGroupBox("源数据目录")
        source_layout = QFormLayout()
        
        # 源目录选择控件
        source_dir_layout = QHBoxLayout()
        self.source_dir_edit = QLineEdit()
        self.source_dir_edit.setPlaceholderText("选择包含图像的源目录")
        self.browse_source_btn = QPushButton("浏览...")
        source_dir_layout.addWidget(self.source_dir_edit, 1)
        source_dir_layout.addWidget(self.browse_source_btn)
        
        # 扫描按钮
        self.scan_btn = QPushButton("扫描目录")
        self.scan_btn.setMinimumHeight(30)
        
        # 将控件添加到布局
        source_layout.addRow("源目录:", source_dir_layout)
        source_layout.addRow("", self.scan_btn)
        source_group.setLayout(source_layout)
        
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
        
        # 添加分割比例选择器
        split_layout = QHBoxLayout()
        self.train_spin = QSpinBox()
        self.train_spin.setValue(80)
        self.train_spin.setMinimum(10)
        self.train_spin.setMaximum(90)
        self.train_spin.setSuffix("%")
        
        self.val_spin = QSpinBox()
        self.val_spin.setValue(10)
        self.val_spin.setMinimum(0)
        self.val_spin.setMaximum(40)
        self.val_spin.setSuffix("%")
        
        self.test_spin = QSpinBox()
        self.test_spin.setValue(10)
        self.test_spin.setMinimum(0)
        self.test_spin.setMaximum(40) 
        self.test_spin.setSuffix("%")
        
        split_layout.addWidget(QLabel("训练集:"))
        split_layout.addWidget(self.train_spin)
        split_layout.addWidget(QLabel("验证集:"))
        split_layout.addWidget(self.val_spin)
        split_layout.addWidget(QLabel("测试集:"))
        split_layout.addWidget(self.test_spin)
        
        # 添加标签生成选项
        self.generate_labels_cb = QCheckBox("生成标签文件")
        self.generate_labels_cb.setChecked(True)
        
        # 将控件添加到布局
        output_layout.addRow("输出目录:", output_dir_layout)
        output_layout.addRow("数据分割:", split_layout)
        output_layout.addRow("", self.generate_labels_cb)
        output_group.setLayout(output_layout)
        
        # 创建类别映射表
        mapping_group = QGroupBox("类别映射配置")
        mapping_layout = QVBoxLayout()
        
        # 说明标签
        mapping_label = QLabel("在下面的表格中设置源文件夹到类别的映射：")
        
        # 表格
        self.mapping_table = QTableWidget(0, 2)
        self.mapping_table.setHorizontalHeaderLabels(["文件夹名称", "类别名称"])
        self.mapping_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.mapping_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.mapping_table.setMinimumHeight(200)
        
        # 添加到布局
        mapping_layout.addWidget(mapping_label)
        mapping_layout.addWidget(self.mapping_table)
        mapping_group.setLayout(mapping_layout)
        
        # 添加高级选项组
        advanced_group = QGroupBox("高级选项")
        advanced_layout = QFormLayout()

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
        advanced_layout.addRow("", size_filter_layout)
        advanced_group.setLayout(advanced_layout)
        
        # 状态栏
        status_layout = QHBoxLayout()
        self.status_label = QLabel("就绪")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.progress_bar)
        
        # 添加所有组件到主布局
        main_layout.addWidget(source_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(mapping_group, 1)
        main_layout.addWidget(advanced_group, 1)
        main_layout.addLayout(status_layout)
        
    def connect_signals(self):
        """连接信号和槽"""
        self.browse_source_btn.clicked.connect(self.browse_source_directory)
        self.browse_output_btn.clicked.connect(self.browse_output_directory)
        self.scan_btn.clicked.connect(self.scan_source_directory)
        
        # 连接微调框的信号
        self.train_spin.valueChanged.connect(self.update_split_values)
        self.val_spin.valueChanged.connect(self.update_split_values)
        self.test_spin.valueChanged.connect(self.update_split_values)
        
        # 连接尺寸过滤复选框
        self.size_filter_cb.toggled.connect(self.toggle_size_filter)
        
    def browse_source_directory(self):
        """浏览源目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择源数据目录")
        if directory:
            self.source_dir_edit.setText(directory)
    
    def browse_output_directory(self):
        """浏览输出目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择数据集输出目录")
        if directory:
            self.output_dir_edit.setText(directory)
    
    def scan_source_directory(self):
        """扫描源目录，查找子目录并填充映射表"""
        source_dir = self.source_dir_edit.text()
        if not source_dir or not os.path.isdir(source_dir):
            QMessageBox.warning(self, "错误", "请选择有效的源目录")
            return
            
        try:
            # 获取子目录
            subdirs = [d for d in os.listdir(source_dir) 
                      if os.path.isdir(os.path.join(source_dir, d)) and not d.startswith('.')]
            
            if not subdirs:
                QMessageBox.warning(self, "警告", "在源目录中未找到子目录。确保每个类别都有单独的文件夹。")
                return
            
            # 清除表格
            self.mapping_table.setRowCount(0)
            
            # 为每个子目录添加一行
            for i, subdir in enumerate(sorted(subdirs)):
                self.mapping_table.insertRow(i)
                self.mapping_table.setItem(i, 0, QTableWidgetItem(subdir))
                self.mapping_table.setItem(i, 1, QTableWidgetItem(subdir.replace("_", " ").title()))
                
            self.status_label.setText(f"找到 {len(subdirs)} 个类别文件夹")
            
        except Exception as e:
            logger.error(f"扫描目录时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"扫描目录时出错: {str(e)}")
            
    def update_split_values(self):
        """确保列车、验证和测试分割值总和为100%"""
        sender = self.sender()
        
        train = self.train_spin.value()
        val = self.val_spin.value()
        test = self.test_spin.value()
        
        total = train + val + test
        
        if total != 100:
            # 调整其他值使总和为100
            if sender == self.train_spin:
                # 保持val和test的比例，但调整它们使总和为100
                if val + test > 0:
                    ratio = (100 - train) / (val + test)
                    new_val = int(val * ratio)
                    new_test = 100 - train - new_val
                    self.val_spin.blockSignals(True)
                    self.test_spin.blockSignals(True)
                    self.val_spin.setValue(new_val)
                    self.test_spin.setValue(new_test)
                    self.val_spin.blockSignals(False)
                    self.test_spin.blockSignals(False)
                else:
                    # 如果val和test都为0，将val设为剩余的百分比
                    self.val_spin.blockSignals(True)
                    self.val_spin.setValue(100 - train)
                    self.val_spin.blockSignals(False)
            
            elif sender == self.val_spin:
                # 主要调整test的值
                new_test = 100 - train - val
                if new_test < 0:
                    new_train = 100 - val
                    new_test = 0
                    self.train_spin.blockSignals(True)
                    self.train_spin.setValue(new_train)
                    self.train_spin.blockSignals(False)
                
                self.test_spin.blockSignals(True)
                self.test_spin.setValue(new_test)
                self.test_spin.blockSignals(False)
            
            elif sender == self.test_spin:
                # 主要调整val的值
                new_val = 100 - train - test
                if new_val < 0:
                    new_train = 100 - test
                    new_val = 0
                    self.train_spin.blockSignals(True)
                    self.train_spin.setValue(new_train)
                    self.train_spin.blockSignals(False)
                
                self.val_spin.blockSignals(True)
                self.val_spin.setValue(new_val)
                self.val_spin.blockSignals(False)
    
    def toggle_size_filter(self, enabled):
        """切换尺寸过滤选项的启用状态"""
        self.min_width_spin.setEnabled(enabled)
        self.min_height_spin.setEnabled(enabled)

    def get_parameters(self):
        """获取导入参数"""
        # 构建类别映射
        class_mapping = {}
        for row in range(self.mapping_table.rowCount()):
            folder_name = self.mapping_table.item(row, 0).text()
            class_name = self.mapping_table.item(row, 1).text()
            class_mapping[folder_name] = class_name
        
        params = {
            "source_dir": self.source_dir_edit.text(),
            "output_dir": self.output_dir_edit.text() or "./data",
            "split_ratio": {
                "train": self.train_spin.value() / 100,
                "val": self.val_spin.value() / 100,
                "test": self.test_spin.value() / 100
            },
            "class_mapping": class_mapping,
            "generate_labels": self.generate_labels_cb.isChecked(),
            "size_filter": {
                "enabled": self.size_filter_cb.isChecked(),
                "min_width": self.min_width_spin.value(),
                "min_height": self.min_height_spin.value()
            }
        }
        
        return params 