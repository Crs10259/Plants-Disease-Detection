import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QScrollArea, QGridLayout, QMessageBox, QFrame)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QRect
from PIL import Image

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据集图片检查器")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                color: #ffffff;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #666666;
            }
            QScrollArea {
                border: 1px solid #3d3d3d;
                border-radius: 5px;
            }
            QLabel {
                color: #ffffff;
            }
        """)
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)
        
        # Control buttons
        self.control_layout = QHBoxLayout()
        self.control_layout.setSpacing(10)
        
        self.select_dir_btn = QPushButton("选择文件夹")
        self.mode_toggle_btn = QPushButton("切换到多图模式")  # 默认显示切换到多图模式
        self.delete_btn = QPushButton("删除选中")
        self.renumber_btn = QPushButton("重新排号选中")
        self.renumber_all_btn = QPushButton("重新排号所有")
        
        # Add initial button
        self.control_layout.addWidget(self.select_dir_btn)
        
        # Hide other buttons initially
        self.mode_toggle_btn.hide()
        self.delete_btn.hide()
        self.renumber_btn.hide()
        self.renumber_all_btn.hide()
        
        self.control_layout.addWidget(self.mode_toggle_btn)
        self.control_layout.addWidget(self.delete_btn)
        self.control_layout.addWidget(self.renumber_btn)
        self.control_layout.addWidget(self.renumber_all_btn)
        
        # Add stretch to keep buttons left-aligned
        self.control_layout.addStretch()
        
        self.layout.addLayout(self.control_layout)
        
        # Message label for empty state
        self.empty_label = QLabel("请选择一个包含图片的文件夹")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 16px;
                padding: 20px;
            }
        """)
        self.layout.addWidget(self.empty_label)
        
        # Scroll area for images
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.grid_layout.setSpacing(10)
        self.scroll.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll)
        
        # Hide scroll area initially
        self.scroll.hide()
        
        # Connect buttons
        self.select_dir_btn.clicked.connect(self.select_directory)
        self.mode_toggle_btn.clicked.connect(self.toggle_view_mode)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.renumber_btn.clicked.connect(self.renumber_selected)
        self.renumber_all_btn.clicked.connect(self.renumber_all)
        
        # Initialize variables
        self.current_dir = ""
        self.image_files = []
        self.current_index = 0
        self.selected_images = set()
        self.is_single_mode = True
        self.image_labels = []
    
    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if dir_path:
            self.current_dir = dir_path
            self.image_files = [f for f in os.listdir(dir_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if not self.image_files:
                QMessageBox.warning(self, "警告", "未找到图片")
                return
                
            # Show control buttons
            self.mode_toggle_btn.show()
            self.delete_btn.show()
            self.renumber_btn.show()
            self.renumber_all_btn.show()
            
            # Hide empty label and show scroll area
            self.empty_label.hide()
            self.scroll.show()
            
            self.current_index = 0
            self.selected_images.clear()
            if self.is_single_mode:
                self.show_single_image()
            else:
                self.show_multiple_images()
    
    def toggle_view_mode(self):
        """切换查看模式"""
        self.is_single_mode = not self.is_single_mode
        # 更新按钮文本
        self.mode_toggle_btn.setText("Switch to Multi Mode" if self.is_single_mode else "Switch to Single Mode")
        # 切换显示模式
        if self.is_single_mode:
            self.show_single_image()
        else:
            self.show_multiple_images()
    
    def show_single_image(self):
        self.clear_layout()
        
        if not self.image_files:
            return
            
        # Create image display
        image_path = os.path.join(self.current_dir, self.image_files[self.current_index])
        pixmap = QPixmap(image_path)
        image_label = QLabel()
        image_label.setPixmap(pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))
        image_label.setStyleSheet("border: 2px solid #3d3d3d; border-radius: 5px; padding: 5px;")
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(10)
        
        prev_btn = QPushButton("上一张")
        next_btn = QPushButton("下一张")
        
        # Update navigation button states
        prev_btn.setEnabled(self.current_index > 0)
        next_btn.setEnabled(self.current_index < len(self.image_files) - 1)
        
        prev_btn.clicked.connect(self.show_previous)
        next_btn.clicked.connect(self.show_next)
        
        nav_layout.addStretch()
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        nav_layout.addStretch()
        
        self.grid_layout.addWidget(image_label, 0, 0, Qt.AlignmentFlag.AlignCenter)
        self.grid_layout.addLayout(nav_layout, 1, 0)
        
        # Add image counter
        counter_label = QLabel(f"第 {self.current_index + 1} 张图片，共 {len(self.image_files)} 张")
        counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        counter_label.setStyleSheet("color: #888888; margin-top: 10px;")
        self.grid_layout.addWidget(counter_label, 2, 0)
    
    def show_multiple_images(self):
        self.clear_layout()
        
        if not self.image_files:
            return
            
        # Show grid of images
        cols = 4
        for i, image_file in enumerate(self.image_files):
            image_path = os.path.join(self.current_dir, image_file)
            pixmap = QPixmap(image_path)
            
            # Create a frame for the image
            frame = QFrame()
            frame.setStyleSheet("""
                QFrame {
                    border: 2px solid #3d3d3d;
                    border-radius: 5px;
                    padding: 5px;
                    background-color: #2b2b2b;
                }
            """)
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(5, 5, 5, 5)
            
            label = QLabel()
            label.setPixmap(pixmap.scaled(280, 280, Qt.AspectRatioMode.KeepAspectRatio))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            frame_layout.addWidget(label)
            
            # Add filename label
            name_label = QLabel(image_file)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet("color: #888888; font-size: 10px;")
            frame_layout.addWidget(name_label)
            
            frame.mousePressEvent = lambda evt, index=i: self.toggle_image_selection(index)
            self.image_labels.append(frame)
            self.grid_layout.addWidget(frame, i // cols, i % cols)
    
    def toggle_image_selection(self, index):
        if index in self.selected_images:
            self.selected_images.remove(index)
            self.image_labels[index].setStyleSheet("""
                QFrame {
                    border: 2px solid #3d3d3d;
                    border-radius: 5px;
                    padding: 5px;
                    background-color: #2b2b2b;
                }
            """)
        else:
            self.selected_images.add(index)
            self.image_labels[index].setStyleSheet("""
                QFrame {
                    border: 2px solid #ff5555;
                    border-radius: 5px;
                    padding: 5px;
                    background-color: #2b2b2b;
                }
            """)
    
    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_single_image()
    
    def show_next(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_single_image()
    
    def delete_selected(self):
        if not self.selected_images and self.is_single_mode:
            self.selected_images.add(self.current_index)
            
        if not self.selected_images:
            QMessageBox.warning(self, "警告", "未选中图片")
            return
            
        reply = QMessageBox.question(self, "确认删除", 
                                   f"确定要删除 {len(self.selected_images)} 张图片吗？",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            indices = sorted(list(self.selected_images), reverse=True)
            for index in indices:
                image_path = os.path.join(self.current_dir, self.image_files[index])
                try:
                    os.remove(image_path)
                    del self.image_files[index]
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to delete {image_path}: {str(e)}")
            
            self.selected_images.clear()
            
            if not self.image_files:
                # No images left, reset the view
                self.current_dir = ""
                self.mode_toggle_btn.hide()
                self.delete_btn.hide()
                self.renumber_btn.hide()
                self.renumber_all_btn.hide()
                self.scroll.hide()
                self.empty_label.show()
            else:
                if self.is_single_mode:
                    self.current_index = min(self.current_index, len(self.image_files) - 1)
                    self.show_single_image()
                else:
                    self.show_multiple_images()
    
    def renumber_selected(self):
        """重新排号选中的图片"""
        if not self.selected_images and self.is_single_mode:
            self.selected_images.add(self.current_index)
            
        if not self.selected_images:
            QMessageBox.warning(self, "警告", "未选中图片")
            return
            
        # 获取文件扩展名
        def get_extension(filename):
            return os.path.splitext(filename)[1].lower()
            
        # 获取所有选中的图片及其索引
        selected_files = [(index, self.image_files[index]) for index in sorted(self.selected_images)]
        
        # 询问用户起始编号
        start_number = 1
        
        # 确认重命名
        reply = QMessageBox.question(
            self,
            "确认重新排号",
            f"确定要重新排号 {len(selected_files)} 张图片，从 {start_number} 开始吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 记录重命名操作，用于处理可能的错误
            rename_operations = []
            new_filenames = []
            
            try:
                # 首先检查新文件名是否会有冲突
                for i, (_, old_filename) in enumerate(selected_files):
                    ext = get_extension(old_filename)
                    new_filename = f"{start_number + i:04d}{ext}"
                    new_path = os.path.join(self.current_dir, new_filename)
                    
                    if new_filename in new_filenames:
                        raise Exception(f"Duplicate filename would be created: {new_filename}")
                    new_filenames.append(new_filename)
                    
                    old_path = os.path.join(self.current_dir, old_filename)
                    rename_operations.append((old_path, new_path, old_filename, new_filename))
                
                # 执行重命名
                for old_path, new_path, old_filename, new_filename in rename_operations:
                    os.rename(old_path, new_path)
                    # 更新image_files列表中的文件名
                    idx = self.image_files.index(old_filename)
                    self.image_files[idx] = new_filename
                
                # 清除选择
                self.selected_images.clear()
                
                # 刷新显示
                if self.is_single_mode:
                    self.show_single_image()
                else:
                    self.show_multiple_images()
                    
                QMessageBox.information(
                    self,
                    "成功",
                    f"成功重新排号 {len(selected_files)} 张图片"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "错误",
                    f"重新排号时发生错误: {str(e)}\n没有文件被重命名。"
                )
    
    def renumber_all(self):
        """重新排号所有图片"""
        if not self.image_files:
            QMessageBox.warning(self, "警告", "当前目录没有图片")
            return
            
        # 获取文件扩展名
        def get_extension(filename):
            return os.path.splitext(filename)[1].lower()
            
        # 确认重命名
        reply = QMessageBox.question(
            self,
            "确认重新排号所有",
            f"确定要重新排号所有 {len(self.image_files)} 张图片吗？\n这将按顺序重命名所有图片。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 记录重命名操作，用于处理可能的错误
            rename_operations = []
            new_filenames = []
            start_number = 1
            
            try:
                # 首先检查新文件名是否会有冲突
                for i, old_filename in enumerate(self.image_files):
                    ext = get_extension(old_filename)
                    new_filename = f"{start_number + i:04d}{ext}"
                    new_path = os.path.join(self.current_dir, new_filename)
                    
                    if new_filename in new_filenames:
                        raise Exception(f"Duplicate filename would be created: {new_filename}")
                    new_filenames.append(new_filename)
                    
                    old_path = os.path.join(self.current_dir, old_filename)
                    rename_operations.append((old_path, new_path, old_filename, new_filename))
                
                # 执行重命名
                for old_path, new_path, old_filename, new_filename in rename_operations:
                    os.rename(old_path, new_path)
                    # 更新image_files列表中的文件名
                    idx = self.image_files.index(old_filename)
                    self.image_files[idx] = new_filename
                
                # 清除选择
                self.selected_images.clear()
                
                # 刷新显示
                if self.is_single_mode:
                    self.show_single_image()
                else:
                    self.show_multiple_images()
                    
                QMessageBox.information(
                    self,
                    "成功",
                    f"成功重新排号所有 {len(self.image_files)} 张图片"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "错误",
                    f"重新排号时发生错误: {str(e)}\n没有文件被重命名。"
                )
    
    def clear_layout(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub_item = item.layout().takeAt(0)
                    if sub_item.widget():
                        sub_item.widget().deleteLater()
        self.image_labels.clear()

def main():
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec()) 