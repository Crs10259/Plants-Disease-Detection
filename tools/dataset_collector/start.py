#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
植物病害数据集制作工具启动脚本
此脚本用于启动数据收集工具，确保所有路径和模块导入正确
"""

import os
import sys

# 添加当前目录到模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入并运行主程序
try:
    import dataset_collector as dataset_collector
    dataset_collector.main()
except ImportError as e:
    print(f"导入模块时出错: {str(e)}")
    print("确保已安装所有依赖项: pip install -r requirements.txt")
    input("按回车键退出...")
except Exception as e:
    print(f"启动程序时出错: {str(e)}")
    import traceback
    traceback.print_exc()
    input("按回车键退出...") 