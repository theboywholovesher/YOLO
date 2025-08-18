#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像增强工具启动器
选择要运行的GUI版本
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os


class LauncherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图像增强工具启动器")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # 设置样式
        style = ttk.Style()
        style.theme_use('clam')
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="图像增强工具启动器", font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 30))
        
        # 说明文字
        desc_label = ttk.Label(main_frame, text="请选择要启动的工具版本：", font=("Arial", 12))
        desc_label.pack(pady=(0, 20))
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        # 基础版本按钮
        basic_btn = ttk.Button(button_frame, text="基础版本", 
                              command=self.run_basic_version, 
                              style="Accent.TButton")
        basic_btn.pack(fill=tk.X, pady=5)
        
        # 高级版本按钮
        advanced_btn = ttk.Button(button_frame, text="高级版本", 
                                 command=self.run_advanced_version)
        advanced_btn.pack(fill=tk.X, pady=5)
        
        # XML转YOLO工具按钮
        converter_btn = ttk.Button(button_frame, text="XML转YOLO工具", 
                                  command=self.run_converter_tool)
        converter_btn.pack(fill=tk.X, pady=5)
        
        # 分隔线
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=20)
        
        # 功能说明
        info_frame = ttk.LabelFrame(main_frame, text="功能说明", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        info_text = """
基础版本: 简单易用的图像增强工具，支持基本的增强功能

高级版本: 功能丰富的图像增强工具，支持参数调整和实时预览

XML转YOLO工具: 专门用于将VOC格式的XML标注转换为YOLO格式

所有工具都支持自动标签文件处理和多种标注格式！
        """
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT, font=("Arial", 10))
        info_label.pack(anchor=tk.NW)
        
    def run_basic_version(self):
        """运行基础版本"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), "image_augmentation_gui.py")
            subprocess.Popen([sys.executable, script_path])
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("错误", f"启动基础版本失败：{str(e)}")
            
    def run_advanced_version(self):
        """运行高级版本"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), "advanced_image_augmentation_gui.py")
            subprocess.Popen([sys.executable, script_path])
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("错误", f"启动高级版本失败：{str(e)}")
            
    def run_converter_tool(self):
        """运行XML转YOLO工具"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), "xml_to_yolo_converter.py")
            subprocess.Popen([sys.executable, script_path])
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("错误", f"启动XML转YOLO工具失败：{str(e)}")


def main():
    """主函数"""
    root = tk.Tk()
    app = LauncherGUI(root)
    
    # 设置窗口图标（如果有的话）
    try:
        root.iconbitmap("icon.ico")
    except:
        pass
        
    root.mainloop()


if __name__ == "__main__":
    main()
