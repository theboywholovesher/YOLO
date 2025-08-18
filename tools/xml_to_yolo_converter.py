#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XML转YOLO格式转换工具
专门用于将VOC格式的XML标注文件转换为YOLO格式的txt文件
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strange_img import batch_xml_to_yolo, parse_xml


class XMLToYOLOConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("XML转YOLO格式转换工具")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 设置样式
        style = ttk.Style()
        style.theme_use('clam')
        
        # 变量
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.class_mapping = {}
        self.processing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="XML转YOLO格式转换工具", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        # 输入目录
        ttk.Label(file_frame, text="XML文件输入目录:").pack(anchor=tk.W, pady=2)
        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(input_frame, textvariable=self.input_dir, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="浏览", command=self.browse_input_dir).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 输出目录
        ttk.Label(file_frame, text="YOLO格式输出目录:").pack(anchor=tk.W, pady=(10, 2))
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(output_frame, textvariable=self.output_dir, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="浏览", command=self.browse_output_dir).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 操作按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="扫描类别", command=self.scan_classes).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="编辑类别映射", command=self.edit_class_mapping).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="开始转换", command=self.start_conversion).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清空", command=self.clear_all).pack(side=tk.RIGHT, padx=5)
        
        # 类别映射显示区域
        mapping_frame = ttk.LabelFrame(main_frame, text="类别映射", padding="10")
        mapping_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建文本框和滚动条
        self.mapping_text = scrolledtext.ScrolledText(mapping_frame, height=15, width=80)
        self.mapping_text.pack(fill=tk.BOTH, expand=True)
        
        # 进度和状态区域
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 10))
        status_label.pack()
        
    def browse_input_dir(self):
        """浏览输入目录"""
        directory = filedialog.askdirectory(title="选择包含XML文件的目录")
        if directory:
            self.input_dir.set(directory)
            self.scan_classes()
            
    def browse_output_dir(self):
        """浏览输出目录"""
        directory = filedialog.askdirectory(title="选择YOLO格式输出目录")
        if directory:
            self.output_dir.set(directory)
            
    def scan_classes(self):
        """扫描XML文件中的类别"""
        input_dir = self.input_dir.get()
        if not input_dir or not os.path.exists(input_dir):
            messagebox.showerror("错误", "请先选择有效的输入目录")
            return
            
        try:
            # 查找所有XML文件
            xml_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.xml')]
            if not xml_files:
                messagebox.showinfo("信息", "输入目录中没有找到XML文件")
                return
                
            # 扫描所有类别
            all_classes = set()
            for xml_file in xml_files[:10]:  # 只扫描前10个文件以提高速度
                xml_path = os.path.join(input_dir, xml_file)
                try:
                    width, height, objects = parse_xml(xml_path)
                    if objects:
                        for obj in objects:
                            all_classes.add(obj['name'])
                except Exception as e:
                    print(f"解析文件 {xml_file} 失败: {e}")
                    continue
                    
            # 创建类别映射
            self.class_mapping = {class_name: i for i, class_name in enumerate(sorted(all_classes))}
            
            # 显示类别映射
            self.update_mapping_display()
            
            messagebox.showinfo("完成", f"扫描完成！发现 {len(self.class_mapping)} 个类别")
            
        except Exception as e:
            messagebox.showerror("错误", f"扫描类别失败：{str(e)}")
            
    def edit_class_mapping(self):
        """编辑类别映射"""
        if not self.class_mapping:
            messagebox.showwarning("警告", "请先扫描类别")
            return
            
        # 创建编辑窗口
        edit_window = tk.Toplevel(self.root)
        edit_window.title("编辑类别映射")
        edit_window.geometry("500x400")
        edit_window.transient(self.root)
        edit_window.grab_set()
        
        # 创建编辑界面
        edit_frame = ttk.Frame(edit_window, padding="10")
        edit_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(edit_frame, text="编辑类别映射 (格式: 类别ID 类别名称)", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        # 创建文本框
        edit_text = scrolledtext.ScrolledText(edit_frame, height=20)
        edit_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 填充当前映射
        current_mapping = ""
        for class_name, class_id in sorted(self.class_mapping.items(), key=lambda x: x[1]):
            current_mapping += f"{class_id} {class_name}\n"
        edit_text.insert(tk.END, current_mapping)
        
        # 按钮
        button_frame = ttk.Frame(edit_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def save_mapping():
            try:
                # 解析编辑后的映射
                new_mapping = {}
                lines = edit_text.get("1.0", tk.END).strip().split('\n')
                
                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            class_id = int(parts[0])
                            class_name = ' '.join(parts[1:])
                            new_mapping[class_name] = class_id
                            
                self.class_mapping = new_mapping
                self.update_mapping_display()
                edit_window.destroy()
                messagebox.showinfo("成功", "类别映射已更新")
                
            except Exception as e:
                messagebox.showerror("错误", f"保存映射失败：{str(e)}")
                
        ttk.Button(button_frame, text="保存", command=save_mapping).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=edit_window.destroy).pack(side=tk.LEFT, padx=5)
        
    def update_mapping_display(self):
        """更新类别映射显示"""
        self.mapping_text.delete("1.0", tk.END)
        
        if not self.class_mapping:
            self.mapping_text.insert(tk.END, "未找到类别映射，请先扫描类别")
            return
            
        # 显示类别映射
        mapping_text = "类别映射 (格式: 类别ID 类别名称):\n"
        mapping_text += "=" * 50 + "\n\n"
        
        for class_name, class_id in sorted(self.class_mapping.items(), key=lambda x: x[1]):
            mapping_text += f"{class_id:3d} {class_name}\n"
            
        mapping_text += f"\n总计: {len(self.class_mapping)} 个类别"
        self.mapping_text.insert(tk.END, mapping_text)
        
    def start_conversion(self):
        """开始转换"""
        if self.processing:
            return
            
        # 验证输入
        if not self.input_dir.get():
            messagebox.showerror("错误", "请选择输入目录")
            return
            
        if not self.output_dir.get():
            messagebox.showerror("错误", "请选择输出目录")
            return
            
        if not self.class_mapping:
            messagebox.showerror("错误", "请先扫描类别")
            return
            
        # 开始转换
        self.processing = True
        self.status_var.set("正在转换...")
        self.progress_var.set(0)
        
        # 在新线程中执行转换
        thread = threading.Thread(target=self._conversion_worker)
        thread.daemon = True
        thread.start()
        
    def _conversion_worker(self):
        """转换工作线程"""
        try:
            success = batch_xml_to_yolo(
                self.input_dir.get(), 
                self.output_dir.get(), 
                self.class_mapping
            )
            
            if success:
                self.root.after(0, lambda: self.status_var.set("转换完成！"))
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: messagebox.showinfo("完成", "XML转YOLO格式转换完成！"))
            else:
                self.root.after(0, lambda: self.status_var.set("转换失败"))
                self.root.after(0, lambda: messagebox.showerror("错误", "XML转YOLO格式转换失败"))
                
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set("转换出错"))
            self.root.after(0, lambda: messagebox.showerror("错误", f"转换过程中出现错误：\n{str(e)}"))
        finally:
            self.root.after(0, lambda: setattr(self, 'processing', False))
            
    def clear_all(self):
        """清空所有设置"""
        self.input_dir.set("")
        self.output_dir.set("")
        self.class_mapping.clear()
        self.mapping_text.delete("1.0", tk.END)
        self.status_var.set("就绪")
        self.progress_var.set(0)


def main():
    """主函数"""
    root = tk.Tk()
    app = XMLToYOLOConverter(root)
    
    # 设置窗口图标（如果有的话）
    try:
        root.iconbitmap("icon.ico")
    except:
        pass
        
    root.mainloop()


if __name__ == "__main__":
    main()
