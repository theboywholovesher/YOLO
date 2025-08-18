import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from PIL import Image, ImageTk
import sys
import ctypes

# 隐藏控制台窗口（解决打包后出现黑框的问题）
try:
    # Windows系统
    if os.name == 'nt':
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
except:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strange_img import batch_augment_images, augment_image_with_labels, batch_xml_to_yolo


class ImageAugmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图像增强工具")
        self.root.geometry("900x800")
        self.root.resizable(True, True)
        
        # 设置样式
        style = ttk.Style()
        style.theme_use('clam')
        
        # 变量
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.label_output_dir = tk.StringVar()
        self.selected_augmentations = []
        self.processing = False
        
        # 增强选项
        self.augmentation_options = {
            "original": "原图",
            "flip_horizontal": "左右翻转",
            "flip_vertical": "上下翻转", 
            "rotate_90": "旋转90°",
            "rotate_180": "旋转180°",
            "rotate_270": "旋转270°",
            "brightness_up": "亮度增加",
            "brightness_down": "亮度减少",
            "contrast_up": "对比度增加",
            "blur": "模糊",
            "sharpen": "锐化",
            "noise": "添加噪声"
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="图像增强工具", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 输入目录选择
        ttk.Label(main_frame, text="输入目录:").grid(row=1, column=0, sticky=tk.W, pady=5)
        input_entry = ttk.Entry(main_frame, textvariable=self.input_dir, width=50)
        input_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_input_dir).grid(row=1, column=2, pady=5)
        
        # 输出目录选择
        ttk.Label(main_frame, text="输出目录:").grid(row=2, column=0, sticky=tk.W, pady=5)
        output_entry = ttk.Entry(main_frame, textvariable=self.output_dir, width=50)
        output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_output_dir).grid(row=2, column=2, pady=5)
        
        # 标签文件输出目录选择
        ttk.Label(main_frame, text="标签输出目录:").grid(row=3, column=0, sticky=tk.W, pady=5)
        label_output_entry = ttk.Entry(main_frame, textvariable=self.label_output_dir, width=50)
        label_output_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_label_output_dir).grid(row=3, column=2, pady=5)
        
        # 标签输出目录说明
        ttk.Label(main_frame, text="(可选，留空则与图像输出目录相同)", 
                 font=("Arial", 8), foreground="gray").grid(row=4, column=1, sticky=tk.W, padx=10)
        
        # 增强选项选择
        ttk.Label(main_frame, text="增强选项:", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
        
        # 创建选项框架
        options_frame = ttk.Frame(main_frame)
        options_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # 创建复选框
        self.checkboxes = {}
        row = 0
        col = 0
        max_cols = 3
        
        for key, value in self.augmentation_options.items():
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(options_frame, text=value, variable=var, 
                               command=lambda k=key, v=var: self.on_checkbox_change(k, v))
            cb.grid(row=row, column=col, sticky=tk.W, padx=10, pady=2)
            self.checkboxes[key] = var
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # 全选/取消全选按钮
        select_frame = ttk.Frame(main_frame)
        select_frame.grid(row=7, column=0, columnspan=3, pady=10)
        
        ttk.Button(select_frame, text="全选", command=self.select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_frame, text="取消全选", command=self.deselect_all).pack(side=tk.LEFT, padx=5)
        
        # 工具按钮
        tools_frame = ttk.Frame(main_frame)
        tools_frame.grid(row=8, column=0, columnspan=3, pady=10)
        
        ttk.Button(tools_frame, text="XML转YOLO", command=self.xml_to_yolo_conversion).pack(side=tk.LEFT, padx=5)
        ttk.Button(tools_frame, text="查看类别映射", command=self.view_class_mapping).pack(side=tk.LEFT, padx=5)
        
        # 进度条
        ttk.Label(main_frame, text="处理进度:", font=("Arial", 12, "bold")).grid(row=9, column=0, columnspan=3, sticky=tk.W, pady=(20, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        # 状态标签
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, font=("Arial", 10))
        status_label.grid(row=11, column=0, columnspan=3, pady=5)
        
        # 操作按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=12, column=0, columnspan=3, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="开始处理", command=self.start_processing, 
                                     style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.LEFT, padx=10)
        
        # 预览区域
        preview_frame = ttk.LabelFrame(main_frame, text="预览", padding="10")
        preview_frame.grid(row=13, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        self.preview_label = ttk.Label(preview_frame, text="选择输入目录后显示预览")
        self.preview_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置主框架的行权重
        main_frame.rowconfigure(13, weight=1)
        
    def browse_input_dir(self):
        """浏览输入目录"""
        directory = filedialog.askdirectory(title="选择输入目录")
        if directory:
            self.input_dir.set(directory)
            self.update_preview()
            
    def browse_output_dir(self):
        """浏览输出目录"""
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_dir.set(directory)
            
    def browse_label_output_dir(self):
        """浏览标签文件输出目录"""
        directory = filedialog.askdirectory(title="选择标签文件输出目录")
        if directory:
            self.label_output_dir.set(directory)
            
    def xml_to_yolo_conversion(self):
        """XML转YOLO格式转换"""
        if not self.input_dir.get():
            messagebox.showerror("错误", "请先选择输入目录")
            return
            
        # 选择输出目录
        output_dir = filedialog.askdirectory(title="选择YOLO格式输出目录")
        if not output_dir:
            return
            
        # 在新线程中执行转换
        thread = threading.Thread(target=self._xml_to_yolo_worker, args=(output_dir,))
        thread.daemon = True
        thread.start()
        
    def _xml_to_yolo_worker(self, output_dir):
        """XML转YOLO工作线程"""
        try:
            self.root.after(0, lambda: self.status_var.set("正在转换XML到YOLO格式..."))
            self.root.after(0, lambda: self.progress_var.set(0))
            
            success = batch_xml_to_yolo(self.input_dir.get(), output_dir)
            
            if success:
                self.root.after(0, lambda: self.status_var.set("XML转YOLO完成！"))
                self.root.after(0, lambda: messagebox.showinfo("完成", "XML转YOLO格式转换完成！"))
            else:
                self.root.after(0, lambda: self.status_var.set("转换失败"))
                self.root.after(0, lambda: messagebox.showerror("错误", "XML转YOLO格式转换失败"))
                
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set("转换出错"))
            self.root.after(0, lambda: messagebox.showerror("错误", f"转换过程中出现错误：\n{str(e)}"))
            
    def view_class_mapping(self):
        """查看类别映射"""
        if not self.input_dir.get():
            messagebox.showerror("错误", "请先选择输入目录")
            return
            
        # 查找第一个XML文件来获取类别信息
        xml_files = [f for f in os.listdir(self.input_dir.get()) if f.lower().endswith('.xml')]
        if not xml_files:
            messagebox.showinfo("信息", "输入目录中没有找到XML文件")
            return
            
        first_xml = os.path.join(self.input_dir.get(), xml_files[0])
        try:
            from strange_img import parse_xml
            width, height, objects = parse_xml(first_xml)
            if objects:
                unique_classes = list(set([obj['name'] for obj in objects]))
                class_mapping = {class_name: i for i, class_name in enumerate(unique_classes)}
                
                # 显示类别映射
                mapping_text = "类别映射:\n"
                for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
                    mapping_text += f"{class_id}: {class_name}\n"
                    
                messagebox.showinfo("类别映射", mapping_text)
            else:
                messagebox.showinfo("信息", "XML文件中没有找到标注对象")
        except Exception as e:
            messagebox.showerror("错误", f"读取类别信息失败：{str(e)}")
            
    def update_preview(self):
        """更新预览"""
        input_dir = self.input_dir.get()
        if not input_dir or not os.path.exists(input_dir):
            return
            
        # 查找第一张图片作为预览
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        preview_image = None
        
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                try:
                    image_path = os.path.join(input_dir, file)
                    img = Image.open(image_path)
                    
                    # 调整图片大小以适应预览区域
                    img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    
                    self.preview_label.configure(image=photo, text="")
                    self.preview_label.image = photo  # 保持引用
                    break
                except Exception as e:
                    print(f"预览图片失败: {e}")
                    continue
                    
    def on_checkbox_change(self, key, var):
        """复选框状态改变时的处理"""
        if var.get():
            if key not in self.selected_augmentations:
                self.selected_augmentations.append(key)
        else:
            if key in self.selected_augmentations:
                self.selected_augmentations.remove(key)
                
    def select_all(self):
        """全选所有选项"""
        for var in self.checkboxes.values():
            var.set(True)
        self.selected_augmentations = list(self.augmentation_options.keys())
        
    def deselect_all(self):
        """取消全选"""
        for var in self.checkboxes.values():
            var.set(False)
        self.selected_augmentations = []
        
    def start_processing(self):
        """开始处理图像"""
        if self.processing:
            return
            
        # 验证输入
        if not self.input_dir.get():
            messagebox.showerror("错误", "请选择输入目录")
            return
            
        if not self.output_dir.get():
            messagebox.showerror("错误", "请选择输出目录")
            return
            
        if not self.selected_augmentations:
            messagebox.showerror("错误", "请至少选择一个增强选项")
            return
            
        # 开始处理
        self.processing = True
        self.start_button.configure(state="disabled")
        self.status_var.set("正在处理...")
        self.progress_var.set(0)
        
        # 在新线程中处理
        thread = threading.Thread(target=self.process_images)
        thread.daemon = True
        thread.start()
        
    def process_images(self):
        """处理图像的函数"""
        try:
            input_dir = self.input_dir.get()
            output_dir = self.output_dir.get()
            label_output_dir = self.label_output_dir.get() if self.label_output_dir.get() else None
            augmentations = self.selected_augmentations
            
            def progress_callback(progress):
                self.root.after(0, lambda: self.progress_var.set(progress))
                
            # 执行批量增强
            success = batch_augment_images(
                input_dir, output_dir, augmentations, progress_callback, label_output_dir
            )
            
            # 处理完成
            self.root.after(0, self.processing_completed, success)
            
        except Exception as e:
            self.root.after(0, self.processing_error, str(e))
            
    def processing_completed(self, success):
        """处理完成"""
        self.processing = False
        self.start_button.configure(state="normal")
        
        if success:
            self.status_var.set("处理完成！")
            messagebox.showinfo("完成", "图像增强处理完成！")
        else:
            self.status_var.set("处理失败")
            messagebox.showerror("错误", "图像增强处理失败")
            
    def processing_error(self, error_msg):
        """处理出错"""
        self.processing = False
        self.start_button.configure(state="normal")
        self.status_var.set("处理出错")
        messagebox.showerror("错误", f"处理过程中出现错误：\n{error_msg}")


def main():
    """主函数"""
    root = tk.Tk()
    app = ImageAugmentationGUI(root)
    
    # 设置窗口图标（如果有的话）
    try:
        root.iconbitmap("icon.ico")
    except:
        pass
        
    root.mainloop()


if __name__ == "__main__":
    main()
