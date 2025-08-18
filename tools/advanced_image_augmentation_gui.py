import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import json
import ctypes
import numpy as np
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strange_img import batch_augment_images, augment_image_with_labels, batch_xml_to_yolo


# 隐藏控制台窗口（解决打包后出现黑框的问题）
try:
    # Windows系统
    if os.name == 'nt':
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
except:
    pass



class AdvancedImageAugmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("高级图像增强工具")
        self.root.geometry("1100x900")
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
        self.current_image = None
        self.current_image_path = None
        
        # 增强选项和参数
        self.augmentation_options = {
            "original": {"name": "原图", "params": {}},
            "flip_horizontal": {"name": "左右翻转", "params": {}},
            "flip_vertical": {"name": "上下翻转", "params": {}},
            "rotate_90": {"name": "旋转90°", "params": {}},
            "rotate_180": {"name": "旋转180°", "params": {}},
            "rotate_270": {"name": "旋转270°", "params": {}},
            "brightness_up": {"name": "亮度增加", "params": {"factor": 1.3}},
            "brightness_down": {"name": "亮度减少", "params": {"factor": 0.7}},
            "contrast_up": {"name": "对比度增加", "params": {"factor": 1.5}},
            "contrast_down": {"name": "对比度减少", "params": {"factor": 0.7}},
            "saturation_up": {"name": "饱和度增加", "params": {"factor": 1.5}},
            "saturation_down": {"name": "饱和度减少", "params": {"factor": 0.7}},
            "blur": {"name": "模糊", "params": {"radius": 2}},
            "sharpen": {"name": "锐化", "params": {"factor": 2.0}},
            "noise": {"name": "添加噪声", "params": {"factor": 0.1}},
            "edge_enhance": {"name": "边缘增强", "params": {"factor": 2.0}},
            "emboss": {"name": "浮雕效果", "params": {}},
            "find_edges": {"name": "查找边缘", "params": {}}
        }
        
        # 参数控件字典
        self.param_widgets = {}
        
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建主分割窗口
        paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧控制面板
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight=1)
        
        # 右侧预览面板
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=2)
        
        self.setup_left_panel(left_frame)
        self.setup_right_panel(right_frame)
        
    def setup_left_panel(self, parent):
        """设置左侧控制面板"""
        # 标题
        title_label = ttk.Label(parent, text="图像增强工具", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(parent, text="文件选择", padding="10")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 输入目录
        ttk.Label(file_frame, text="输入目录:").pack(anchor=tk.W)
        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(input_frame, textvariable=self.input_dir, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="浏览", command=self.browse_input_dir).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 输出目录
        ttk.Label(file_frame, text="输出目录:").pack(anchor=tk.W, pady=(10, 0))
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(output_frame, textvariable=self.output_dir, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="浏览", command=self.browse_output_dir).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 标签文件输出目录
        ttk.Label(file_frame, text="标签输出目录:").pack(anchor=tk.W, pady=(10, 0))
        label_output_frame = ttk.Frame(file_frame)
        label_output_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(label_output_frame, textvariable=self.label_output_dir, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(label_output_frame, text="浏览", command=self.browse_label_output_dir).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 标签输出目录说明
        ttk.Label(file_frame, text="(可选，留空则与图像输出目录相同)", 
                 font=("Arial", 8), foreground="gray").pack(anchor=tk.W, pady=(2, 0))
        
        # 工具按钮区域
        tools_frame = ttk.LabelFrame(parent, text="工具", padding="10")
        tools_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(tools_frame, text="XML转YOLO", command=self.xml_to_yolo_conversion).pack(side=tk.LEFT, padx=5)
        ttk.Button(tools_frame, text="查看类别映射", command=self.view_class_mapping).pack(side=tk.LEFT, padx=5)
        
        # 增强选项区域
        options_frame = ttk.LabelFrame(parent, text="增强选项", padding="10")
        options_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建选项滚动区域
        canvas = tk.Canvas(options_frame)
        scrollbar = ttk.Scrollbar(options_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 创建复选框和参数控件
        self.create_option_widgets(scrollable_frame)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 全选/取消全选按钮
        select_frame = ttk.Frame(parent)
        select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(select_frame, text="全选", command=self.select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_frame, text="取消全选", command=self.deselect_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_frame, text="保存设置", command=self.save_settings).pack(side=tk.RIGHT, padx=5)
        
        # 进度和状态区域
        progress_frame = ttk.LabelFrame(parent, text="处理状态", padding="10")
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var, font=("Arial", 10))
        status_label.pack()
        
        # 操作按钮
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="开始处理", command=self.start_processing, 
                                     style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="预览效果", command=self.preview_effects).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.RIGHT, padx=5)
        
    def setup_right_panel(self, parent):
        """设置右侧预览面板"""
        # 预览标题
        preview_title = ttk.Label(parent, text="图像预览", font=("Arial", 14, "bold"))
        preview_title.pack(pady=(0, 10))
        
        # 预览区域
        self.preview_frame = ttk.LabelFrame(parent, text="原图", padding="10")
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.preview_label = ttk.Label(self.preview_frame, text="选择输入目录后显示预览")
        self.preview_label.pack(expand=True)
        
        # 效果预览区域
        self.effect_preview_frame = ttk.LabelFrame(parent, text="效果预览", padding="10")
        self.effect_preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.effect_preview_label = ttk.Label(self.effect_preview_frame, text="点击预览效果查看增强后的图像")
        self.effect_preview_label.pack(expand=True)
        
    def create_option_widgets(self, parent):
        """创建选项控件"""
        self.checkboxes = {}
        
        for key, config in self.augmentation_options.items():
            # 创建选项框架
            option_frame = ttk.Frame(parent)
            option_frame.pack(fill=tk.X, pady=2)
            
            # 复选框
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(option_frame, text=config["name"], variable=var, 
                               command=lambda k=key, v=var: self.on_checkbox_change(k, v))
            cb.pack(side=tk.LEFT)
            self.checkboxes[key] = var
            
            # 参数控件
            if config["params"]:
                param_frame = ttk.Frame(option_frame)
                param_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                
                self.param_widgets[key] = {}
                
                for param_name, default_value in config["params"].items():
                    param_label = ttk.Label(param_frame, text=f"{param_name}:")
                    param_label.pack(side=tk.LEFT, padx=(10, 2))
                    
                    if isinstance(default_value, (int, float)):
                        # 数值参数使用滑动条
                        var = tk.DoubleVar(value=default_value)
                        scale = ttk.Scale(param_frame, from_=0.1, to=5.0, variable=var, 
                                        orient=tk.HORIZONTAL, length=100)
                        scale.pack(side=tk.LEFT, padx=2)
                        
                        # 显示数值
                        value_label = ttk.Label(param_frame, textvariable=var)
                        value_label.pack(side=tk.LEFT, padx=2)
                        
                        self.param_widgets[key][param_name] = var
                    else:
                        # 其他类型参数使用输入框
                        var = tk.StringVar(value=str(default_value))
                        entry = ttk.Entry(param_frame, textvariable=var, width=8)
                        entry.pack(side=tk.LEFT, padx=2)
                        self.param_widgets[key][param_name] = var
                        
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
        
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                try:
                    image_path = os.path.join(input_dir, file)
                    self.current_image_path = image_path
                    self.current_image = Image.open(image_path)
                    
                    # 调整图片大小以适应预览区域
                    preview_img = self.current_image.copy()
                    preview_img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(preview_img)
                    
                    self.preview_label.configure(image=photo, text="")
                    self.preview_label.image = photo
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
        
    def preview_effects(self):
        """预览增强效果"""
        if not self.current_image or not self.selected_augmentations:
            messagebox.showwarning("警告", "请先选择图像和增强选项")
            return
            
        # 选择第一个增强选项进行预览
        aug_type = self.selected_augmentations[0]
        preview_img = self.apply_augmentation(self.current_image, aug_type)
        
        if preview_img:
            # 调整大小并显示
            preview_img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(preview_img)
            
            self.effect_preview_label.configure(image=photo, text="")
            self.effect_preview_label.image = photo
            
    def apply_augmentation(self, image, aug_type):
        """应用单个增强效果"""
        try:
            if aug_type == "original":
                return image.copy()
            elif aug_type == "flip_horizontal":
                return image.transpose(Image.FLIP_LEFT_RIGHT)
            elif aug_type == "flip_vertical":
                return image.transpose(Image.FLIP_TOP_BOTTOM)
            elif aug_type == "rotate_90":
                return image.transpose(Image.ROTATE_90)
            elif aug_type == "rotate_180":
                return image.transpose(Image.ROTATE_180)
            elif aug_type == "rotate_270":
                return image.transpose(Image.ROTATE_270)
            elif aug_type == "brightness_up":
                factor = self.param_widgets.get(aug_type, {}).get("factor", 1.3).get()
                enhancer = ImageEnhance.Brightness(image)
                return enhancer.enhance(factor)
            elif aug_type == "brightness_down":
                factor = self.param_widgets.get(aug_type, {}).get("factor", 0.7).get()
                enhancer = ImageEnhance.Brightness(image)
                return enhancer.enhance(factor)
            elif aug_type == "contrast_up":
                factor = self.param_widgets.get(aug_type, {}).get("factor", 1.5).get()
                enhancer = ImageEnhance.Contrast(image)
                return enhancer.enhance(factor)
            elif aug_type == "contrast_down":
                factor = self.param_widgets.get(aug_type, {}).get("factor", 0.7).get()
                enhancer = ImageEnhance.Contrast(image)
                return enhancer.enhance(factor)
            elif aug_type == "saturation_up":
                factor = self.param_widgets.get(aug_type, {}).get("factor", 1.5).get()
                enhancer = ImageEnhance.Color(image)
                return enhancer.enhance(factor)
            elif aug_type == "saturation_down":
                factor = self.param_widgets.get(aug_type, {}).get("factor", 0.7).get()
                enhancer = ImageEnhance.Color(image)
                return enhancer.enhance(factor)
            elif aug_type == "blur":
                radius = self.param_widgets.get(aug_type, {}).get("radius", 2).get()
                return image.filter(ImageFilter.GaussianBlur(radius))
            elif aug_type == "sharpen":
                factor = self.param_widgets.get(aug_type, {}).get("factor", 2.0).get()
                enhancer = ImageEnhance.Sharpness(image)
                return enhancer.enhance(factor)
            elif aug_type == "noise":
                factor = self.param_widgets.get(aug_type, {}).get("factor", 0.1).get()
                img_array = np.array(image)
                noise = np.random.normal(0, factor * 255, img_array.shape)
                noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                return Image.fromarray(noisy_img)
            elif aug_type == "edge_enhance":
                factor = self.param_widgets.get(aug_type, {}).get("factor", 2.0).get()
                enhancer = ImageEnhance.Sharpness(image)
                return enhancer.enhance(factor)
            elif aug_type == "emboss":
                return image.filter(ImageFilter.EMBOSS)
            elif aug_type == "find_edges":
                return image.filter(ImageFilter.FIND_EDGES)
            else:
                return image.copy()
        except Exception as e:
            print(f"应用增强效果失败: {e}")
            return image.copy()
            
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
        
    def save_settings(self):
        """保存设置到文件"""
        settings = {
            "input_dir": self.input_dir.get(),
            "output_dir": self.output_dir.get(),
            "label_output_dir": self.label_output_dir.get(),
            "selected_augmentations": self.selected_augmentations,
            "parameters": {}
        }
        
        # 保存参数值
        for aug_type, params in self.param_widgets.items():
            settings["parameters"][aug_type] = {}
            for param_name, widget in params.items():
                settings["parameters"][aug_type][param_name] = widget.get()
        
        try:
            with open("augmentation_settings.json", "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("成功", "设置已保存")
        except Exception as e:
            messagebox.showerror("错误", f"保存设置失败：{e}")
            
    def load_settings(self):
        """从文件加载设置"""
        try:
            if os.path.exists("augmentation_settings.json"):
                with open("augmentation_settings.json", "r", encoding="utf-8") as f:
                    settings = json.load(f)
                
                # 恢复目录设置
                if "input_dir" in settings:
                    self.input_dir.set(settings["input_dir"])
                if "output_dir" in settings:
                    self.output_dir.set(settings["output_dir"])
                if "label_output_dir" in settings:
                    self.label_output_dir.set(settings["label_output_dir"])
                
                # 恢复增强选项
                if "selected_augmentations" in settings:
                    for aug_type in settings["selected_augmentations"]:
                        if aug_type in self.checkboxes:
                            self.checkboxes[aug_type].set(True)
                            if aug_type not in self.selected_augmentations:
                                self.selected_augmentations.append(aug_type)
                
                # 恢复参数值
                if "parameters" in settings:
                    for aug_type, params in settings["parameters"].items():
                        if aug_type in self.param_widgets:
                            for param_name, value in params.items():
                                if param_name in self.param_widgets[aug_type]:
                                    self.param_widgets[aug_type][param_name].set(value)
                                    
        except Exception as e:
            print(f"加载设置失败: {e}")


def main():
    """主函数"""
    root = tk.Tk()
    app = AdvancedImageAugmentationGUI(root)
    
    # 设置窗口图标（如果有的话）
    try:
        root.iconbitmap("icon.ico")
    except:
        pass
        
    root.mainloop()


if __name__ == "__main__":
    main()
