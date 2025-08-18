import multiprocessing
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from PIL import Image, ImageTk
import cv2
import mss
import numpy as np
import os
import glob

from detection.detector import detect_region
from utils.app_window_utils import list_all_visible_apps, get_app_window_region, divide_region
from config import MODELS_FOLDER, DEFAULT_MODEL, SUPPORTED_MODEL_EXTENSIONS


class AppYOLOMultiRegionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 App 多区域 YOLO 实时检测")
        self.root.geometry("1200x800")

        self.is_detecting = False
        self.detection_thread = None
        self.processes = []
        self.shared_list = None
        self.manager = None
        self.image_queue = queue.Queue(maxsize=5)  # 最多保存 5 帧图像
        self.app_region = None
        self.region_divisions = []
        self.mode = None
        self.selected_model_path = DEFAULT_MODEL
        self.setup_ui()
        self.frame_stack = []  # 每一项是一个元组 (frame, detections)，或者更复杂的对象
        self.max_stack_size = 5  # 最多保存 5 帧z

    def setup_ui(self):
        # --- App 选择区 ---
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10, padx=10, fill="x")

        tk.Label(input_frame, text="📱 输入要监控的 App 窗口标题关键字：").pack(anchor="w")
        self.app_entry = tk.Entry(input_frame, width=30)
        self.app_entry.pack(anchor="w", pady=5)
        self.app_entry.insert(0, "Chrome")

        # --- 模型选择区和App列表区（同一行） ---
        control_frame = tk.Frame(input_frame)
        control_frame.pack(pady=10, fill="x")
        
        # 左侧：模型选择区（缩小版）
        model_frame = tk.Frame(control_frame)
        model_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 20))
        
        tk.Label(model_frame, text="🤖 选择检测模型：", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # 模型路径显示（缩小宽度）
        self.model_path_var = tk.StringVar(value=self.selected_model_path)
        self.model_path_entry = tk.Entry(model_frame, textvariable=self.model_path_var, width=35, state="readonly")
        self.model_path_entry.pack(anchor="w", pady=2)
        
        # 模型控制按钮（水平排列）
        model_btn_frame = tk.Frame(model_frame)
        model_btn_frame.pack(anchor="w", pady=2)
        
        browse_btn = tk.Button(model_btn_frame, text="📁 浏览", command=self.browse_model_file, bg="lightblue", width=8)
        browse_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        refresh_btn = tk.Button(model_btn_frame, text="🔄 刷新", command=self.refresh_model_list, bg="lightgreen", width=8)
        refresh_btn.pack(side=tk.LEFT)
        
        # 模型信息显示（缩小字体）
        self.model_info_label = tk.Label(model_frame, text="", fg="blue", font=("Arial", 9))
        self.model_info_label.pack(anchor="w", pady=2)
        
        # 模型列表显示（缩小高度和宽度）
        list_frame = tk.Frame(model_frame)
        list_frame.pack(anchor="w", pady=2, fill="x")
        
        tk.Label(list_frame, text="📋 本地模型：", font=("Arial", 9)).pack(anchor="w")
        
        # 创建模型列表框架
        list_container = tk.Frame(list_frame)
        list_container.pack(anchor="w", fill="x")
        
        # 模型列表（使用Listbox，缩小尺寸）
        self.model_listbox = tk.Listbox(list_container, height=3, width=40)
        self.model_listbox.pack(side=tk.LEFT, fill="x", expand=True)
        self.model_listbox.bind('<<ListboxSelect>>', self.on_model_listbox_select)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.model_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_listbox.configure(yscrollcommand=scrollbar.set)
        
        # 右侧：App列表区
        app_frame = tk.Frame(control_frame)
        app_frame.pack(side=tk.RIGHT, fill="x", expand=True)
        
        tk.Label(app_frame, text="🔍 当前可检测的 App 列表：", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # App列表按钮
        list_btn = tk.Button(app_frame, text="📋 列出所有可见 App", command=self.list_visible_apps, bg="lightyellow", width=20)
        list_btn.pack(anchor="w", pady=2)

        # --- 控制按钮 ---
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.start_btn = tk.Button(btn_frame, text="▶️ 开始检测", command=self.start_detection)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="⏹️ 停止检测", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # --- Canvas 显示区域 ---
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(pady=10, padx=10, fill="both", expand=True)

        tk.Label(canvas_frame, text="🖥️ 实时检测画面（App 窗口 + 检测框）：").pack(anchor="w")
        self.canvas = tk.Canvas(canvas_frame, bg="black", height=400)
        self.canvas.pack(fill="both", expand=True)

        # --- 检测结果文本框 ---
        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=10, padx=10, fill="both", expand=True)

        tk.Label(result_frame, text="📋 实时检测结果（类别 + 置信度 + 区域）：").pack(anchor="w")
        self.result_text = tk.Text(result_frame, height=12)
        self.result_text.pack(fill="both", expand=True)
        
        # 初始化模型列表
        self.refresh_model_list()

    def browse_model_file(self):
        """浏览并选择模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[
                ("模型文件", "*.pt *.pth *.onnx *.engine"),
                ("PyTorch模型", "*.pt *.pth"),
                ("ONNX模型", "*.onnx"),
                ("TensorRT模型", "*.engine"),
                ("所有文件", "*.*")
            ],
            initialdir=os.getcwd()
        )
        
        if file_path:
            self.selected_model_path = file_path
            self.model_path_var.set(file_path)
            self.update_model_info()
            self.log(f"🤖 已选择模型：{os.path.basename(file_path)}")
            
    def refresh_model_list(self):
        """刷新本地模型列表"""
        self.model_listbox.delete(0, tk.END)
        
        # 搜索当前目录和models子目录中的模型文件
        model_files = []
        
        # 搜索当前目录
        for ext in SUPPORTED_MODEL_EXTENSIONS:
            model_files.extend(glob.glob(f"*{ext}"))
            
        # 搜索models子目录
        if os.path.exists(MODELS_FOLDER):
            for ext in SUPPORTED_MODEL_EXTENSIONS:
                model_files.extend(glob.glob(os.path.join(MODELS_FOLDER, f"*{ext}")))
                
        # 去重并排序
        model_files = sorted(list(set(model_files)))
        
        if model_files:
            for model_file in model_files:
                # 显示相对路径，确保路径正确
                if model_file.startswith(MODELS_FOLDER + os.sep):
                    # 对于models子目录中的文件，显示完整路径
                    display_name = f"📁 {model_file}"
                else:
                    # 对于根目录中的文件，显示文件名
                    display_name = f"📄 {model_file}"
                self.model_listbox.insert(tk.END, display_name)
        else:
            self.model_listbox.insert(tk.END, "❌ 未找到模型文件")
            
        self.update_model_info()
        
    def on_model_listbox_select(self, event):
        """当用户在模型列表中选择模型时"""
        selection = self.model_listbox.curselection()
        if selection:
            index = selection[0]
            model_item = self.model_listbox.get(index)
            
            # 提取模型文件路径
            if model_item.startswith("📁 "):
                # 找到第一个空格的位置，去掉前缀
                space_index = model_item.find(" ")
                if space_index != -1:
                    model_path = model_item[space_index + 1:]
                else:
                    model_path = model_item[2:]  # 备用方案
            elif model_item.startswith("📄 "):
                # 找到第一个空格的位置，去掉前缀
                space_index = model_item.find(" ")
                if space_index != -1:
                    model_path = model_item[space_index + 1:]
                else:
                    model_path = model_item[2:]  # 备用方案
            else:
                return
                
            self.selected_model_path = model_path
            self.model_path_var.set(model_path)
            self.update_model_info()
            self.log(f"🤖 已选择模型：{os.path.basename(model_path)}")
            
    def update_model_info(self):
        """更新模型信息显示"""
        if self.selected_model_path and os.path.exists(self.selected_model_path):
            file_size = os.path.getsize(self.selected_model_path)
            file_size_mb = file_size / (1024 * 1024)
            self.model_info_label.config(
                text=f"✅ {os.path.basename(self.selected_model_path)} ({file_size_mb:.1f}MB)", 
                fg="green"
            )
        else:
            self.model_info_label.config(text="⚠️ 模型文件不存在", fg="orange")
            
    def log(self, msg):
        self.result_text.insert(tk.END, f"{msg}\n")
        self.result_text.see(tk.END)
        self.root.update()

    def list_visible_apps(self):
        apps = list_all_visible_apps()
        if not apps:
            messagebox.showinfo("提示", "❌ 当前没有检测到可展示的 App 窗口。请打开任意应用后重试。")
            return

        # 弹出一个新窗口来显示可复制的 App 列表
        list_window = tk.Toplevel(self.root)
        list_window.title("📋 可检测的 App 列表（可复制）")
        list_window.geometry("400x500")

        tk.Label(list_window, text="🖥️ 当前可检测的 App 窗口列表（可鼠标选中复制）：", font=("Arial", 12, "bold")).pack(
            pady=10)

        # 创建一个 Text 控件，支持选中 & 复制
        text_widget = tk.Text(list_window, wrap=tk.WORD, height=20, width=50)
        text_widget.pack(padx=10, pady=10, fill="both", expand=True)

        # 插入 App 列表内容
        app_text = "🖥️ 当前可检测的 App 窗口有：\n\n"
        for i, app in enumerate(apps[:50]):  # 最多显示 50 个，避免太长
            app_text += f"{i + 1}. {app}\n"
        if len(apps) > 50:
            app_text += f"\n... 还有 {len(apps) - 50} 个 App 未显示"
        text_widget.insert(tk.END, app_text)

        # 设置为只读（可选，防止误修改）
        text_widget.config(state=tk.DISABLED)

        # 可选：加一个复制全部按钮
        def copy_all():
            text_widget.config(state=tk.NORMAL)
            text_widget.tag_add(tk.SEL, "1.0", tk.END)
            text_widget.mark_set(tk.INSERT, "1.0")
            text_widget.event_generate("<<Copy>>")
            text_widget.config(state=tk.DISABLED)
            self.log("📋 已复制全部 App 列表到剪贴板")

        copy_btn = tk.Button(list_window, text="📋 复制全部", command=copy_all)
        copy_btn.pack(pady=5)

        # 让文本可选中（如果不想设为只读，直接删掉上面两行 config(state=DISABLED) 即可）

    def show_app_selection_buttons(self):
        # 先清空已有的 App 按钮区域（避免重复添加）
        if hasattr(self, 'app_button_frame'):
            self.app_button_frame.destroy()

        # 创建一个 Frame 用来放置 App 按钮
        self.app_button_frame = tk.Frame(self.root)
        self.app_button_frame.pack(pady=10, padx=10, fill="x")

        tk.Label(self.app_button_frame, text="📋 点击选择要监控的 App：").pack(anchor="w")

        # 获取当前所有可见的 App
        try:
            apps = list_all_visible_apps()
        except Exception as e:
            messagebox.showerror("错误", f"获取 App 列表失败：{e}")
            return

        if not apps:
            tk.Label(self.app_button_frame, text="❌ 当前没有检测到可展示的 App 窗口。请打开一个应用后重试。",
                     fg="red").pack()
            return

        # 每个 App 生成一个 Button
        for app in apps[:20]:  # 只显示前 20 个，避免太多按钮
            btn = tk.Button(
                self.app_button_frame,
                text=app,
                command=lambda a=app: self.on_app_selected(a),  # 注意 lambda 传参技巧
                width=40,
                anchor="w"
            )
            btn.pack(anchor="w", pady=1)  # 每个按钮占一行，左对齐

    def on_app_selected(self, app_title):
        print(f"[INFO] 用户选择了 App：{app_title}")
        self.app_entry.delete(0, tk.END)
        self.app_entry.insert(0, app_title)  # 将选中的 App 名称填入输入框
        self.log(f"✅ 已选择 App（通过按钮）：{app_title}")

    def get_app_region(self):
        app_name = self.app_entry.get().strip()
        if not app_name:
            raise Exception("请输入 App 名称")
        self.app_region = get_app_window_region(app_name)
        self.region_divisions = divide_region(self.app_region)
        self.log(f"✅ 已选择 App：{self.app_region['title']} | 区域：{self.app_region}")

    def start_detection(self):
        try:
            # 检查选择的模型文件是否存在
            if not self.selected_model_path:
                raise Exception("请选择模型文件")
            
            if not os.path.exists(self.selected_model_path):
                raise Exception(f"模型文件不存在：{self.selected_model_path}")
            
            self.get_app_region()
            self.is_detecting = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)

            self.manager = multiprocessing.Manager()
            self.shared_list = [self.manager.list() for _ in range(4)]  # 所有子进程写入的检测结果

            # 启动 4 个子进程，每个负责一个区域，传递选择的模型路径
            i = 0
            for region in self.region_divisions:
                p = multiprocessing.Process(target=detect_region, args=(region, self.shared_list, i, self.selected_model_path))
                i += 1
                p.daemon = True
                p.start()
                self.processes.append(p)

            # 启动检测结果显示线程
            self.detection_thread = threading.Thread(target=self.detection_display_loop, daemon=True)
            self.detection_thread.start()

            self.log(f"🚀 检测已启动，使用模型：{os.path.basename(self.selected_model_path)}")

        except Exception as e:
            messagebox.showerror("错误", f"启动检测失败：{e}")
            self.stop_detection()

    def stop_detection(self):
        self.is_detecting = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        for p in self.processes:
            try:
                p.terminate()
            except:
                pass
        self.processes.clear()
        self.log("⏹️ 检测已停止")

    def detection_display_loop(self):
        sct = mss.mss()
        app_region = self.app_region
        if not app_region:
            self.log("❌ 未获取到 App 区域，无法显示图像")
            return

        monitor = {
            "left": app_region["left"],
            "top": app_region["top"],
            "width": app_region["width"],
            "height": app_region["height"]
        }

        try:
            while self.is_detecting:
                try:
                    # =====================
                    # 1. 截取当前最新屏幕图像（只截一次！）
                    # =====================
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # =====================
                    # 2. 绘制检测框（在当前帧上绘制！）
                    # =====================
                    for items in self.shared_list:  # 当前最新的检测结果
                        for item in items:
                            if item is not None:
                                x1, y1, x2, y2, conf, cls_name, region_id, region_box = item
                                abs_x1 = x1 + monitor["left"]
                                abs_y1 = y1 + monitor["top"]
                                abs_x2 = x2 + monitor["left"]
                                abs_y2 = y2 + monitor["top"]
                                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                                label = f"Class {cls_name}: {conf:.2f}"
                                cv2.putText(frame, label, (abs_x1, abs_y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                    # =====================
                    # 3. 转换为 PIL -> PhotoImage
                    # =====================
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    canvas_width = self.canvas.winfo_width()
                    canvas_height = self.canvas.winfo_height()

                    if canvas_width > 1 and canvas_height > 1:
                        img_ratio = min(canvas_width / pil_image.width, canvas_height / pil_image.height)
                        new_width = int(pil_image.width * img_ratio)
                        new_height = int(pil_image.height * img_ratio)
                        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    tk_image = ImageTk.PhotoImage(pil_image)

                    # =====================
                    # 4. 清除上一帧，显示最新图像
                    # =====================
                    x_offset = (canvas_width - pil_image.width) // 2
                    y_offset = (canvas_height - pil_image.height) // 2
                    self.canvas.create_image(x_offset, y_offset, anchor="nw", image=tk_image)
                    self.canvas.image = tk_image  # 必须保留引用，否则图像消失

                except Exception as e:
                    self.log(f"渲染出错：{e}")
        except Exception as e:
            self.log(f"主循环出错：{e}")
        finally:
            sct.close()


