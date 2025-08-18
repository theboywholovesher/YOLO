import mss
import mss.tools
import time
import os
from datetime import datetime, UTC
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
from PIL import Image


class MediaConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("媒体转图片工具（截图+视频）")
        self.root.geometry("800x600")  # 调整窗口大小适应新组件

        # 初始化变量
        self.is_running = False  # 截图任务运行状态
        self.video_converting = False  # 视频转换运行状态
        self.screenshot_thread = None
        self.video_thread = None

        # 支持的图片格式
        self.supported_formats = ['PNG', 'JPG', 'BMP', 'WEBP']

        # 创建界面组件
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="15")  # 减小整体内边距
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ---------------------- 截图功能区域（横向排列） ----------------------
        ttk.Label(main_frame, text="=== 自动截图设置 ===", font=('微软雅黑', 10, 'bold')).grid(row=0, column=0,
                                                                                               columnspan=4,
                                                                                               sticky=tk.W,
                                                                                               pady=(0, 10))  # 顶部标题

        # 截图间隔（标签+输入框同一行）
        ttk.Label(main_frame, text="截图间隔（秒）:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.interval_var = tk.StringVar(value="10")
        self.interval_entry = ttk.Entry(main_frame, textvariable=self.interval_var, width=8)
        self.interval_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=3)

        # 截图保存目录（标签+输入框同一行）
        ttk.Label(main_frame, text="截图保存目录:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.save_dir_var = tk.StringVar()
        self.save_dir_entry = ttk.Entry(main_frame, textvariable=self.save_dir_var, width=20)
        self.save_dir_entry.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=3)
        ttk.Button(main_frame, text="选择目录", command=self.select_screenshot_directory, width=8).grid(row=2, column=3,
                                                                                                        sticky=tk.W,
                                                                                                        padx=5,
                                                                                                        pady=3)

        # 图片格式（标签+下拉框同一行）
        ttk.Label(main_frame, text="图片格式:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=3)
        self.format_var = tk.StringVar(value="PNG")
        self.format_combobox = ttk.Combobox(main_frame, textvariable=self.format_var, values=self.supported_formats,
                                            state="readonly", width=8)
        self.format_combobox.grid(row=3, column=1, sticky=tk.W, padx=5, pady=3)

        # 开始截图按钮
        self.screenshot_btn = ttk.Button(main_frame, text="开始截图", command=self.start_screenshot, width=12)
        self.screenshot_btn.grid(row=4, column=0, columnspan=4, pady=10, sticky=tk.EW)

        # 停止截图按钮（新增 ✅）
        self.stop_screenshot_btn = ttk.Button(main_frame, text="停止截图", command=self.stop_screenshot, width=12)
        self.stop_screenshot_btn.grid(row=5, column=0, columnspan=4, pady=5, sticky=tk.EW)

        # ---------------------- 视频转图片区域（横向排列） ----------------------
        ttk.Label(main_frame, text="=== 视频转图片设置 ===", font=('微软雅黑', 10, 'bold')).grid(row=6, column=0,
                                                                                                 columnspan=4,
                                                                                                 sticky=tk.W,
                                                                                                 pady=(10, 10))  # 分隔标题

        # 视频文件（标签+输入框同一行）
        ttk.Label(main_frame, text="视频文件:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=3)
        self.video_path_var = tk.StringVar()
        self.video_path_entry = ttk.Entry(main_frame, textvariable=self.video_path_var, width=20)  # 缩小宽度
        self.video_path_entry.grid(row=7, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=3)  # 跨2列扩展
        ttk.Button(main_frame, text="选择视频", command=self.select_video, width=8).grid(row=6, column=3, sticky=tk.W,
                                                                                         padx=5, pady=3)  # 按钮靠左

        # 视频保存目录（标签+输入框同一行）
        ttk.Label(main_frame, text="视频保存目录:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=3)
        self.video_save_dir_var = tk.StringVar()
        self.video_save_dir_entry = ttk.Entry(main_frame, textvariable=self.video_save_dir_var, width=20)  # 缩小宽度
        self.video_save_dir_entry.grid(row=8, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=3)  # 跨2列扩展
        ttk.Button(main_frame, text="选择目录", command=self.select_video_directory, width=8).grid(row=7, column=3,
                                                                                                   sticky=tk.W, padx=5,
                                                                                                   pady=3)  # 按钮靠左

        # 切分间隔（标签+输入框同一行）
        ttk.Label(main_frame, text="切分间隔（秒）:").grid(row=9, column=0, sticky=tk.W, padx=5, pady=3)
        self.split_interval_var = tk.StringVar(value="1")
        self.split_interval_entry = ttk.Entry(main_frame, textvariable=self.split_interval_var, width=8)  # 缩小宽度
        self.split_interval_entry.grid(row=8, column=1, sticky=tk.W, padx=5, pady=3)  # 靠左对齐

        # 起始时间（标签+输入框同一行）
        ttk.Label(main_frame, text="起始时间（秒）:").grid(row=10, column=0, sticky=tk.W, padx=5, pady=3)
        self.start_time_var = tk.StringVar(value="0")
        self.start_time_entry = ttk.Entry(main_frame, textvariable=self.start_time_var, width=8)  # 缩小宽度
        self.start_time_entry.grid(row=9, column=1, sticky=tk.W, padx=5, pady=3)  # 靠左对齐

        # 结束时间（标签+输入框同一行）
        ttk.Label(main_frame, text="结束时间（秒）:").grid(row=11, column=0, sticky=tk.W, padx=5, pady=3)
        self.end_time_var = tk.StringVar(value="0")
        self.end_time_entry = ttk.Entry(main_frame, textvariable=self.end_time_var, width=8)  # 缩小宽度
        self.end_time_entry.grid(row=10, column=1, sticky=tk.W, padx=5, pady=3)  # 靠左对齐

        # 视频转图片按钮（单独一行，跨列居中）
        self.convert_btn = ttk.Button(main_frame, text="开始转换视频", command=self.start_video_conversion, width=15)
        self.convert_btn.grid(row=12, column=0, columnspan=4, pady=10, sticky=tk.EW)  # 跨4列居中

        # ---------------------- 日志显示区域（自适应高度） ----------------------
        self.log_text = tk.Text(main_frame, height=8, state=tk.DISABLED)  # 减小高度
        self.log_text.grid(row=13, column=0, columnspan=4, sticky=tk.NSEW, pady=(10, 0))  # 顶部间距
        main_frame.grid_columnconfigure(1, weight=1)  # 列1（输入框主列）可扩展
        main_frame.grid_rowconfigure(14, weight=1)  # 日志行可扩展

    def select_screenshot_directory(self):
        dir_path = filedialog.askdirectory(title="选择截图保存目录")
        if dir_path:
            self.save_dir_var.set(dir_path)

    def select_video_directory(self):
        dir_path = filedialog.askdirectory(title="选择视频转图片保存目录")
        if dir_path:
            self.video_save_dir_var.set(dir_path)

    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv"), ("所有文件", "*.*")]
        )
        if file_path:
            self.video_path_var.set(file_path)
            video_dir = os.path.dirname(file_path)
            self.video_save_dir_var.set(video_dir)

    def log_message(self, message):
        self.log_text.configure(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def show_error(self, title, message):
        messagebox.showerror(title, message)

    def validate_inputs(self, is_video=False):
        try:
            save_format = self.format_var.get()
            if save_format not in self.supported_formats:
                raise ValueError(f"不支持的图片格式：{save_format}")

            if not is_video:
                save_dir = self.save_dir_var.get()
                if not save_dir:
                    raise ValueError("请选择截图保存目录")

            if is_video:
                split_interval = float(self.split_interval_var.get())
                if split_interval <= 0:
                    raise ValueError("切分间隔必须大于0秒")

                start_time = float(self.start_time_var.get())
                if start_time < 0:
                    raise ValueError("起始时间不能小于0秒")

                end_time = float(self.end_time_var.get())
                if end_time < start_time:
                    raise ValueError("结束时间必须大于等于起始时间")

                video_save_dir = self.video_save_dir_var.get()
                if video_save_dir and not os.path.isdir(video_save_dir):
                    raise ValueError("视频保存目录不存在")

        except ValueError as e:
            self.show_error("输入错误", str(e))
            return False
        return True

    def capture_screenshot(self):
        while self.is_running:
            try:
                interval = int(self.interval_var.get())
                save_dir = self.save_dir_var.get()
                save_format = self.format_var.get().lower()

                os.makedirs(save_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"screenshot_{timestamp}.{save_format}"
                filepath = os.path.join(save_dir, filename)

                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    screenshot = sct.grab(monitor)

                    img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
                    img.save(filepath, format=save_format.upper())

                self.log_message(f"截图保存成功：{filename}（格式：{save_format.upper()}，间隔{interval}秒）")

                start_time = time.time()
                while time.time() - start_time < interval:
                    time.sleep(0.1)

            except Exception as e:
                self.log_message(f"截图失败：{str(e)}")
                time.sleep(5)
                break

    def start_screenshot(self):
        if self.is_running:
            return

        if not self.validate_inputs(is_video=False):
            return

        self.is_running = True
        self.screenshot_thread = threading.Thread(target=self.capture_screenshot, daemon=True)
        self.screenshot_thread.start()

        self.log_message(f"开始自动截图（格式：{self.format_var.get()}）...")

    def stop_screenshot(self):
        if not self.is_running:
            return

        self.is_running = False
        if self.screenshot_thread and self.screenshot_thread.is_alive():
            self.screenshot_thread.join(timeout=2)

        self.log_message("截图任务已停止")

    def convert_video_to_images(self):
        if not self.validate_inputs(is_video=True):
            return

        try:
            video_path = self.video_path_var.get()
            video_save_dir = self.video_save_dir_var.get() or os.path.dirname(video_path)
            save_format = self.format_var.get().lower()
            start_time = float(self.start_time_var.get())
            end_time = float(self.end_time_var.get()) or float('inf')
            split_interval = float(self.split_interval_var.get())

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("无法打开视频文件，请检查文件格式")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_duration = total_frames / fps
            self.log_message(f"视频信息：分辨率 {frame_width}x{frame_height}，帧率 {fps:.1f}fps，总时长 {video_duration:.1f}秒")

            end_time = min(end_time, video_duration)
            self.log_message(f"实际处理时间范围：{start_time:.1f}秒 ~ {end_time:.1f}秒")

            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            current_frame = start_frame

            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            self.log_message(f"从第 {start_frame} 帧开始处理（对应时间：{start_time:.1f}秒）")

            os.makedirs(video_save_dir, exist_ok=True)
            video_basename = os.path.splitext(os.path.basename(video_path))[0]

            saved_count = 0
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    self.log_message("警告：读取帧失败，可能到达视频末尾")
                    break

                current_time_sec = current_frame / fps
                dt = datetime.fromtimestamp(current_time_sec, UTC)
                current_time_str = f"{dt.hour:02d}-{dt.minute:02d}-{dt.second:02d}"

                filename = f"{video_basename}_{current_time_str}.{save_format}"
                filepath = os.path.join(video_save_dir, filename)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_image.save(filepath, format=save_format.upper())
                saved_count += 1
                self.log_message(f"保存成功：{filename}（时间：{current_time_str}）")

                current_time_sec += split_interval
                current_frame = min(int(current_time_sec * fps), end_frame)

            cap.release()
            self.log_message(f"视频转换完成！共保存 {saved_count} 张（间隔 {split_interval} 秒）")
            messagebox.showinfo("完成", f"成功保存 {saved_count} 张图片到：{video_save_dir}")

        except Exception as e:
            self.log_message(f"转换失败：{str(e)}")
            messagebox.showerror("错误", f"转换失败：{str(e)}")

    def start_video_conversion(self):
        if self.video_converting:
            return

        if not self.validate_inputs(is_video=True):
            return

        video_path = self.video_path_var.get()
        if not os.path.isfile(video_path):
            self.show_error("文件不存在", "请选择有效的视频文件")
            return

        self.video_converting = True
        self.video_thread = threading.Thread(target=self.convert_video_to_images, daemon=True)
        self.video_thread.start()

        self.log_message("开始视频转图片任务...")

    def on_close(self):
        self.stop_screenshot()
        self.video_converting = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MediaConverterGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()