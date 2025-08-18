import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class YOLOCropAnnotateTool:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 标注 & 图片裁剪工具（左右分区版）")
        self.root.geometry("1000x600")

        # 图片相关
        self.image_files = []
        self.current_index = 0
        self.original_image = None
        self.current_display_image = None
        self.tk_image = None

        # 显示区域
        self.display_width = 600
        self.display_height = 400

        # 点击相关
        self.click_points = []
        self.point_ids = []

        # 模式
        self.mode = "crop"  # 'crop' 或 'yolo'
        self.current_class_id = 0

        # 保存路径
        self.crop_save_dir = None
        self.yolo_save_dir = None

        self.setup_ui()

    def setup_ui(self):
        # ==================== 图片显示区域（居中）====================
        self.canvas = tk.Canvas(self.root, bg="gray90", highlightthickness=0)
        self.canvas.place(x=200, y=50, width=self.display_width, height=self.display_height)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # ==================== 左侧区域：裁剪功能控件（x=10 ~ 180, y=50~550）====================
        left_x = 10
        left_y = 50

        tk.Label(self.root, text="【裁剪功能】", fg="blue").place(x=left_x, y=left_y)

        tk.Button(self.root, text="📁 加载图片文件夹", command=self.load_image_folder).place(x=left_x, y=left_y + 30)
        self.btn_prev = tk.Button(self.root, text="⬅️ 上一张", command=self.prev_image, state=tk.DISABLED)
        self.btn_prev.place(x=left_x, y=left_y + 70)
        self.btn_next = tk.Button(self.root, text="下一张 ➡️", command=self.next_image, state=tk.DISABLED)
        self.btn_next.place(x=left_x, y=left_y + 110)

        tk.Button(self.root, text="🔄 恢复初始状态", command=self.reset_state).place(x=left_x, y=left_y + 150)

        tk.Button(self.root, text="裁剪保存目录:选择", command=lambda: self.select_path("crop")).place(x=left_x,
                                                                                                       y=left_y + 200)

        self.btn_crop_action = tk.Button(self.root, text="✂️ 裁剪", command=self.perform_action)
        self.btn_crop_action.place(x=left_x, y=left_y + 250)

        # ==================== 右侧区域：YOLO标注功能控件（x=820 ~ 990, y=50~550）====================
        right_x = 820
        right_y = 50

        tk.Label(self.root, text="【YOLO 标注功能】", fg="green").place(x=right_x, y=right_y)

        tk.Button(self.root, text="📁 加载图片文件夹", command=self.load_image_folder).place(x=right_x, y=right_y + 30)
        self.btn_prev2 = tk.Button(self.root, text="⬅️ 上一张", command=self.prev_image, state=tk.DISABLED)
        self.btn_prev2.place(x=right_x, y=right_y + 70)
        self.btn_next2 = tk.Button(self.root, text="下一张 ➡️", command=self.next_image, state=tk.DISABLED)
        self.btn_next2.place(x=right_x, y=right_y + 110)

        tk.Button(self.root, text="🔄 恢复初始状态", command=self.reset_state).place(x=right_x, y=right_y + 150)

        tk.Label(self.root, text="Class ID:").place(x=right_x, y=right_y + 200)
        self.class_entry = tk.Entry(self.root, width=5)
        self.class_entry.place(x=right_x + 60, y=right_y + 200)
        self.class_entry.insert(0, "0")

        tk.Button(self.root, text="YOLO标签保存目录:选择", command=lambda: self.select_path("yolo")).place(x=right_x,
                                                                                                           y=right_y + 250)

        self.btn_yolo_action = tk.Button(self.root, text="🏷️ 生成 YOLO 标签", command=self.perform_action)
        self.btn_yolo_action.place(x=right_x, y=right_y + 300)

        # ===== 新增：保存当前图片按钮 & 格式选择（仅 YOLO 功能区）=====
        tk.Label(self.root, text="保存图片格式:").place(x=right_x, y=right_y + 340)
        self.yolo_image_format_var = tk.StringVar(value="png")  # 默认 png
        format_options = ["png", "jpg", "jpeg", "bmp"]
        self.yolo_image_format_menu = tk.OptionMenu(self.root, self.yolo_image_format_var, *format_options)
        self.yolo_image_format_menu.place(x=right_x + 100, y=right_y + 330)

        tk.Button(self.root, text="💾 保存当前图片", command=self.save_current_image).place(x=right_x, y=right_y + 380)

        # ===== 新增：标注文件格式选择（仅 YOLO 功能区）=====
        tk.Label(self.root, text="标注保存格式:").place(x=right_x, y=right_y + 420)
        self.yolo_anno_format_var = tk.StringVar(value="both")  # 默认：两者都保存
        format_options = ["txt", "xml", "both"]  # 用户可选择只存 txt / 只存 xml / 都存
        self.yolo_anno_format_menu = tk.OptionMenu(self.root, self.yolo_anno_format_var, *format_options)
        self.yolo_anno_format_menu.place(x=right_x + 100, y=right_y + 410)

        # ==================== 模式切换（可选，也可以用按钮）====================
        tk.Button(self.root, text="🔀 切换到 YOLO 标注", command=self.switch_to_yolo).place(x=400, y=20)
        tk.Button(self.root, text="🔀 切换到裁剪", command=self.switch_to_crop).place(x=500, y=20)

        # 当前模式显示
        self.mode_label = tk.Label(self.root, text="当前模式: 裁剪", fg="blue")
        self.mode_label.place(x=400, y=50)

        # 图片信息
        self.label_info = tk.Label(self.root, text="点击「加载图片文件夹」开始", fg="gray")
        self.label_info.place(x=400, y=75)

    def select_path(self, mode):
        if mode == "crop":
            self.crop_save_dir = filedialog.askdirectory(title="选择裁剪图片保存目录")
            if self.crop_save_dir:
                messagebox.showinfo("提示", f"裁剪图片将保存到：{self.crop_save_dir}")
        elif mode == "yolo":
            self.yolo_save_dir = filedialog.askdirectory(title="选择 YOLO 标签保存目录")
            if self.yolo_save_dir:
                messagebox.showinfo("提示", f"YOLO 标签将保存到：{self.yolo_save_dir}")

    def switch_to_crop(self):
        self.mode = "crop"
        self.mode_label.config(text="当前模式: 裁剪", fg="blue")
        self.class_entry.config(state=tk.DISABLED)

    def save_current_image(self):
        if not self.original_image:
            messagebox.showwarning("提示", "未加载图片！")
            return

        try:
            # 获取用户选择的格式，如 png, jpg
            save_format = self.yolo_image_format_var.get().lower()

            # 如果是 jpg/jpeg，PIL 要求格式名为 'jpeg'
            if save_format in ["jpg", "jpeg"]:
                save_format_pil = "jpeg"
                file_ext = "jpg"
            else:
                save_format_pil = save_format
                file_ext = save_format

            # 图片文件名：原文件名 + _marked + 序号（可选）+ .ext
            filename = os.path.basename(self.image_files[self.current_index])
            name_part = os.path.splitext(filename)[0]
            save_filename = f"{name_part}_marked.{file_ext}"
            save_path = os.path.join(self.yolo_save_dir, save_filename)

            # 如果你想保存到其他目录也可以改，比如 self.crop_save_dir 或新建一个目录

            # 将原图缩放为 600x400 后保存（即显示的图）
            disp_img = self.original_image.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
            disp_img.save(save_path, format=save_format_pil)

            messagebox.showinfo("保存成功", f"当前图片已保存为 {save_format.upper()} 格式：\n{save_path}")

        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存当前图片：{e}")

    def switch_to_yolo(self):
        self.mode = "yolo"
        self.mode_label.config(text="当前模式: YOLO 标注", fg="green")
        self.class_entry.config(state=tk.NORMAL)

    def load_image_folder(self):
        folder = filedialog.askdirectory(title="选择包含图片的文件夹")
        if not folder:
            return
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
        self.image_files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_ext)])
        if not self.image_files:
            messagebox.showinfo("提示", "未找到支持的图片文件！")
            return
        self.current_index = 0
        self.load_current_image()
        self.btn_prev.config(state=tk.NORMAL)
        self.btn_next.config(state=tk.NORMAL)
        self.btn_prev2.config(state=tk.NORMAL)
        self.btn_next2.config(state=tk.NORMAL)
        self.update_label_info()

    def load_current_image(self):
        if not self.image_files:
            return
        img_path = self.image_files[self.current_index]
        try:
            self.original_image = Image.open(img_path)
            self.display_image_on_canvas()
            self.reset_state()
            self.update_label_info()
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片：{e}")

    def display_image_on_canvas(self):
        if not self.original_image:
            return
        disp_img = self.original_image.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(disp_img)
        self.canvas.create_image(self.display_width // 2, self.display_height // 2, image=self.tk_image,
                                 anchor="center")
        self.canvas.image = self.tk_image

    def update_label_info(self):
        if self.original_image:
            w, h = self.original_image.size
            self.label_info.config(text=f"图片尺寸: {w}x{h} | 第 {self.current_index + 1}/{len(self.image_files)} 张")
        else:
            self.label_info.config(text="未加载图片")

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        self.click_points.append((x, y))
        self.draw_red_dot(x, y)
        if len(self.click_points) == 2:
            self.perform_action()

    def draw_red_dot(self, x, y):
        dot_id = self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="red", outline="")
        self.point_ids.append(dot_id)

    def reset_state(self):
        for dot_id in self.point_ids:
            self.canvas.delete(dot_id)
        self.point_ids = []
        self.click_points = []
        try:
            self.current_class_id = int(self.class_entry.get())
        except ValueError:
            self.current_class_id = 0

    def perform_action(self):
        if not self.original_image:
            messagebox.showwarning("提示", "请先加载图片！")
            return
        if len(self.click_points) != 2:
            messagebox.showwarning("提示", "请点击两个点！")
            return

        x1, y1 = self.click_points[0]
        x2, y2 = self.click_points[1]

        orig_w, orig_h = self.original_image.size
        disp_w, disp_h = self.display_width, self.display_height

        real_x1 = x1 * (orig_w / disp_w)
        real_y1 = y1 * (orig_h / disp_h)
        real_x2 = x2 * (orig_w / disp_w)
        real_y2 = y2 * (orig_h / disp_h)

        left = min(real_x1, real_x2)
        top = min(real_y1, real_y2)
        right = max(real_x1, real_x2)
        bottom = max(real_y1, real_y2)

        if self.mode == "crop":
            self.perform_crop(left, top, right, bottom)
        elif self.mode == "yolo":
            self.perform_yolo_annotation(left, top, right, bottom)

    def perform_crop(self, left, top, right, bottom):
        try:
            cropped = self.original_image.crop((left, top, right, bottom))
            if not self.crop_save_dir:
                messagebox.showwarning("提示", "请先选择裁剪图片保存目录！")
                return
            filename = os.path.basename(self.image_files[self.current_index])
            name_part = os.path.splitext(filename)[0]
            save_path = os.path.join(self.crop_save_dir, f"{name_part}_cropped.png")
            cropped.save(save_path)
            messagebox.showinfo("保存成功", f"裁剪图片已保存到：\n{save_path}")
        except Exception as e:
            messagebox.showerror("裁剪失败", f"无法裁剪图片：{e}")

    def generate_voc_xml(self, xml_path, image_filename, image_path, width, height, depth, objects):
        from xml.etree.ElementTree import Element, SubElement, tostring
        from xml.dom import minidom

        annotation = Element("annotation")

        folder = SubElement(annotation, "folder")
        folder.text = "images"

        filename = SubElement(annotation, "filename")
        filename.text = image_filename

        path = SubElement(annotation, "path")
        path.text = image_path

        source = SubElement(annotation, "source")
        database = SubElement(source, "database")
        database.text = "Unknown"

        size = SubElement(annotation, "size")
        width_elem = SubElement(size, "width")
        width_elem.text = str(width)
        height_elem = SubElement(size, "height")
        height_elem.text = str(height)
        depth_elem = SubElement(size, "depth")
        depth_elem.text = str(depth)

        segmented = SubElement(annotation, "segmented")
        segmented.text = "0"

        for obj in objects:
            object_elem = SubElement(annotation, "object")
            name = SubElement(object_elem, "name")
            name.text = obj["name"]

            pose = SubElement(object_elem, "pose")
            pose.text = "Unspecified"

            truncated = SubElement(object_elem, "truncated")
            truncated.text = "0"

            difficult = SubElement(object_elem, "difficult")
            difficult.text = "0"

            bndbox = SubElement(object_elem, "bndbox")
            xmin = SubElement(bndbox, "xmin")
            xmin.text = str(obj["xmin"])
            ymin = SubElement(bndbox, "ymin")
            ymin.text = str(obj["ymin"])
            xmax = SubElement(bndbox, "xmax")
            xmax.text = str(obj["xmax"])
            ymax = SubElement(bndbox, "ymax")
            ymax.text = str(obj["ymax"])

        xml_str = tostring(annotation, encoding="utf-8")
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

    def perform_yolo_annotation(self, left, top, right, bottom):
        try:
            if not self.yolo_save_dir:
                messagebox.showwarning("提示", "请先选择 YOLO 标签保存目录！")
                return

            orig_w, orig_h = self.original_image.size
            x_center = ((left + right) / 2) / orig_w
            y_center = ((top + bottom) / 2) / orig_h
            width = (right - left) / orig_w
            height = (bottom - top) / orig_h

            class_id = self.current_class_id

            txt_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

            filename = os.path.basename(self.image_files[self.current_index])
            name_part = os.path.splitext(filename)[0]
            txt_save_path = os.path.join(self.yolo_save_dir, f"{name_part}.txt")

            xml_save_path = os.path.join(self.yolo_save_dir, f"{name_part}.xml")

            # 获取用户选择的标注保存格式
            anno_format = self.yolo_anno_format_var.get()  # 'txt', 'xml', 'both'

            if anno_format in ["txt", "both"]:
                with open(txt_save_path, "a", encoding="utf-8") as f:
                    f.write(txt_line)
                print(f"✅ 已保存 YOLO 标签到：{txt_save_path}")

            if anno_format in ["xml", "both"]:
                self.generate_voc_xml(
                    xml_path=xml_save_path,
                    image_filename=filename,
                    image_path=os.path.join(os.getcwd(), self.image_files[self.current_index]),
                    width=orig_w,
                    height=orig_h,
                    depth=3,
                    objects=[{
                        "name": "object",
                        "xmin": int(left),
                        "ymin": int(top),
                        "xmax": int(right),
                        "ymax": int(bottom),
                    }]
                )
                print(f"✅ 已保存 VOC XML 到：{xml_save_path}")

            # 提示信息
            saved = []
            if anno_format in ["txt", "both"]:
                saved.append(f"YOLO 标签文件：{txt_save_path}")
            if anno_format in ["xml", "both"]:
                saved.append(f"VOC XML 文件：{xml_save_path}")
            messagebox.showinfo("保存成功", "\n".join(saved))

        except Exception as e:
            messagebox.showerror("标注失败", f"无法保存 YOLO 标签或 XML：{e}")

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOCropAnnotateTool(root)
    root.mainloop()
