import sys
import os
import threading
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QLineEdit, QFileDialog,
                             QTextEdit, QSpinBox, QMessageBox, QGroupBox, QCheckBox,
                             QListWidget)
from PyQt5.QtCore import Qt
import torch
from torch import nn

def fix_param_size(param):
    """将参数统一为二元元组 (h, w)"""
    if isinstance(param, int):
        return (param, param)  # 整数 → 二元元组
    elif len(param) >= 2:
        return tuple(param[:2])  # 截取前两个维度
    else:
        return (1, 1)  # 默认值
class YOLOv8TrainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 训练工具 (PyQt5 GUI)")
        self.setGeometry(100, 100, 800, 800)
        self.original_class_names = []

        # 主窗口部件
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # --- 数据集路径 ---
        self.add_section_title("1. 数据集路径设置")
        self.train_img_path = self.add_path_input("训练图片文件夹路径:")
        self.train_label_path = self.add_path_input("训练标签文件夹路径:")

        # --- 模型选择 ---
        self.add_section_title("2. 预训练模型选择")
        model_layout = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("选择本地 .pt 预训练模型文件，如 yolov8n.pt")
        self.btn_browse_model = QPushButton("浏览模型文件")
        self.btn_browse_model.clicked.connect(self.browse_model_file)
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(self.btn_browse_model)
        self.layout.addLayout(model_layout)

        # --- 模型类别信息展示 ---
        self.model_info_group = QGroupBox("模型类别信息")
        self.model_info_layout = QVBoxLayout()
        self.class_list_widget = QListWidget()
        self.class_list_widget.setMaximumHeight(100)
        self.model_info_layout.addWidget(QLabel("当前模型包含的类别:"))
        self.model_info_layout.addWidget(self.class_list_widget)
        self.model_info_group.setLayout(self.model_info_layout)
        self.layout.addWidget(self.model_info_group)

        # --- 新标签训练选项 ---
        self.add_label_group = QGroupBox("新标签训练选项")
        self.label_layout = QVBoxLayout()

        self.new_label_checkbox = QCheckBox("添加新标签")
        self.new_label_checkbox.stateChanged.connect(self.toggle_label_options)
        self.label_layout.addWidget(self.new_label_checkbox)

        # 新标签输入区域（默认隐藏）
        self.label_input_layout = QHBoxLayout()
        self.new_label_input = QLineEdit()
        self.new_label_input.setPlaceholderText("输入新标签名称，多个用逗号分隔")
        self.label_input_layout.addWidget(QLabel("新标签:"))
        self.label_input_layout.addWidget(self.new_label_input)
        self.label_input_widget = QWidget()
        self.label_input_widget.setLayout(self.label_input_layout)
        self.label_input_widget.setVisible(False)
        self.label_layout.addWidget(self.label_input_widget)

        self.add_label_group.setLayout(self.label_layout)
        self.layout.addWidget(self.add_label_group)

        # --- 训练参数 ---
        self.add_section_title("3. 训练参数设置")
        params_layout = QHBoxLayout()

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(100)
        self.epochs.setPrefix("Epochs: ")
        params_layout.addWidget(self.epochs)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(16)
        self.batch_size.setPrefix("Batch: ")
        params_layout.addWidget(self.batch_size)

        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1280)
        self.imgsz.setValue(640)
        self.imgsz.setPrefix("Image Size: ")
        params_layout.addWidget(self.imgsz)

        self.layout.addLayout(params_layout)

        # --- 训练按钮 ---
        self.btn_train = QPushButton("🚀 开始训练")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_train.setStyleSheet("QPushButton { font-size: 16px; padding: 10px; }")
        self.layout.addWidget(self.btn_train, alignment=Qt.AlignCenter)

        # --- 日志显示区域 ---
        self.add_section_title("4. 训练日志")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(250)
        self.layout.addWidget(self.log_output)

        # --- 最终模型路径显示 ---
        self.result_label = QLabel("训练完成后，最佳模型路径将显示在这里")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; }")
        self.layout.addWidget(self.result_label)

    def add_section_title(self, title):
        label = QLabel(title)
        label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        self.layout.addWidget(label)
        return label

    def add_path_input(self, label_text):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        path_input = QLineEdit()
        path_input.setPlaceholderText("选择文件夹路径...")
        btn_browse = QPushButton("浏览")
        btn_browse.clicked.connect(lambda: self.browse_folder(path_input))
        layout.addWidget(label)
        layout.addWidget(path_input)
        layout.addWidget(btn_browse)
        self.layout.addLayout(layout)
        return path_input

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            line_edit.setText(folder)

    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择预训练模型文件", "", "PyTorch Models (*.pt)"
        )
        if file_path:
            self.model_path.setText(file_path)
            # 加载模型并提取类别信息
            try:
                model = YOLO(file_path)
                self.original_class_names = list(model.names.values()) if hasattr(model, 'names') else []
                self.update_class_list()
            except Exception as e:
                self.log(f"❌ 加载模型失败: {str(e)}")

    def update_class_list(self):
        """更新类别列表显示"""
        self.class_list_widget.clear()
        if self.original_class_names:
            for i, name in enumerate(self.original_class_names):
                self.class_list_widget.addItem(f"{i}: {name}")
        else:
            self.class_list_widget.addItem("⚠️ 未检测到类别信息")

    def toggle_label_options(self, state):
        """切换新标签输入框的可见性"""
        self.label_input_widget.setVisible(state == Qt.Checked)

    def log(self, message):
        self.log_output.append(message)
        self.log_output.ensureCursorVisible()
        QApplication.processEvents()  # 实时刷新界面

    def start_training(self):
        # 检查输入
        train_img = self.train_img_path.text().strip()
        train_label = self.train_label_path.text().strip()
        model_pt = self.model_path.text().strip()

        if not train_img or not train_label or not model_pt:
            QMessageBox.warning(self, "输入错误", "请填写完整的训练图片路径、标签路径，并选择预训练模型！")
            return

        if not os.path.isdir(train_img):
            QMessageBox.warning(self, "路径错误", f"训练图片文件夹不存在：{train_img}")
            return
        if not os.path.isdir(train_label):
            QMessageBox.warning(self, "路径错误", f"训练标签文件夹不存在：{train_label}")
            return
        if not os.path.isfile(model_pt):
            QMessageBox.warning(self, "文件错误", f"预训练模型文件不存在：{model_pt}")
            return

        # 参数
        epochs = self.epochs.value()
        batch = self.batch_size.value()
        imgsz = self.imgsz.value()

        # 后台线程中训练
        self.btn_train.setEnabled(False)
        self.log("🔥 开始训练 YOLOv8 模型...")
        thread = threading.Thread(
            target=self.run_training,
            args=(train_img, train_label, model_pt, epochs, batch, imgsz)
        )
        thread.start()

    def run_training(self, train_img, train_label, model_pt, epochs, batch, imgsz):
        try:
            # 获取新标签信息
            add_new_labels = self.new_label_checkbox.isChecked()
            new_labels = [label.strip() for label in self.new_label_input.text().split(',')
                          if label.strip()] if add_new_labels else []

            # 构建类别配置
            if add_new_labels and new_labels:
                nc = len(self.original_class_names) + len(new_labels)
                names_list = self.original_class_names + new_labels
            else:
                nc = len(self.original_class_names)
                names_list = self.original_class_names

            # 验证新标签
            if add_new_labels and not new_labels:
                self.log("⚠️ 已启用新标签训练但未输入有效标签，使用标准训练模式")
                add_new_labels = False

            # 构建YAML配置
            data_yaml_content = f"""
            train: {os.path.abspath(train_img)}
            val: {os.path.abspath(train_img)}  # 简化验证集使用训练集

            nc: {nc}
            names: {names_list}
            """
            data_path = "data_temp.yaml"
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(data_yaml_content)

            # 训练模式选择
            model = YOLO(model_pt)
            if add_new_labels and new_labels:
                self.log("🔧 使用迁移学习模式训练新标签...")
                model = YOLO(model_pt)

                # 获取所有检测层（YOLOv8有3个尺度）
                detect_layers = [m for m in model.model.modules() if hasattr(m, 'nc')]

                if not detect_layers:
                    raise Exception("未找到检测层，请检查模型结构")

                # === 关键修复：重建输出层 ===
                num_anchors = 3  # YOLOv8默认每层3个anchor[4](@ref)
                for detect_layer in detect_layers:
                    # 更新类别数
                    old_nc = detect_layer.nc
                    detect_layer.nc = nc

                    # 定位输出卷积层
                    output_conv = None
                    for name, module in detect_layer.named_modules():
                        if isinstance(module, nn.Conv2d) and module.out_channels == (old_nc + 5) * num_anchors:
                            output_conv = module
                            break

                    if output_conv:
                        # 计算新输出通道
                        new_out_channels = (nc + 5) * num_anchors

                        output_conv.kernel_size = fix_param_size(output_conv.kernel_size)
                        output_conv.stride = fix_param_size(output_conv.stride)
                        # 创建新卷积层（保持其他参数不变）
                        new_conv = nn.Conv2d(
                            in_channels=output_conv.in_channels,
                            out_channels=new_out_channels,
                            kernel_size=output_conv.kernel_size,
                            stride=output_conv.stride,
                            padding=output_conv.padding,
                            bias=True
                        )

                        # 迁移权重（保留可用部分）
                        with torch.no_grad():
                            # 复制可匹配的权重
                            min_channels = min(output_conv.out_channels, new_out_channels)
                            new_conv.weight[:min_channels] = output_conv.weight[:min_channels]

                            # 迁移bias（保留坐标和原始类别参数）
                            if output_conv.bias is not None:
                                new_bias = torch.zeros(new_out_channels)
                                min_bias = min(output_conv.bias。shape[0], new_out_channels)
                                new_bias[:min_bias] = output_conv.bias[:min_bias]
                                new_conv.bias = nn.Parameter(new_bias)

                        # 替换原始卷积层
                        detect_layer.conv = new_conv

                # === 冻结骨干网络 ===
                for name, param in model.model.named_parameters():
                    if "model." in name and int(name.split(".")[1]) < 15:
                        param.requires_grad = False
            else:
                self.log("🔧 使用标准训练模式...")

            # 训练日志
            self.log(f"📂 训练图片: {train_img}")
            self.log(f"📂 训练标签: {train_label}")
            self.log(f"🤖 预训练模型: {model_pt}")
            self.log(f"🏷️ 类别配置: {len(names_list)}类 ({', '.join(names_list)})")
            self.log(f"⚙️  训练参数: epochs={epochs}, batch={batch}, imgsz={imgsz}")

            # 开始训练
            results = model.train(
                data=data_path,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                name="exp",
                verbose=False
            )

            # 训练完成处理
            best_pt = "runs/train/exp/weights/best.pt"
            if os.path.isfile(best_pt):
                result_msg = f"✅ 训练完成!\n🎯 最佳模型已保存到:\n{best_pt}"
                if add_new_labels:
                    result_msg += f"\n\n✨ 新增标签: {', '.join(new_labels)}"
                self.result_label.setText(result_msg)
                self.log(f"💾 模型已保存至: {best_pt}")
            else:
                self.result_label。setText("⚠️ 训练完成，但未找到保存的模型文件，请检查日志。")

        except Exception as e:
            self.log(f"❌ 训练出错: {str(e)}")
            QMessageBox.critical(self, "训练错误", f"训练过程中发生异常：{str(e)}")
        finally:
            self.btn_train。setEnabled(True)
            if os.path.exists("data_temp.yaml"):
                os.remove("data_temp.yaml")


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = YOLOv8TrainGUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"数组越界: {e}")
