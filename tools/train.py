import sys
import os
import threading
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QLineEdit, QFileDialog,
                             QTextEdit, QSpinBox,QMessageBox)
from PyQt5.QtCore import Qt


class YOLOv8TrainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 训练工具 (PyQt5 GUI)")
        self.setGeometry(100, 100, 800, 700)

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

    def log(self, message):
        self.log_output.append(message)
        self.log_output.ensureCursorVisible()
        QApplication.processEvents()  # 实时刷新界面

    def start_training(self):
        # --- 检查输入 ---
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

        # --- 参数 ---
        epochs = self.epochs.value()
        batch = self.batch_size.value()
        imgsz = self.imgsz.value()

        # --- 在后台线程中训练 ---
        self.btn_train.setEnabled(False)
        self.log("🔥 开始训练 YOLOv8 模型...")
        thread = threading.Thread(target=self.run_training, args=(train_img, train_label, model_pt, epochs, batch, imgsz))
        thread.start()

    def run_training(self, train_img, train_label, model_pt, epochs, batch, imgsz):
        try:
            # --- 构造 data 配置（仅训练集，无验证集）---
            data_yaml_content = f"""
            train: {os.path.abspath(train_img)}
            val: {os.path.abspath(train_label)}

            nc: 1  # 假设只有一个类别，根据你的实际类别数修改
            names: ['object']  # 类别名称，可修改
            """
            data_path = "data_temp.yaml"
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(data_yaml_content)

            # --- 加载模型并训练 ---
            self.log(f"📂 训练图片: {train_img}")
            self.log(f"📂 训练标签: {train_label}")
            self.log(f"🤖 预训练模型: {model_pt}")
            self.log(f"⚙️  训练参数: epochs={epochs}, batch={batch}, imgsz={imgsz}")

            model = YOLO(model_pt)  # 加载预训练模型
            results = model.train(
                data=data_path,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                name="exp",  # 实验名称
                verbose=False
            )

            # --- 训练完成 ---
            best_pt = "runs/train/exp/weights/best.pt"
            if os.path.isfile(best_pt):
                self.result_label.setText(f"✅ 训练完成！\n🎯 最佳模型已保存到：\n{best_pt}\n\n🔧 这是一个标准的 .pt 格式 YOLOv8 模型，可直接用于推理。")
                self.log(f"💾 模型已保存至: {best_pt}")
            else:
                self.result_label.setText("⚠️ 训练完成，但未找到保存的模型文件，请检查日志。")

        except Exception as e:
            self.log(f"❌ 训练出错: {str(e)}")
            QMessageBox.critical(self, "训练错误", f"训练过程中发生异常：{str(e)}")
        finally:
            self.btn_train.setEnabled(True)
            # 清理临时 YAML（可选）
            if os.path.exists("data_temp.yaml"):
                os.remove("data_temp.yaml")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOv8TrainGUI()
    window.show()
    sys.exit(app.exec_())