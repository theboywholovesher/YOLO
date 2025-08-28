import os
import sys
import cv2
import numpy as np
import torch
import yaml
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QLineEdit, QPushButton, QTextEdit, QListWidget,
                             QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QComboBox,
                             QGridLayout, QCheckBox)
from torch.utils.data import Dataset
from ultralytics import YOLO


class YOLODataset(Dataset):
    """YOLO格式数据集类"""

    def __init__(self, img_dir, label_dir):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)
                                   if f.endswith('.txt')])

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.img_paths[idx])
            if img is None:
                raise ValueError(f"无法读取图像: {self.img_paths[idx]}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换BGR为RGB
            label = np.loadtxt(self.label_paths[idx], ndmin=2)  # 加载标签
            return torch.tensor(img).permute(2, 0, 1), torch.tensor(label)
        except Exception as e:
            print(f"数据加载错误 (索引{idx}): {e}")
            raise

    def __len__(self):
        return len(self.img_paths)


class YOLOTrainerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8模型训练管理器")
        self.setGeometry(100, 100, 1000, 800)

        # 存储当前类别名称和数据集路径
        self.current_names = []  # 用于存储类别名称
        self.dataset_paths = {
            'train_images': '', 'train_labels': '',
            'val_images': '', 'val_labels': ''
        }
        self.model = None  # 存储YOLO模型

        # 初始化训练参数为默认值
        self.training_params = {
            'epochs': 50,
            'batch_size': 4,
            'img_size': 640,
            'learning_rate': 0.0001,
            'weight_decay': 0.0001,
            'optimizer': 'AdamW',
            'device': '0' if torch.cuda.is_available() else 'cpu',
            'patience': 50,
            'freeze_layers': 10,
            'augment': True,
            'mixup': 0.0,
            'cos_lr': False,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1
        }

        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建一个主网格布局，用于更灵活地安排各部分
        main_grid = QGridLayout()

        # 1. 数据集配置区域
        dataset_group = QGroupBox("数据集配置")
        dataset_layout = QVBoxLayout()

        # 训练集路径选择
        train_img_btn = QPushButton("选择训练图像文件夹")
        train_img_btn.clicked.connect(lambda: self.select_dataset_path('train_images'))
        dataset_layout.addWidget(train_img_btn)

        train_label_btn = QPushButton("选择训练标签文件夹")
        train_label_btn.clicked.connect(lambda: self.select_dataset_path('train_labels'))
        dataset_layout.addWidget(train_label_btn)

        # 验证集路径选择
        val_img_btn = QPushButton("选择验证图像文件夹")
        val_img_btn.clicked.connect(lambda: self.select_dataset_path('val_images'))
        dataset_layout.addWidget(val_img_btn)

        val_label_btn = QPushButton("选择验证标签文件夹")
        val_label_btn.clicked.connect(lambda: self.select_dataset_path('val_labels'))
        dataset_layout.addWidget(val_label_btn)

        # 生成YAML按钮
        gen_yaml_btn = QPushButton("生成YAML配置文件")
        gen_yaml_btn.clicked.connect(self.generate_yaml_config)
        dataset_layout.addWidget(gen_yaml_btn)

        dataset_group.setLayout(dataset_layout)
        main_grid.addWidget(dataset_group, 0, 0)  # 第0行，第0列

        # 2. 类别管理区域
        class_group = QGroupBox("类别管理")
        class_layout = QVBoxLayout()

        # 当前类别显示
        class_layout.addWidget(QLabel("当前类别列表:"))
        self.class_list_widget = QListWidget()  # 用于显示当前类别
        class_layout.addWidget(self.class_list_widget)

        # 添加新类别
        add_class_layout = QHBoxLayout()
        self.new_class_input = QLineEdit()
        self.new_class_input.setPlaceholderText("输入新类别名称")
        add_class_btn = QPushButton("添加类别")
        add_class_btn.clicked.connect(self.add_new_class)
        add_class_layout.addWidget(self.new_class_input)
        add_class_layout.addWidget(add_class_btn)
        class_layout.addLayout(add_class_layout)

        # 删除选中类别
        del_class_btn = QPushButton("删除选中类别")
        del_class_btn.clicked.connect(self.delete_selected_class)
        class_layout.addWidget(del_class_btn)

        class_group.setLayout(class_layout)
        main_grid.addWidget(class_group, 0, 1)  # 第0行，第1列

        # 3. 训练参数配置区域
        params_group = QGroupBox("训练参数配置")
        params_layout = QGridLayout()

        # 基本参数
        row = 0
        params_layout.addWidget(QLabel("迭代次数 (epochs):"), row, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(self.training_params['epochs'])
        self.epochs_spin.valueChanged.connect(lambda v: self.update_param('epochs', v))
        params_layout.addWidget(self.epochs_spin, row, 1)

        row += 1
        params_layout.addWidget(QLabel("批次大小 (batch):"), row, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(self.training_params['batch_size'])
        self.batch_spin.valueChanged.connect(lambda v: self.update_param('batch_size', v))
        params_layout.addWidget(self.batch_spin, row, 1)

        row += 1
        params_layout.addWidget(QLabel("图像尺寸 (imgsz):"), row, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(self.training_params['img_size'])
        self.imgsz_spin.valueChanged.connect(lambda v: self.update_param('img_size', v))
        params_layout.addWidget(self.imgsz_spin, row, 1)

        row += 1
        params_layout.addWidget(QLabel("优化器 (optimizer):"), row, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['SGD', 'Adam', 'AdamW', 'RMSProp'])
        self.optimizer_combo.setCurrentText(self.training_params['optimizer'])
        self.optimizer_combo.currentTextChanged.connect(lambda v: self.update_param('optimizer', v))
        params_layout.addWidget(self.optimizer_combo, row, 1)

        row += 1
        params_layout.addWidget(QLabel("初始学习率 (lr0):"), row, 0)
        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setRange(0.00001, 0.1)
        self.lr0_spin.setDecimals(6)
        self.lr0_spin.setSingleStep(0.0001)
        self.lr0_spin.setValue(self.training_params['lr0'])
        self.lr0_spin.valueChanged.connect(lambda v: self.update_param('lr0', v))
        params_layout.addWidget(self.lr0_spin, row, 1)

        row += 1
        params_layout.addWidget(QLabel("最终学习率因子 (lrf):"), row, 0)
        self.lrf_spin = QDoubleSpinBox()
        self.lrf_spin.setRange(0.0001, 1.0)
        self.lrf_spin.setDecimals(4)
        self.lrf_spin.setValue(self.training_params['lrf'])
        self.lrf_spin.valueChanged.connect(lambda v: self.update_param('lrf', v))
        params_layout.addWidget(self.lrf_spin, row, 1)

        row += 1
        params_layout.addWidget(QLabel("动量 (momentum):"), row, 0)
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setDecimals(3)
        self.momentum_spin.setValue(self.training_params['momentum'])
        self.momentum_spin.valueChanged.connect(lambda v: self.update_param('momentum', v))
        params_layout.addWidget(self.momentum_spin, row, 1)

        row += 1
        params_layout.addWidget(QLabel("权重衰减 (weight_decay):"), row, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.1)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setValue(self.training_params['weight_decay'])
        self.weight_decay_spin.valueChanged.connect(lambda v: self.update_param('weight_decay', v))
        params_layout.addWidget(self.weight_decay_spin, row, 1)

        row += 1
        params_layout.addWidget(QLabel("设备 (device):"), row, 0)
        self.device_combo = QComboBox()
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.extend([str(i) for i in range(torch.cuda.device_count())])
            devices.append('all')
        self.device_combo.addItems(devices)
        self.device_combo.setCurrentText(self.training_params['device'])
        self.device_combo.currentTextChanged.connect(lambda v: self.update_param('device', v))
        params_layout.addWidget(self.device_combo, row, 1)

        row += 1
        params_layout.addWidget(QLabel("早停轮次 (patience):"), row, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 100)
        self.patience_spin.setValue(self.training_params['patience'])
        self.patience_spin.valueChanged.connect(lambda v: self.update_param('patience', v))
        params_layout.addWidget(self.patience_spin, row, 1)

        row += 1
        params_layout.addWidget(QLabel("冻结层数:"), row, 0)
        self.freeze_spin = QSpinBox()
        self.freeze_spin.setRange(0, 30)
        self.freeze_spin.setValue(self.training_params['freeze_layers'])
        self.freeze_spin.valueChanged.connect(lambda v: self.update_param('freeze_layers', v))
        params_layout.addWidget(self.freeze_spin, row, 1)

        row += 1
        params_layout.addWidget(QLabel("混合数据增强 (mixup):"), row, 0)
        self.mixup_spin = QDoubleSpinBox()
        self.mixup_spin.setRange(0.0, 1.0)
        self.mixup_spin.setDecimals(2)
        self.mixup_spin.setValue(self.training_params['mixup'])
        self.mixup_spin.valueChanged.connect(lambda v: self.update_param('mixup', v))
        params_layout.addWidget(self.mixup_spin, row, 1)

        row += 1
        self.cos_lr_check = QCheckBox("使用余弦学习率调度")
        self.cos_lr_check.setChecked(self.training_params['cos_lr'])
        self.cos_lr_check.stateChanged.connect(lambda v: self.update_param('cos_lr', bool(v)))
        params_layout.addWidget(self.cos_lr_check, row, 0, 1, 2)

        row += 1
        self.augment_check = QCheckBox("启用数据增强")
        self.augment_check.setChecked(self.training_params['augment'])
        self.augment_check.stateChanged.connect(lambda v: self.update_param('augment', bool(v)))
        params_layout.addWidget(self.augment_check, row, 0, 1, 2)

        params_group.setLayout(params_layout)
        main_grid.addWidget(params_group, 1, 0, 1, 2)  # 第1行，占据0-1列

        # 4. 模型操作区域
        model_group = QGroupBox("模型操作")
        model_layout = QHBoxLayout()  # 使用水平布局让按钮并排显示

        # 加载模型按钮
        load_model_btn = QPushButton("加载模型")
        load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_btn)

        # 训练按钮
        train_btn = QPushButton("开始训练")
        train_btn.clicked.connect(self.start_training)
        model_layout.addWidget(train_btn)

        # 冻结训练按钮
        freeze_train_btn = QPushButton("冻结骨干网络训练")
        freeze_train_btn.clicked.connect(self.freeze_backbone_training)
        model_layout.addWidget(freeze_train_btn)

        # 导出模型按钮
        export_btn = QPushButton("导出模型")
        export_btn.clicked.connect(self.export_model)
        model_layout.addWidget(export_btn)

        model_group.setLayout(model_layout)
        main_grid.addWidget(model_group, 2, 0, 1, 2)  # 第2行，占据0-1列

        # 5. 状态显示区域
        status_group = QGroupBox("训练状态")
        status_layout = QVBoxLayout()
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)
        status_group.setLayout(status_layout)
        main_grid.addWidget(status_group, 3, 0, 1, 2)  # 第3行，占据0-1列

        main_layout.addLayout(main_grid)

    def update_param(self, param_name, value):
        """更新训练参数值"""
        self.training_params[param_name] = value
        self.status_text.append(f"已更新参数: {param_name} = {value}")

    def select_dataset_path(self, path_type):
        """选择数据集路径"""
        folder = QFileDialog.getExistingDirectory(self, f"选择{path_type}文件夹")
        if folder:
            self.dataset_paths[path_type] = folder
            self.status_text.append(f"{path_type}路径已设置: {folder}")

    def load_model(self):
        """加载YOLOv8模型并显示类别名称"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.pt)")
        if file_path:
            try:
                self.model = YOLO(file_path)
                # 将模型的类别字典转换为按ID排序的列表
                self.current_names = [self.model.names[i] for i in sorted(self.model.names.keys())]
                self.update_class_list()  # 更新界面显示的类别列表
                self.status_text.append(f"模型加载成功: {file_path}")
                self.status_text.append(f"当前模型类别: {self.current_names}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
                self.status_text.append(f"加载模型失败: {str(e)}")

    def update_class_list(self):
        """更新界面上的类别列表显示"""
        self.class_list_widget.clear()
        for name in self.current_names:
            self.class_list_widget.addItem(str(name))

    def add_new_class(self):
        """添加新类别"""
        new_class = self.new_class_input.text().strip()
        if new_class:
            # 过滤可能导致问题的特殊字符
            new_class = new_class.replace('\x00', '').replace('\r', '').replace('\n', '')

            if new_class not in self.current_names:
                self.current_names.append(new_class)
                self.class_list_widget.addItem(new_class)
                self.new_class_input.clear()
                self.status_text.append(f"已添加类别: {new_class}")
            else:
                QMessageBox.warning(self, "警告", "该类别已存在！")
        else:
            QMessageBox.warning(self, "警告", "请输入有效的类别名称！")

    def delete_selected_class(self):
        """删除选中的类别"""
        selected_items = self.class_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要删除的类别！")
            return

        for item in selected_items:
            class_name = item.text()
            if class_name in self.current_names:
                self.current_names.remove(class_name)
                self.status_text.append(f"已删除类别: {class_name}")

        self.update_class_list()

    def generate_yaml_config(self):
        """生成YAML配置文件"""
        # 检查必要的路径是否已设置
        required_paths = ['train_images', 'train_labels', 'val_images', 'val_labels']
        for path_type in required_paths:
            if not self.dataset_paths[path_type]:
                QMessageBox.warning(self, "警告", f"请先设置{path_type}路径！")
                return

        # 检查是否有类别
        if not self.current_names:
            QMessageBox.warning(self, "警告", "请先添加类别！")
            return

        # 创建YAML配置内容
        config = {
            'train': self.dataset_paths['train_images'],
            'val': self.dataset_paths['val_images'],
            'nc': len(self.current_names),
            'names': self.current_names
        }

        # 保存YAML文件
        try:
            yaml_path = 'dataset_config.yaml'
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            self.status_text.append(f"YAML配置文件已生成: {os.path.abspath(yaml_path)}")
            QMessageBox.information(self, "成功", f"YAML配置文件已保存至: {yaml_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成YAML文件失败: {str(e)}")
            self.status_text.append(f"生成YAML文件失败: {str(e)}")

    def _generate_temp_yaml(self):
        """生成临时YAML配置文件并返回路径"""
        temp_yaml = 'temp_train_config.yaml'
        try:
            config = {
                'train': self.dataset_paths['train_images'],
                'val': self.dataset_paths['val_images'],
                'nc': len(self.current_names),
                'names': self.current_names
            }

            with open(temp_yaml, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            self.status_text.append(f"生成临时训练配置: 类别数={len(self.current_names)}, 类别={self.current_names}")
            return temp_yaml
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成临时YAML失败: {str(e)}")
            self.status_text.append(f"生成临时YAML失败: {str(e)}")
            return None

    def _validate_training_conditions(self):
        """验证训练条件是否满足"""
        if not self.model:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return False

        if not all(self.dataset_paths.values()):
            QMessageBox.warning(self, "警告", "请先设置所有数据集路径！")
            return False

        if not self.current_names:
            QMessageBox.warning(self, "警告", "请先添加类别！")
            return False

        # 验证标签中的类别ID是否在有效范围内
        max_class_id = len(self.current_names) - 1
        if max_class_id < 0:
            QMessageBox.warning(self, "警告", "请至少添加一个类别！")
            return False

        # 检查训练标签
        if not self._validate_labels(self.dataset_paths['train_labels'], max_class_id):
            return False

        # 检查验证标签
        if not self._validate_labels(self.dataset_paths['val_labels'], max_class_id):
            return False

        return True

    def _validate_labels(self, label_dir, max_class_id):
        """验证标签文件中的类别ID是否在有效范围内"""
        if not os.path.exists(label_dir):
            QMessageBox.warning(self, "标签错误", f"标签文件夹不存在: {label_dir}")
            return False

        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                label_path = os.path.join(label_dir, label_file)
                try:
                    labels = np.loadtxt(label_path, ndmin=2)
                    if labels.size > 0 and np.any(labels[:, 0] > max_class_id):
                        QMessageBox.warning(
                            self, "标签错误",
                            f"标签文件 {label_file} 中存在超出范围的类别ID，最大允许值为 {max_class_id}"
                        )
                        return False
                except Exception as e:
                    QMessageBox.warning(
                        self, "标签错误",
                        f"解析标签文件 {label_file} 时出错: {str(e)}"
                    )
                    return False
        return True

    def start_training(self):
        """开始训练模型"""
        if not self._validate_training_conditions():
            return

        try:
            # 生成临时的YAML配置用于训练
            temp_yaml = self._generate_temp_yaml()
            if not temp_yaml:
                return

            # 开始训练，使用配置的参数
            self.status_text.append("开始训练模型...")
            results = self.model.train(
                data=temp_yaml,
                epochs=self.training_params['epochs'],
                imgsz=self.training_params['img_size'],
                batch=self.training_params['batch_size'],
                optimizer=self.training_params['optimizer'],
                lr0=self.training_params['lr0'],
                lrf=self.training_params['lrf'],
                momentum=self.training_params['momentum'],
                weight_decay=self.training_params['weight_decay'],
                device=self.training_params['device'],
                patience=self.training_params['patience'],
                cos_lr=self.training_params['cos_lr'],
                mixup=self.training_params['mixup'],
                augment=self.training_params['augment'],
                warmup_epochs=self.training_params['warmup_epochs'],
                warmup_momentum=self.training_params['warmup_momentum'],
                warmup_bias_lr=self.training_params['warmup_bias_lr']
            )

            self.status_text.append("训练完成！")
            # 训练完成后，更新模型的类别名称
            self.model.names = {i: name for i, name in enumerate(self.current_names)}
            self.update_class_list()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"训练失败: {str(e)}")
            self.status_text.append(f"训练错误: {str(e)}")

    def freeze_backbone_training(self):
        """冻结骨干网络进行训练"""
        if not self._validate_training_conditions():
            return

        try:
            # 生成临时的YAML配置用于训练
            temp_yaml = self._generate_temp_yaml()
            if not temp_yaml:
                return

            # 冻结指定数量的骨干网络层
            freeze_layers = [f"model.{i}" for i in range(self.training_params['freeze_layers'])]
            for name, param in self.model.model.named_parameters():
                if any(layer in name for layer in freeze_layers):
                    param.requires_grad = False

            self.status_text.append(f"已冻结 {self.training_params['freeze_layers']} 层骨干网络，开始训练...")

            # 开始训练，使用配置的参数
            results = self.model.train(
                data=temp_yaml,
                epochs=self.training_params['epochs'],
                imgsz=self.training_params['img_size'],
                batch=self.training_params['batch_size'],
                optimizer=self.training_params['optimizer'],
                lr0=self.training_params['lr0'],
                lrf=self.training_params['lrf'],
                momentum=self.training_params['momentum'],
                weight_decay=self.training_params['weight_decay'],
                device=self.training_params['device'],
                patience=self.training_params['patience'],
                cos_lr=self.training_params['cos_lr'],
                mixup=self.training_params['mixup'],
                augment=self.training_params['augment'],
                warmup_epochs=self.training_params['warmup_epochs'],
                warmup_momentum=self.training_params['warmup_momentum'],
                warmup_bias_lr=self.training_params['warmup_bias_lr']
            )

            self.status_text.append("冻结训练完成！")
            # 训练完成后，更新模型的类别名称
            self.model.names = {i: name for i, name in enumerate(self.current_names)}
            self.update_class_list()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"冻结训练失败: {str(e)}")
            self.status_text.append(f"冻结训练错误: {str(e)}")

    def export_model(self):
        """导出模型为多种格式"""
        if not self.model:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return

        try:
            # 创建格式选择对话框
            format_dialog = QWidget()
            format_dialog.setWindowTitle("选择导出格式")
            layout = QVBoxLayout(format_dialog)

            layout.addWidget(QLabel("请选择要导出的模型格式:"))
            format_combo = QComboBox()
            format_combo.addItems(["engine", "onnx", "coreml", "pb", "tflite", "torchscript"])
            format_combo.setCurrentText("engine")
            layout.addWidget(format_combo)

            btn_layout = QHBoxLayout()
            ok_btn = QPushButton("确定")
            cancel_btn = QPushButton("取消")
            btn_layout.addWidget(ok_btn)
            btn_layout.addWidget(cancel_btn)
            layout.addLayout(btn_layout)

            # 对话框结果标志
            result = [False]

            def on_ok():
                result[0] = True
                format_dialog.close()

            ok_btn.clicked.connect(on_ok)
            cancel_btn.clicked.connect(format_dialog.close)

            format_dialog.exec_()

            if not result[0]:
                return

            export_format = format_combo.currentText()
            self.status_text.append(f"开始导出模型为{export_format}格式...")

            # 导出模型
            export_path = self.model.export(
                format=export_format,
                device=self.training_params['device']
            )

            self.status_text.append(f"模型已成功导出为{export_format}格式: {export_path}")
            QMessageBox.information(self, "成功", f"模型已成功导出为{export_format}格式: {export_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出模型失败: {str(e)}")
            self.status_text.append(f"导出模型错误: {str(e)}")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = YOLOTrainerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
