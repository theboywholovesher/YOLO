from PIL import Image, ImageEnhance, ImageFilter
import os
import xml.etree.ElementTree as ET
import shutil
from typing import List, Dict, Tuple
import cv2
import numpy as np
import json


def transform_bbox(bndbox, width, height, transpose_type):
    """转换边界框坐标"""
    xmin, ymin, xmax, ymax = bndbox

    if transpose_type == 0:  # 左右翻转
        new_xmin = width - xmax
        new_xmax = width - xmin
        return [new_xmin, ymin, new_xmax, ymax]

    elif transpose_type == 1:  # 上下翻转
        new_ymin = height - ymax
        new_ymax = height - ymin
        return [xmin, new_ymin, xmax, new_ymax]

    elif transpose_type == 2:  # 旋转90°（注意：宽高互换）
        new_xmin = ymin
        new_ymin = width - xmax
        new_xmax = ymax
        new_ymax = width - xmin
        return [new_xmin, new_ymin, new_xmax, new_ymax]

    elif transpose_type == 3:  # 旋转180°
        new_xmin = width - xmax
        new_ymin = height - ymax
        new_xmax = width - xmin
        new_ymax = height - ymin
        return [new_xmin, new_ymin, new_xmax, new_ymax]

    elif transpose_type == 4:  # 旋转270°
        new_xmin = height - ymax
        new_ymin = xmin
        new_xmax = height - ymin
        new_ymax = xmax
        return [new_xmin, new_ymin, new_xmax, new_ymax]

    elif transpose_type == 5:  # 沿左上-右下对角线转置 (x <-> y)
        new_xmin = ymin
        new_ymin = xmin
        new_xmax = ymax
        new_ymax = xmax
        return [new_xmin, new_ymin, new_xmax, new_ymax]

    elif transpose_type == 6:  # 沿右上-左下对角线转置
        new_xmin = height - ymax
        new_ymin = width - xmax
        new_xmax = height - ymin
        new_ymax = width - xmin
        return [new_xmin, new_ymin, new_xmax, new_ymax]

    else:  # 原图不变
        return bndbox


def transform_yolo_bbox(x_center, y_center, width, height, img_width, img_height, transpose_type):
    """转换YOLO格式的边界框坐标"""
    # 转换为绝对坐标
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    # 转换为xmin, ymin, xmax, ymax格式
    xmin = x_center_abs - width_abs / 2
    ymin = y_center_abs - height_abs / 2
    xmax = x_center_abs + width_abs / 2
    ymax = y_center_abs + height_abs / 2
    
    # 应用变换
    new_bbox = transform_bbox([xmin, ymin, xmax, ymax], img_width, img_height, transpose_type)
    
    # 转换回YOLO格式
    new_x_center = (new_bbox[0] + new_bbox[2]) / 2
    new_y_center = (new_bbox[1] + new_bbox[3]) / 2
    new_width = new_bbox[2] - new_bbox[0]
    new_height = new_bbox[3] - new_bbox[1]
    
    # 如果图像尺寸发生变化（如旋转90°或270°），需要调整
    if transpose_type in [2, 4]:  # 旋转90°或270°
        new_img_width = img_height
        new_img_height = img_width
    else:
        new_img_width = img_width
        new_img_height = img_height
    
    # 归一化坐标
    new_x_center_norm = new_x_center / new_img_width
    new_y_center_norm = new_y_center / new_img_height
    new_width_norm = new_width / new_img_width
    new_height_norm = new_height / new_img_height
    
    return [new_x_center_norm, new_y_center_norm, new_width_norm, new_height_norm]


def detect_label_format(label_path):
    """检测标签文件格式"""
    if not os.path.exists(label_path):
        return None
    
    file_ext = os.path.splitext(label_path)[1].lower()
    
    if file_ext == '.xml':
        return 'xml'
    elif file_ext == '.txt':
        # 检查是否为YOLO格式
        try:
            with open(label_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    parts = first_line.split()
                    if len(parts) >= 5:  # YOLO格式：class_id x_center y_center width height
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                                return 'yolo'
                        except:
                            pass
        except:
            pass
        return 'txt'
    elif file_ext == '.json':
        return 'coco'
    else:
        return 'unknown'


def parse_xml(xml_path):
    """解析XML标注文件"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            objects.append({
                'name': name,
                'bndbox': [xmin, ymin, xmax, ymax]
            })
        return width, height, objects
    except Exception as e:
        print(f"解析XML文件失败: {e}")
        return None, None, []


def parse_yolo(label_path, img_width, img_height):
    """解析YOLO格式标签文件"""
    try:
        objects = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        objects.append({
                            'class_id': class_id,
                            'bbox': [x_center, y_center, width, height]
                        })
        return objects
    except Exception as e:
        print(f"解析YOLO标签文件失败: {e}")
        return []


def parse_coco(json_path):
    """解析COCO格式标签文件"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 这里需要根据具体的COCO文件结构来解析
        # 简化版本，实际使用时可能需要更复杂的处理
        return data
    except Exception as e:
        print(f"解析COCO标签文件失败: {e}")
        return None


def create_xml(xml_path, output_path, width, height, objects, transpose_type):
    """创建新的XML标注文件"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 更新尺寸信息
        size = root.find('size')
        if transpose_type in [2, 4]:  # 旋转90°或270°时宽高互换
            size.find('width').text = str(height)
            size.find('height').text = str(width)
        else:
            size.find('width').text = str(width)
            size.find('height').text = str(height)
        
        # 更新边界框坐标
        for i, obj in enumerate(root.findall('object')):
            if i < len(objects):
                bndbox = obj.find('bndbox')
                new_bbox = transform_bbox(objects[i]['bndbox'], width, height, transpose_type)
                bndbox.find('xmin').text = str(new_bbox[0])
                bndbox.find('ymin').text = str(new_bbox[1])
                bndbox.find('xmax').text = str(new_bbox[2])
                bndbox.find('ymax').text = str(new_bbox[3])
        
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return True
    except Exception as e:
        print(f"创建XML文件失败: {e}")
        return False


def create_yolo_label(label_path, output_path, img_width, img_height, objects, transpose_type):
    """创建新的YOLO格式标签文件"""
    try:
        new_objects = []
        
        for obj in objects:
            class_id = obj['class_id']
            bbox = obj['bbox']
            
            # 转换边界框坐标
            new_bbox = transform_yolo_bbox(bbox[0], bbox[1], bbox[2], bbox[3], 
                                         img_width, img_height, transpose_type)
            
            new_objects.append({
                'class_id': class_id,
                'bbox': new_bbox
            })
        
        # 写入新文件
        with open(output_path, 'w') as f:
            for obj in new_objects:
                bbox = obj['bbox']
                f.write(f"{obj['class_id']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        return True
    except Exception as e:
        print(f"创建YOLO标签文件失败: {e}")
        return False


def create_coco_label(coco_data, output_path, img_width, img_height, transpose_type):
    """创建新的COCO格式标签文件"""
    try:
        # 这里需要根据具体的COCO数据结构来更新
        # 简化版本，实际使用时可能需要更复杂的处理
        new_data = coco_data.copy()
        
        # 更新图像尺寸
        if 'images' in new_data:
            for img in new_data['images']:
                if transpose_type in [2, 4]:  # 旋转90°或270°时宽高互换
                    img['width'], img['height'] = img['height'], img['width']
        
        # 更新标注信息
        if 'annotations' in new_data:
            for ann in new_data['annotations']:
                if 'bbox' in ann:
                    # 转换边界框坐标
                    bbox = ann['bbox']  # [x, y, width, height]
                    x, y, w, h = bbox
                    
                    # 转换为xmin, ymin, xmax, ymax格式
                    xmin, ymin, xmax, ymax = x, y, x + w, y + h
                    
                    # 应用变换
                    new_bbox = transform_bbox([xmin, ymin, xmax, ymax], img_width, img_height, transpose_type)
                    
                    # 转换回[x, y, width, height]格式
                    new_x = new_bbox[0]
                    new_y = new_bbox[1]
                    new_w = new_bbox[2] - new_bbox[0]
                    new_h = new_bbox[3] - new_bbox[1]
                    
                    ann['bbox'] = [new_x, new_y, new_w, new_h]
        
        # 写入新文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"创建COCO标签文件失败: {e}")
        return False


def copy_label_file(label_path, output_path, label_format):
    """复制标签文件（用于不需要坐标变换的增强）"""
    try:
        shutil.copy2(label_path, output_path)
        return True
    except Exception as e:
        print(f"复制标签文件失败: {e}")
        return False


def xml_to_yolo(xml_path, output_path, class_mapping=None):
    """将XML格式转换为YOLO格式"""
    try:
        # 解析XML文件
        width, height, objects = parse_xml(xml_path)
        if width is None:
            return False
        
        # 如果没有提供类别映射，创建默认映射
        if class_mapping is None:
            # 自动创建类别映射
            unique_classes = list(set([obj['name'] for obj in objects]))
            class_mapping = {class_name: i for i, class_name in enumerate(unique_classes)}
        
        # 转换并写入YOLO格式
        with open(output_path, 'w') as f:
            for obj in objects:
                class_name = obj['name']
                bbox = obj['bndbox']
                
                # 获取类别ID
                if class_name in class_mapping:
                    class_id = class_mapping[class_name]
                else:
                    # 如果类别不在映射中，添加新类别
                    class_id = len(class_mapping)
                    class_mapping[class_name] = class_id
                
                # 转换为YOLO格式 (x_center, y_center, width, height)
                xmin, ymin, xmax, ymax = bbox
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                
                # 确保坐标在[0,1]范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                
                # 写入YOLO格式
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
        return True, class_mapping
    except Exception as e:
        print(f"XML转YOLO失败: {e}")
        return False, None


def batch_xml_to_yolo(input_dir, output_dir, class_mapping=None):
    """批量转换XML文件为YOLO格式"""
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有XML文件
    xml_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.xml')]
    
    if not xml_files:
        print("未找到XML文件")
        return False
    
    success_count = 0
    total_files = len(xml_files)
    
    # 如果没有提供类别映射，从第一个文件创建
    if class_mapping is None:
        first_xml = os.path.join(input_dir, xml_files[0])
        _, _, first_objects = parse_xml(first_xml)
        if first_objects:
            unique_classes = list(set([obj['name'] for obj in first_objects]))
            class_mapping = {class_name: i for i, class_name in enumerate(unique_classes)}
            print(f"自动创建类别映射: {class_mapping}")
    
    for i, xml_file in enumerate(xml_files):
        xml_path = os.path.join(input_dir, xml_file)
        
        # 生成输出文件名
        base_name = os.path.splitext(xml_file)[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        
        # 转换文件
        success, updated_mapping = xml_to_yolo(xml_path, output_path, class_mapping)
        if success:
            success_count += 1
            # 更新类别映射（如果有新类别）
            if updated_mapping:
                class_mapping = updated_mapping
        
        # 显示进度
        progress = (i + 1) / total_files * 100
        print(f"进度: {progress:.1f}% - 处理: {xml_file}")
    
    print(f"批量转换完成: {success_count}/{total_files} 个文件成功")
    
    # 保存类别映射文件
    if class_mapping:
        mapping_file = os.path.join(output_dir, "classes.txt")
        try:
            with open(mapping_file, 'w', encoding='utf-8') as f:
                for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
                    f.write(f"{class_id} {class_name}\n")
            print(f"类别映射已保存到: {mapping_file}")
        except Exception as e:
            print(f"保存类别映射失败: {e}")
    
    return success_count > 0


def process_label_file(image_path, label_path, output_dir, base_name, aug_type, transpose_type, label_output_dir=None):
    """处理标签文件"""
    if not os.path.exists(label_path):
        return True  # 没有标签文件时返回成功
    
    # 检测标签格式
    label_format = detect_label_format(label_path)
    
    if label_format is None:
        return True
    
    # 确定标签文件输出目录
    if label_output_dir:
        final_output_dir = label_output_dir
    else:
        final_output_dir = output_dir
    
    # 创建标签输出目录
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 生成输出标签文件路径
    label_ext = os.path.splitext(label_path)[1]
    output_label_path = os.path.join(final_output_dir, f"{base_name}_{aug_type}{label_ext}")
    
    # 获取图像尺寸
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
    except:
        print(f"无法获取图像尺寸: {image_path}")
        return False
    
    # 根据标签格式和增强类型处理
    if transpose_type in [0, 1, 2, 3, 4, 5, 6]:  # 需要坐标变换的增强
        if label_format == 'xml':
            width, height, objects = parse_xml(label_path)
            if width is not None:
                return create_xml(label_path, output_label_path, width, height, objects, transpose_type)
        elif label_format == 'yolo':
            objects = parse_yolo(label_path, img_width, img_height)
            if objects:
                return create_yolo_label(label_path, output_label_path, img_width, img_height, objects, transpose_type)
        elif label_format == 'coco':
            coco_data = parse_coco(label_path)
            if coco_data:
                return create_coco_label(coco_data, output_label_path, img_width, img_height, transpose_type)
    else:
        # 不需要坐标变换的增强，直接复制标签文件
        return copy_label_file(label_path, output_label_path, label_format)
    
    return False


def augment_image_with_labels(image_path, output_dir, augmentations, progress_callback=None, label_output_dir=None):
    """执行图像增强并更新标签文件"""
    if not os.path.exists(image_path):
        print(f"图像路径不存在: {image_path}")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找对应的标签文件
    label_path = find_label_file(image_path)
    
    # 加载图像
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except Exception as e:
        print(f"加载图像失败: {e}")
        return False
    
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    success_count = 0
    total_operations = len(augmentations)
    
    for i, aug_type in enumerate(augmentations):
        try:
            if aug_type == "original":
                # 保存原图
                output_img_path = os.path.join(output_dir, f"{base_name}_original.jpg")
                img.save(output_img_path)
                
                # 处理标签文件
                if label_path:
                    process_label_file(image_path, label_path, output_dir, base_name, "original", -1, label_output_dir)
                
                success_count += 1
                
            elif aug_type == "flip_horizontal":
                # 左右翻转
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                output_img_path = os.path.join(output_dir, f"{base_name}_flip_h.jpg")
                flipped_img.save(output_img_path)
                
                # 处理标签文件
                if label_path:
                    process_label_file(image_path, label_path, output_dir, base_name, "flip_h", 0, label_output_dir)
                
                success_count += 1
                
            elif aug_type == "flip_vertical":
                # 上下翻转
                flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                output_img_path = os.path.join(output_dir, f"{base_name}_flip_v.jpg")
                flipped_img.save(output_img_path)
                
                # 处理标签文件
                if label_path:
                    process_label_file(image_path, label_path, output_dir, base_name, "flip_v", 1, label_output_dir)
                
                success_count += 1
                
            elif aug_type == "rotate_90":
                # 旋转90°
                rotated_img = img.transpose(Image.ROTATE_90)
                output_img_path = os.path.join(output_dir, f"{base_name}_rot90.jpg")
                rotated_img.save(output_img_path)
                
                # 处理标签文件
                if label_path:
                    process_label_file(image_path, label_path, output_dir, base_name, "rot90", 2, label_output_dir)
                
                success_count += 1
                
            elif aug_type == "rotate_180":
                # 旋转180°
                rotated_img = img.transpose(Image.ROTATE_180)
                output_img_path = os.path.join(output_dir, f"{base_name}_rot180.jpg")
                rotated_img.save(output_img_path)
                
                # 处理标签文件
                if label_path:
                    process_label_file(image_path, label_path, output_dir, base_name, "rot180", 3, label_output_dir)
                
                success_count += 1
                
            elif aug_type == "rotate_270":
                # 旋转270°
                rotated_img = img.transpose(Image.ROTATE_270)
                output_img_path = os.path.join(output_dir, f"{base_name}_rot270.jpg")
                rotated_img.save(output_img_path)
                
                # 处理标签文件
                if label_path:
                    process_label_file(image_path, label_path, output_dir, base_name, "rot270", 4, label_output_dir)
                
                success_count += 1
                
            elif aug_type == "brightness_up":
                # 亮度增加
                bright_img = apply_brightness_contrast(img, brightness_factor=1.3)
                output_img_path = os.path.join(output_dir, f"{base_name}_bright_up.jpg")
                bright_img.save(output_img_path)
                
                # 处理标签文件（不需要坐标变换）
                if label_path:
                    process_label_file(image_path, label_path, output_dir, base_name, "bright_up", -1, label_output_dir)
                
                success_count += 1
                
            elif aug_type == "brightness_down":
                # 亮度减少
                dark_img = apply_brightness_contrast(img, brightness_factor=0.7)
                output_img_path = os.path.join(output_dir, f"{base_name}_bright_down.jpg")
                dark_img.save(output_img_path)
                
                # 处理标签文件（不需要坐标变换）
                if label_path:
                    process_label_file(image_path, label_path, output_dir, base_name, "bright_down", -1, label_output_dir)
                
                success_count += 1
                
            elif aug_type == "contrast_up":
                # 对比度增加
                contrast_img = apply_brightness_contrast(img, contrast_factor=1.5)
                output_img_path = os.path.join(output_dir, f"{base_name}_contrast_up.jpg")
                contrast_img.save(output_img_path)
                
                # 处理标签文件（不需要坐标变换）
                if label_path:
                    process_label_file(image_path, label_path, output_dir, base_name, "contrast_up", -1, label_output_dir)
                
                success_count += 1
                
            elif aug_type == "blur":
                # 模糊
                blur_img = apply_blur_sharpen(img, blur_radius=2)
                output_img_path = os.path.join(output_dir, f"{base_name}_blur.jpg")
                blur_img.save(output_img_path)
                
                # 处理标签文件（不需要坐标变换）
                if label_path:
                    process_label_file(image_path, label_path, output_dir, base_name, "blur", -1, label_output_dir)
                
                success_count += 1
                
            elif aug_type == "sharpen":
                # 锐化
                sharp_img = apply_blur_sharpen(img, sharpen_factor=2.0)
                output_img_path = os.path.join(output_dir, f"{base_name}_sharpen.jpg")
                sharp_img.save(output_img_path)
                
                # 处理标签文件（不需要坐标变换）
                if label_path:
                    process_label_file(image_path, label_path, output_dir, base_name, "sharpen", -1, label_output_dir)
                
                success_count += 1
                
            elif aug_type == "noise":
                # 添加噪声
                noisy_img = apply_noise(img, noise_factor=0.1)
                output_img_path = os.path.join(output_dir, f"{base_name}_noise.jpg")
                noisy_img.save(output_img_path)
                
                # 处理标签文件（不需要坐标变换）
                if label_path:
                    process_label_file(image_path, label_path, output_dir, base_name, "noise", -1, label_output_dir)
                
                success_count += 1
            
            # 更新进度
            if progress_callback:
                progress = (i + 1) / total_operations * 100
                progress_callback(progress)
                
        except Exception as e:
            print(f"处理增强类型 {aug_type} 时出错: {e}")
            continue
    
    print(f"成功处理 {success_count}/{total_operations} 个增强操作")
    return success_count > 0


def batch_augment_images(input_dir, output_dir, augmentations, progress_callback=None, label_output_dir=None):
    """批量处理文件夹中的图像"""
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print("未找到图像文件")
        return False
    
    total_files = len(image_files)
    success_count = 0
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(input_dir, image_file)
        
        # 执行增强
        success = augment_image_with_labels(
            image_path, output_dir, augmentations, 
            lambda p: progress_callback(p * (i + 1) / total_files) if progress_callback else None,
            label_output_dir
        )
        
        if success:
            success_count += 1
        
        # 更新总体进度
        if progress_callback:
            progress = (i + 1) / total_files * 100
            progress_callback(progress)
    
    print(f"批量处理完成: {success_count}/{total_files} 个文件成功")
    return success_count > 0


def augment_image_simple(image_path, output_dir, augmentations):
    """简单的图像增强（无标签文件）"""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        for aug_type in augmentations:
            if aug_type == "original":
                output_path = os.path.join(output_dir, f"{base_name}_original.jpg")
                img.save(output_path)
            elif aug_type == "flip_horizontal":
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                output_path = os.path.join(output_dir, f"{base_name}_flip_h.jpg")
                flipped_img.save(output_path)
            elif aug_type == "rotate_90":
                rotated_img = img.transpose(Image.ROTATE_90)
                output_path = os.path.join(output_dir, f"{base_name}_rot90.jpg")
                rotated_img.save(output_path)
            elif aug_type == "brightness_up":
                bright_img = apply_brightness_contrast(img, brightness_factor=1.3)
                output_path = os.path.join(output_dir, f"{base_name}_bright_up.jpg")
                bright_img.save(output_path)
            elif aug_type == "noise":
                noisy_img = apply_noise(img, noise_factor=0.1)
                output_path = os.path.join(output_dir, f"{base_name}_noise.jpg")
                noisy_img.save(output_path)
        
        return True
    except Exception as e:
        print(f"处理图像失败: {e}")
        return False


if __name__ == '__main__':
    # 测试代码
    augmentations = ["original", "flip_horizontal", "rotate_90", "brightness_up"]
    success = augment_image_with_labels(
        "test_image.jpg", 
        "output", 
        augmentations
    )
    print(f"增强完成: {success}")
