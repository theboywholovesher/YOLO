#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标签文件处理功能测试脚本
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strange_img import (
    detect_label_format, parse_xml, parse_yolo,
    transform_bbox, transform_yolo_bbox,
    create_xml, create_yolo_label
)


def test_label_format_detection():
    """测试标签格式检测"""
    print("=== 测试标签格式检测 ===")

    # 创建测试文件
    test_files = {
        'test.xml': '''<?xml version="1.0" encoding="UTF-8"?>
<annotation>
    <size>
        <width>800</width>
        <height>600</height>
    </size>
    <object>
        <name>person</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>200</ymin>
            <xmax>300</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
</annotation>''',

        'test.txt': '0 0.5 0.5 0.2 0.3\n1 0.7 0.8 0.1 0.2',

        'test.json': '{"images": [], "annotations": []}'
    }

    for filename, content in test_files.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        format_type = detect_label_format(filename)
        print(f"{filename}: {format_type}")

        # 清理测试文件
        os.remove(filename)

    print()


def test_xml_parsing():
    """测试XML解析"""
    print("=== 测试XML解析 ===")

    # 创建测试XML文件
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<annotation>
    <size>
        <width>800</width>
        <height>600</height>
    </size>
    <object>
        <name>person</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>200</ymin>
            <xmax>300</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
    <object>
        <name>car</name>
        <bndbox>
            <xmin>400</xmin>
            <ymin>100</ymin>
            <xmax>600</xmax>
            <ymax>300</ymax>
        </bndbox>
    </object>
</annotation>'''

    with open('test.xml', 'w', encoding='utf-8') as f:
        f.write(xml_content)

    width, height, objects = parse_xml('test.xml')
    print(f"图像尺寸: {width}x{height}")
    print(f"检测到 {len(objects)} 个对象:")

    for i, obj in enumerate(objects):
        print(f"  对象 {i + 1}: {obj['name']} - 边界框: {obj['bndbox']}")

    # 清理测试文件
    os.remove('test.xml')
    print()


def test_yolo_parsing():
    """测试YOLO格式解析"""
    print("=== 测试YOLO格式解析 ===")

    # 创建测试YOLO文件
    yolo_content = '''0 0.5 0.5 0.2 0.3
1 0.7 0.8 0.1 0.2
2 0.3 0.4 0.15 0.25'''

    with open('test.txt', 'w') as f:
        f.write(yolo_content)

    objects = parse_yolo('test.txt', 800, 600)
    print(f"检测到 {len(objects)} 个对象:")

    for i, obj in enumerate(objects):
        bbox = obj['bbox']
        print(f"  对象 {i + 1}: 类别ID {obj['class_id']} - 边界框: {bbox}")

    # 清理测试文件
    os.remove('test.txt')
    print()


def test_bbox_transformation():
    """测试边界框坐标变换"""
    print("=== 测试边界框坐标变换 ===")

    # 测试数据
    original_bbox = [100, 200, 300, 400]  # [xmin, ymin, xmax, ymax]
    width, height = 800, 600

    print(f"原始边界框: {original_bbox}")
    print(f"图像尺寸: {width}x{height}")
    print()

    # 测试不同的变换
    transformations = [
        (0, "左右翻转"),
        (1, "上下翻转"),
        (2, "旋转90°"),
        (3, "旋转180°"),
        (4, "旋转270°")
    ]

    for trans_type, desc in transformations:
        new_bbox = transform_bbox(original_bbox, width, height, trans_type)
        print(f"{desc}: {new_bbox}")

    print()


def test_yolo_bbox_transformation():
    """测试YOLO格式边界框变换"""
    print("=== 测试YOLO格式边界框变换 ===")

    # 测试数据 (归一化坐标)
    x_center, y_center, w, h = 0.5, 0.5, 0.2, 0.3
    img_width, img_height = 800, 600

    print(f"原始YOLO坐标: 中心({x_center:.2f}, {y_center:.2f}), 宽高({w:.2f}, {h:.2f})")
    print(f"图像尺寸: {img_width}x{img_height}")
    print()

    # 测试不同的变换
    transformations = [
        (0, "左右翻转"),
        (1, "上下翻转"),
        (2, "旋转90°"),
        (3, "旋转180°"),
        (4, "旋转270°")
    ]

    for trans_type, desc in transformations:
        new_bbox = transform_yolo_bbox(x_center, y_center, w, h, img_width, img_height, trans_type)
        print(f"{desc}: 中心({new_bbox[0]:.4f}, {new_bbox[1]:.4f}), 宽高({new_bbox[2]:.4f}, {new_bbox[3]:.4f})")

    print()


def test_label_file_creation():
    """测试标签文件创建"""
    print("=== 测试标签文件创建 ===")

    # 创建测试XML文件
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<annotation>
    <size>
        <width>800</width>
        <height>600</height>
    </size>
    <object>
        <name>person</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>200</ymin>
            <xmax>300</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
</annotation>'''

    with open('test.xml', 'w', encoding='utf-8') as f:
        f.write(xml_content)

    # 测试XML文件创建
    width, height, objects = parse_xml('test.xml')
    if width is not None:
        success = create_xml('test.xml', 'test_output.xml', width, height, objects, 0)  # 左右翻转
        print(f"XML文件创建: {'成功' if success else '失败'}")

    # 测试YOLO标签文件创建
    yolo_objects = [
        {'class_id': 0, 'bbox': [0.5, 0.5, 0.2, 0.3]},
        {'class_id': 1, 'bbox': [0.7, 0.8, 0.1, 0.2]}
    ]

    success = create_yolo_label('test.txt', 'test_output.txt', 800, 600, yolo_objects, 0)  # 左右翻转
    print(f"YOLO标签文件创建: {'成功' if success else '失败'}")

    # 清理测试文件
    for filename in ['test.xml', 'test_output.xml', 'test_output.txt']:
        if os.path.exists(filename):
            os.remove(filename)

    print()


def main():
    """主函数"""
    print("标签文件处理功能测试")
    print("=" * 50)

    try:
        test_label_format_detection()
        test_xml_parsing()
        test_yolo_parsing()
        test_bbox_transformation()
        test_yolo_bbox_transformation()
        test_label_file_creation()

        print("所有测试完成！")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
