# utils/app_window_utils.py

import pygetwindow as gw


def list_all_visible_apps():
    apps = []
    for w in gw.getAllWindows():
        title = w.title.strip()
        if title and w.visible:
            apps.append(title)
    return apps

def get_app_window_region(app_name_keyword):
    windows = gw.getWindowsWithTitle(app_name_keyword)
    if not windows:
        raise Exception(f"未找到标题包含 '{app_name_keyword}' 的窗口")
    win = windows[0]
    if not win.visible:
        raise Exception(f"窗口 '{win.title}' 当前不可见（可能已最小化）")
    return {
        "left": win.left,
        "top": win.top,
        "width": win.width,
        "height": win.height,
        "title": win.title
    }

def divide_region(app_region):
    left = app_region["left"]
    top = app_region["top"]
    w = app_region["width"]
    h = app_region["height"]

    region_width = w // 2
    region_height = h // 2

    return [
        {"left": left, "top": top, "width": region_width, "height": region_height, "id": 0},  # 左上
        {"left": left + region_width, "top": top, "width": region_width, "height": region_height, "id": 1},  # 右上
        {"left": left, "top": top + region_height, "width": region_width, "height": region_height, "id": 2},  # 左下
        {"left": left + region_width, "top": top + region_height, "width": region_width, "height": region_height, "id": 3},  # 右下
    ]