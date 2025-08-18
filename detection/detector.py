import cv2
import numpy as np
import mss

from ultralytics import YOLO


def detect_region(region, shared_list, i, model_path="yolov8n.pt"):
    global sct
    try:
        model = YOLO(model_path)
        names = model.names
        monitor = {
            "left": region["left"],
            "top": region["top"],
            "width": region["width"],
            "height": region["height"]
        }

        sct = mss.mss()

        while True:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            results = model(frame, conf=0.5, verbose=False, device=0)
            shared_list[i][:] = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = names[cls]
                abs_x1 = region["left"] + x1
                abs_y1 = region["top"] + y1
                abs_x2 = region["left"] + x2
                abs_y2 = region["top"] + y2
                res = [abs_x1, abs_y1, abs_x2, abs_y2, conf, cls_name, region["id"],monitor]
                shared_list[i].append(res)

    except Exception as e:
        print(f"[子进程] 检测出错：{e}")
    finally:
        sct.close()


