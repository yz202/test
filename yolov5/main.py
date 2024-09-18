import argparse
import sys
from mainwindow import Ui_Form
import cv2
# import mydetect
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QDir, QTimer
from PyQt5.QtGui import QPixmap, QImage
import os
import platform
import sys
from pathlib import Path
import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)


def draw_boxes(img, detections, class_names):
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        label = f"{class_names[int(cls)]}: {conf:.2f}"
        color = (255, 0, 0)  # Blue color for the bounding box
        thickness = 2  # Thickness of the bounding box

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)

    return img


def process_frame(frame, model, conf_thres, iou_thres, classes, agnostic_nms, max_det, names, line_thickness=3,
                  augment=False,
                  visualize=False):
    im0s = frame.copy()
    im0 = frame.astype(float) / 255.0  # Normalize pixel values to range 0-1
    im = torch.from_numpy(im0).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im = im.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        pred = model(im, augment=augment, visualize=visualize)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    annotator = Annotator(im0s, line_width=line_thickness, example=str(names))

    # Process predictions
    for i, det in enumerate(pred):  # per image

        s = f'{i}: %gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                print(f"Detected {n} {names[int(c)]}{'s' * (n > 1)}")
        im0s = draw_boxes(im0s, det, names)

    return im0s


# 定义摄像头类
class CamConfig:
    def __init__(self):
        # 设置时钟
        self.v_timer = QTimer()
        # 打开摄像头
        self.cap = cv2.VideoCapture(1)
        # 设置定时器周期，单位毫秒
        self.v_timer.start(20)
        # 连接定时器周期溢出的槽函数，用于显示一帧视
        self.v_timer.timeout.connect(self.show_pic)
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.view_img = True  # show results
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.update = False  # update all models
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.vid_stride = 1  # video frame-rate stride
        self.model = DetectMultiBackend('./runs/train/exp12/weights/best.pt', device='CUDA', dnn=False, data='./data/myvoc.yaml', fp16=False)
    def show_pic(self):

        # 读取摄像头的一帧画面
        success, frame = self.cap.read()
        frame = cv2.flip(frame, 1)  # 镜像处理
        if success:
            # 检测
            # 将摄像头读到的frame传入检测函数mydetect.predict()
            frame = process_frame(frame, self.model, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                  self.max_det, self.model.names)
            # 将画面显示在前端UI上
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            window.label.setPixmap(QPixmap.fromImage(showImage))


def CamConfig_init():
    window.f_type = CamConfig()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    CamConfig_init()
    sys.exit(app.exec_())
