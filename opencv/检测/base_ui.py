import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog
from PySide6.QtGui import QPixmap, QImage
from main_window_ui import Ui_MainWindow  # 确保文件名和类名正确

# 加载data数据集的类别名称
classNames = []
classFile = 'data.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# 加载并配置检测模型
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # 初始化UI
        self.bind_slots()  # 绑定信号槽

    def image_pred(self, file_path):
        image = cv2.imread(file_path)
        if image is None:
            print("Error: Image not loaded.")
            return
        if image.ndim == 2:  # 检查是否为灰度图像
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        classIds, confs, bbox = net.detect(image, confThreshold=0.45)
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(image, box, color=(0, 255, 0), thickness=2)
            cv2.putText(image, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.output.setPixmap(QPixmap.fromImage(qImg))

    def open_image(self):
        print("点击了检测图片")
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "./image", "Image Files (*.jpg *.png *.jpeg)")
        if file_paths:  # 如果选择了文件
            file_path = file_paths[0]  # 获取第一个文件路径
            self.input.setPixmap(QPixmap(file_path))  # 假设self.input是一个QLabel
            self.image_pred(file_path)

    def video_pred(self):
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            if not success:
                break
            if img.ndim == 2:  # 检查是否为灰度图像
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            classIds, confs, bbox = net.detect(img, confThreshold=0.45)
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            self.output.setPixmap(QPixmap.fromImage(qImg))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    def open_video(self):
        print("点击了检测视频")
        self.video_pred()

    def bind_slots(self):
        # 假设你的UI中有名为det_image和det_video的QPushButton
        self.det_image.clicked.connect(self.open_image)
        self.det_video.clicked.connect(self.open_video)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())