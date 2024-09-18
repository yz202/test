import sys
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog
from PySide6.QtGui import QPixmap
from main_window_ui import Ui_MainWindow  # 确保文件名和类名正确


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # 初始化UI
        self.bind_slots()  # 绑定信号槽

    def image_pred(self,file_path):
        pass
    def open_image(self):
        print("点击了检测图片")
        # 使用QFileDialog打开文件对话框
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "./image", "Image Files (*.jpg *.png *.jpeg)")
        if file_paths:  # 如果选择了文件
            file_path = file_paths[0]  # 获取第一个文件路径
            self.input.setPixmap(QPixmap(file_path))  # 假设self.input是一个QLabel

    def open_video(self):
        pass

    def bind_slots(self):
        # 假设你的UI中有名为det_image和det_video的QPushButton
        self.det_image.clicked.connect(self.open_image)
        self.det_video.clicked.connect(self.open_video)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())  # 使用sys.exit确保程序正常退出