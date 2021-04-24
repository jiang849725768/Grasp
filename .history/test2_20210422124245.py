
import time
 
import threading as th
from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, Signal, Slot)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient, QImage)
from PySide2.QtWidgets import *
import pyrealsense2 as rs
import numpy as np
 
from realsense2_acquisition import Ui_MainWindow
 
import sys
 
DELAY = 0
 
 
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
 
        self.dis_update.connect(self.camera_view)
 
    # 在对应的页面类的内部，与def定义的函数同级
    dis_update = Signal(QPixmap)
 
    def updateStatusBar(self, str):
        self.ui.statusbar.showMessage(str)
 
    def open_realsense(self):
        print('open_realsense')
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
 
        profile = pipeline.start(config)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # Color Intrinsics
        intr = color_frame.profile.as_video_stream_profile().intrinsics
 
        align_to = rs.stream.color
        align = rs.align(align_to)
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
 
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue
            c = np.asanyarray(color_frame.get_data())
            qimage = QImage(c, 1280, 720, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimage)
            self.dis_update.emit(pixmap)
            # print('data\r\n')
            time.sleep(DELAY)
        pipeline.stop()
 
    def open_camera(self):
        # target选择开启摄像头的函数
        t = th.Thread(target=self.open_realsense)
        t.start()
        print('Open Camera')
 
    def camera_view(self, c):
        # 调用setPixmap函数设置显示Pixmap
        self.ui.label.setPixmap(c)
        # 调用setScaledContents将图像比例化显示在QLabel上
        self.ui.label.setScaledContents(True)
 
 
if __name__ == "__main__":
    app = QApplication([])
    widget = MainWindow()
    # widget.showFullScreen()
    widget.updateStatusBar("系统已准备,  Facial feature calculation version: 1.0")
    widget.show()
    widget.open_camera()
