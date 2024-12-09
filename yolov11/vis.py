import sys
import argparse
import cv2
import os
import time
from ultralytics import YOLO
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# Parse arguments for detection
parser = argparse.ArgumentParser()
parser.add_argument('--weights', default=r"C:\Users\gg\Documents\WeChat Files\wxid_xir3gkublyb522\FileStorage\File\2024-12\best.pt", type=str, help='weights path')
parser.add_argument('--conf_thre', type=float, default=0.2, help='conf_thre')
parser.add_argument('--iou_thre', type=float, default=0.2, help='iou_thre')
opt = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

class Detector(object):
    def __init__(self, weight_path, conf_threshold=0.5, iou_threshold=0.5):
        self.device = device
        self.model = YOLO(weight_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.names = self.model.names

    def detect_image(self, img_bgr):
        results = self.model(img_bgr, verbose=True, conf=self.conf_threshold,
                             iou=self.iou_threshold, device=self.device)
        bboxes_cls = results[0].boxes.cls
        bboxes_conf = results[0].boxes.conf
        bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
        for idx in range(len(bboxes_cls)):
            box_conf = f"{bboxes_conf[idx]:.2f}"
            box_cls = int(bboxes_cls[idx])
            bbox_xyxy = bboxes_xyxy[idx]
            bbox_label = self.names[box_cls]
            xmin, ymin, xmax, ymax = bbox_xyxy
            img_bgr = cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), get_color(box_cls + 3), 2)
            cv2.putText(img_bgr, f'{str(bbox_label)}/{str(box_conf)}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color(box_cls + 3), 2)
        return img_bgr

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("YOLO Detector")
        self.setGeometry(100, 100, 800, 600)
        self.detector = Detector(weight_path=opt.weights, conf_threshold=opt.conf_thre, iou_threshold=opt.iou_thre)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.select_image_button = QPushButton("Select Image", self)
        self.select_image_button.clicked.connect(self.open_image)

        self.video_button = QPushButton("Start Video Detection", self)
        self.video_button.clicked.connect(self.toggle_video_detection)
        self.video_active = False

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.select_image_button)
        layout.addWidget(self.video_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.xpm *.jpg)", options=options)
        if file_name:
            img_bgr = cv2.imread(file_name)
            img_bgr = self.detector.detect_image(img_bgr)
            self.display_image(img_bgr)

    def toggle_video_detection(self):
        if self.video_active:
            self.video_active = False
            self.video_button.setText("Start Video Detection")
        else:
            self.video_active = True
            self.video_button.setText("Stop Video Detection")
            self.detect_video()

    def detect_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi *.mov)", options=options)
        if file_name:
            cap = cv2.VideoCapture(file_name)
            while self.video_active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.detector.detect_image(frame)
                self.display_image(frame)
                QApplication.processEvents()
                time.sleep(0.03)
            cap.release()

    def display_image(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
