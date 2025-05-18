import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QHBoxLayout, QMessageBox, QInputDialog
)
from PyQt5.QtCore import Qt
from pygrabber.dshow_graph import FilterGraph
from gui.static_image_viewer import StaticImageViewer
from gui.video_processor import VideoProcessorViewer
from gui.camera_viewer import CameraViewer


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bird Identification System")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        # Кнопки
        self.button_image = QPushButton("Open image")
        self.button_video = QPushButton("Open video")
        self.button_camera = QPushButton("Open camera")

        self.button_image.clicked.connect(self.open_image)
        self.button_video.clicked.connect(self.open_video)
        self.button_camera.clicked.connect(self.open_camera)

        # Вікно для показу зображення
        self.image_label = QLabel("Image will be here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black")

        # Додаємо до layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button_image)
        button_layout.addWidget(self.button_video)
        button_layout.addWidget(self.button_camera)

        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Pick an image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.viewer = StaticImageViewer(file_path)
            self.viewer.show()

    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Pick a video", "", "Videos (*.mp4 *.avi)")
        if file_path:
            print("Picked video:", file_path)
            self.video_viewer = VideoProcessorViewer(file_path)
            self.video_viewer.show()

    def open_camera(self):
        graph = FilterGraph()
        device_names = graph.get_input_devices()

        if not device_names:
            print("No camera devices found")
            return

        # Діалог вибору
        item, ok = QInputDialog.getItem(
            self,
            "Select Camera",
            "Available Cameras:",
            device_names,
            0,
            False
        )

        if ok and item:
            camera_index = device_names.index(item)
            self.camera_viewer = CameraViewer(camera_index)
            self.camera_viewer.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
