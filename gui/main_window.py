import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2


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
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Pick a video", "", "Videos (*.mp4 *.avi)")
        if file_path:
            print("Picked video:", file_path)
            # Тут буде логіка обробки відео

    def open_camera(self):
        print("Loading camera ...")
        # Тут буде логіка для обробки потоку з камери


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
