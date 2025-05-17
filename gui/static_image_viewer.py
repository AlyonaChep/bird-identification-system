import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from src.bird_detector import detect_birds
from src.frame_classifier import classify_bird


class StaticImageViewer(QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Result of image classification")
        self.setGeometry(200, 200, 600, 600)

        self.layout = QVBoxLayout()
        self.image_label = QLabel()
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.result_label)
        self.setLayout(self.layout)

        self.process_image(image_path)

    def process_image(self, path):
        image = cv2.imread(path)
        original_image = image.copy()

        boxes = detect_birds(image)

        if not boxes:
            self.result_label.setText("No bird detected.")
            return

        # Беремо першу пташку (можна обробляти всі, якщо потрібно)
        x1, y1, x2, y2, _ = boxes[0]
        bird_crop = original_image[y1:y2, x1:x2]
        predicted_class, conf = classify_bird(bird_crop)

        # Показуємо обрізане зображення пташки
        rgb_image = cv2.cvtColor(bird_crop, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

        # Текст під фото
        self.result_label.setText(f"{predicted_class} ({conf:.2f})")
