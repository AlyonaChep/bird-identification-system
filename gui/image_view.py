import os
import cv2
import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from src.bird_detector import detect_birds
from src.frame_classifier import classify_bird


class ImageView(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.image_path = None
        self.image = None

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("📷 Bird identification by image")
        self.label.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.select_button = QPushButton("🔍 Select an image")
        self.process_button = QPushButton("🧠 Identify")
        self.back_button = QPushButton("← Back")

        self.select_button.clicked.connect(self.select_image)
        self.process_button.clicked.connect(self.process_image)
        self.back_button.clicked.connect(self.go_back)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.back_button)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.result_label)
        self.layout.addLayout(button_layout)

    def select_image(self):
        self.result_label.setText("")
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path)
            self.show_image(self.image)
            self.result_label.setText("Selected photo. Ready for identification")

    def show_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))

    def process_image(self):
        if self.image is None:
            QMessageBox.warning(self, "Error", "Please select an image first.")
            return

        image_copy = self.image.copy()
        boxes = detect_birds(self.image)

        if not boxes:
            self.result_label.setText("No birds were found.")
            return

        x1, y1, x2, y2, _ = boxes[0]
        bird_crop = image_copy[y1:y2, x1:x2]
        predicted_class, conf = classify_bird(bird_crop)

        self.show_image(bird_crop)
        self.result_label.setText(f"{predicted_class} ({conf:.2f})")

        # Зробити квадратну область
        h, w, _ = bird_crop.shape
        size = max(h, w)
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left
        square_crop = cv2.copyMakeBorder(bird_crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Зберігаємо результат
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = "../dataset/observations"
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, f"{predicted_class}_{now}.jpg")
        cv2.imwrite(save_path, square_crop)

    def go_back(self):
        self.image = None
        self.image_label.clear()
        self.result_label.clear()
        self.main_window.show_home()
