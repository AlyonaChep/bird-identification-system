import os

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QSizePolicy

from gui.ui_helpers import create_button, create_label
from gui.widgets.feedback_widget import FeedbackWidget
from src.core.bird_detector import detect_birds
from src.core.frame_classifier import classify_bird
from src.feedback_handler import handle_user_feedback


class ImageView(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.image_path = None
        self.image = None
        self.predicted_class = None
        self.current_crop = None

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.label = create_label("📷 Bird identification by image", bold=True, size=16)
        self.path_label = create_label("", center=True)
        self.image_label = create_label("", center=True)
        self.result_label = create_label("", bold=True)

        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)

        btn_layout = QHBoxLayout()
        self.select_button = create_button("🔍 Select an image", self.select_image)
        self.process_button = create_button("🧠 Identify", self.process_image)
        self.back_button = create_button("⬅️ Back", self.go_back)

        btn_layout.addWidget(self.select_button)
        btn_layout.addWidget(self.process_button)
        btn_layout.addWidget(self.back_button)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.path_label)
        self.layout.addLayout(btn_layout)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.result_label)

        self.feedback_widget = None
        self.thank_you_label = None

    def select_image(self):
        self.path_label.setText("")
        self.result_label.setText("")
        self.clear_feedback()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path)
            filename = os.path.basename(file_path)
            self.path_label.setText(f"Selected photo: {filename}")
            self.show_image(self.image)
            self.result_label.setText("Selected photo. Ready for identification")

    def show_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        max_height = 350
        if h > max_height:
            # Обчислюємо пропорційно нову ширину, щоб зберегти співвідношення сторін
            new_height = max_height
            new_width = int(w * max_height / h)
        else:
            new_width = w
            new_height = h

        scaled_pixmap = pixmap.scaled(
            new_width,
            new_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def process_image(self):
        if self.image is None:
            QMessageBox.warning(self, "Error", "Please select an image first.")
            return

        image_copy = self.image.copy()
        boxes = detect_birds(image_copy)
        if not boxes:
            self.result_label.setText("No birds were found.")
            return

        x1, y1, x2, y2, _ = boxes[0]
        crop = image_copy[y1:y2, x1:x2]
        self.current_crop = crop
        pred, conf = classify_bird(crop)
        self.predicted_class = pred

        self.show_image(crop)
        self.result_label.setText(f"{pred} ({conf:.2f})")
        self.show_feedback()

    def show_feedback(self):
        self.clear_feedback()
        self.feedback_widget = FeedbackWidget()
        self.feedback_widget.feedback_given.connect(self.handle_feedback)
        index = self.layout.indexOf(self.result_label)
        self.layout.insertWidget(index + 1, self.feedback_widget)

    def handle_feedback(self, user_choice, corrected_class=None):
        path = handle_user_feedback(
            image=self.current_crop,
            predicted_class=self.predicted_class,
            user_choice=user_choice,
            corrected_class=corrected_class
        )
        if path:
            self.result_label.setText(f"✅ Saved: {os.path.basename(path)}")
        else:
            self.result_label.setText("ℹ️ Result was not saved.")
        self.show_thank_you()

    def show_thank_you(self):
        self.clear_feedback()
        self.thank_you_label = create_label("Thank you for your feedback! 🙏", bold=True, size=16, color="green")
        index = self.layout.indexOf(self.result_label)
        self.layout.insertWidget(index + 1, self.thank_you_label)

    def clear_feedback(self):
        if self.feedback_widget:
            self.feedback_widget.setParent(None)
            self.feedback_widget = None
        if self.thank_you_label:
            self.layout.removeWidget(self.thank_you_label)
            self.thank_you_label.deleteLater()
            self.thank_you_label = None

    def go_back(self):
        self.image = None
        self.image_label.clear()
        self.path_label.clear()
        self.result_label.clear()
        self.clear_feedback()
        self.main_window.show_home()