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

        # TOP частина (заголовок, зображення, результат)
        self.top_layout = QVBoxLayout()
        self.label = QLabel("📷 Bird identification by image")
        self.label.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.top_layout.addWidget(self.label)
        self.top_layout.addWidget(self.image_label)
        self.top_layout.addWidget(self.result_label)

        # FEEDBACK — створимо пізніше (динамічно)
        self.feedback_layout = None  # тимчасово

        # BOTTOM кнопки
        self.select_button = QPushButton("🔍 Select an image")
        self.process_button = QPushButton("🧠 Identify")
        self.back_button = QPushButton("← Back")

        self.select_button.clicked.connect(self.select_image)
        self.process_button.clicked.connect(self.process_image)
        self.back_button.clicked.connect(self.go_back)

        self.bottom_layout = QHBoxLayout()
        self.bottom_layout.addWidget(self.select_button)
        self.bottom_layout.addWidget(self.process_button)
        self.bottom_layout.addWidget(self.back_button)

        # Тепер збираємо все в головний layout
        self.layout.addLayout(self.top_layout)
        self.layout.addStretch()  # для розтяжки між топом і кнопками
        self.layout.addLayout(self.bottom_layout)

    def select_image(self):
        self.result_label.setText("")
        self.clear_thank_you_label()

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
        self.current_crop = bird_crop  # Збережемо для подальшої обробки

        predicted_class, conf = classify_bird(bird_crop)
        self.predicted_class = predicted_class

        self.show_image(bird_crop)
        self.result_label.setText(f"{predicted_class} ({conf:.2f})")

        self.show_feedback_options()

    def show_feedback_options(self):
        from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout

        self.clear_feedback_layout()
        self.clear_combo_layout()
        self.clear_thank_you_label()

        self.feedback_layout = QVBoxLayout()
        self.feedback_layout.addWidget(QLabel("Is it correct?"))

        self.yes_button = QPushButton("✅ Yes")
        self.no_button = QPushButton("❌ No")
        self.unsure_button = QPushButton("❓ Not sure")

        self.yes_button.clicked.connect(lambda: self.send_feedback("yes"))
        self.no_button.clicked.connect(self.show_class_selector)
        self.unsure_button.clicked.connect(lambda: self.send_feedback("unsure"))

        self.feedback_layout.addWidget(self.yes_button)
        self.feedback_layout.addWidget(self.no_button)
        self.feedback_layout.addWidget(self.unsure_button)

        self.layout.insertLayout(self.layout.count() - 2, self.feedback_layout)

    def send_feedback(self, user_choice, corrected_class=None):
        from src.feedback_handler import handle_user_feedback

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

        # Після відправки зворотного зв'язку відображаємо повідомлення "Thank you"
        self.show_thank_you_message()

    def show_class_selector(self):
        from PyQt5.QtWidgets import QComboBox
        from src.feedback_handler import get_bird_classes

        self.combo_layout = QHBoxLayout()
        self.class_selector = QComboBox()
        self.class_selector.addItems(get_bird_classes() + ["(Not in the list)"])

        self.confirm_other_button = QPushButton("Confirm")
        self.confirm_other_button.clicked.connect(self.confirm_corrected_class)

        self.combo_layout.addWidget(self.class_selector)
        self.combo_layout.addWidget(self.confirm_other_button)
        self.feedback_layout.addLayout(self.combo_layout)

    def confirm_corrected_class(self):
        selected_class = self.class_selector.currentText()
        if selected_class == "(Not in the list)":
            QMessageBox.information(self, "Info", "Image will not be saved.")
            return
        self.send_feedback("no", corrected_class=selected_class)

    def show_thank_you_message(self):
        self.clear_feedback_layout()
        self.clear_combo_layout()

        self.thank_you_label = QLabel("Thank you for your feedback! 🙏")
        self.thank_you_label.setAlignment(Qt.AlignCenter)
        self.thank_you_label.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")
        self.layout.insertWidget(self.layout.count() - 2, self.thank_you_label)

    def clear_feedback_layout(self):
        if self.feedback_layout:
            for i in reversed(range(self.feedback_layout.count())):
                widget = self.feedback_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            self.layout.removeItem(self.feedback_layout)
            self.feedback_layout = None

    def clear_combo_layout(self):
        if hasattr(self, 'combo_layout'):
            for i in reversed(range(self.combo_layout.count())):
                widget = self.combo_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            del self.combo_layout

    def clear_thank_you_label(self):
        if hasattr(self, 'thank_you_label'):
            self.layout.removeWidget(self.thank_you_label)
            self.thank_you_label.deleteLater()
            del self.thank_you_label

    def go_back(self):
        self.image = None
        self.image_label.clear()
        self.result_label.clear()

        self.clear_feedback_layout()
        self.clear_combo_layout()
        self.clear_thank_you_label()

        self.main_window.show_home()
