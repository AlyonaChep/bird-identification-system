from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QListWidget, QListWidgetItem, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import QSize, Qt
import os
import cv2

from feedback_widget import FeedbackWidget
from src.feedback_handler import handle_user_feedback


class SnapshotReviewDialog(QDialog):
    def __init__(self, snapshot_dir):
        super().__init__()
        self.setWindowTitle("🖼️ Review Bird Snapshots")
        self.resize(800, 600)

        self.snapshot_dir = snapshot_dir
        self.layout = QHBoxLayout(self)

        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(100, 100))
        self.image_list.itemClicked.connect(self.load_image)
        self.layout.addWidget(self.image_list)

        self.right_panel = QVBoxLayout()
        self.image_label = QLabel("Select an image to review")
        self.image_label.setFixedSize(400, 300)
        self.image_label.setScaledContents(True)

        self.feedback_widget = FeedbackWidget()
        self.feedback_widget.feedback_given.connect(self.process_feedback)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        self.right_panel.addWidget(self.image_label)
        self.right_panel.addWidget(self.feedback_widget)
        self.right_panel.addWidget(self.status_label)

        self.layout.addLayout(self.right_panel)

        self.populate_image_list()

        self.current_image_path = None
        self.predicted_class = None

    def populate_image_list(self):
        self.image_list.clear()
        self.image_files = []

        for fname in os.listdir(self.snapshot_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")) and "_confirmed" not in fname:
                self.image_files.append(fname)
                item = QListWidgetItem(fname)
                pixmap = QPixmap(os.path.join(self.snapshot_dir, fname)).scaled(100, 100, Qt.KeepAspectRatio)
                icon = QIcon(pixmap)
                item.setIcon(icon)
                self.image_list.addItem(item)

    def load_image(self, item):
        image_path = os.path.join(self.snapshot_dir, item.text())
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        self.current_image_path = image_path

        parts = item.text().split("_")
        if len(parts) >= 2:
            self.predicted_class = parts[1].split(".")[0]
        else:
            self.predicted_class = "unknown"

        self.status_label.clear()

        if "_confirmed" in item.text():
            self.status_label.setText("✅ Already confirmed")
            self.feedback_widget.hide()
        else:
            self.status_label.clear()
            self.feedback_widget.show()

        self.feedback_widget.clear()

    def process_feedback(self, user_choice, corrected_class=None):
        frame_id = os.path.basename(self.current_image_path).split("_")[0]  # отримуємо номер фрейму з імені файлу
        video_name = os.path.basename(self.snapshot_dir)  # ім’я папки з відео

        context = {
            "source": "video",
            "video_name": video_name,
            "frame_id": frame_id
        }

        # Завантажуємо зображення з поточного шляху
        image = cv2.imread(self.current_image_path)

        path = handle_user_feedback(
            image=image,
            predicted_class=self.predicted_class,
            user_choice=user_choice,
            corrected_class=corrected_class,
            context=context
        )

        if path:
            new_filename = os.path.basename(path)
            self.image_list.currentItem().setText(new_filename)
            self.current_image_path = path
            self.predicted_class = new_filename.split("_")[1].split(".")[0]
            self.status_label.setText(f"✅ Saved: {new_filename}")
            self.feedback_widget.hide()
        else:
            self.status_label.setText("ℹ️ Result was not saved.")

        self.feedback_widget.clear()
