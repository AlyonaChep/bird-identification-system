from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from gui.ui_helpers import create_button, create_label


class HomeView(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        title = create_label("🕊 Bird Identification System", bold=True, size=22)

        self.image_button = create_button("📷 Open Image", self.main_window.show_image_view)
        self.video_button = create_button("🎥 Open Video", self.main_window.show_video_view)
        self.camera_button = create_button("📹 Open Live-camera", self.main_window.show_camera_view)
        self.archive_button = create_button("🗃️ Open Archive", self.main_window.show_archive_view)

        button_style = "font-size: 18px; padding: 10px;"
        for btn in [self.image_button, self.video_button, self.camera_button, self.archive_button]:
            btn.setStyleSheet(button_style)

        layout.addWidget(title)
        layout.addSpacing(40)
        layout.addWidget(self.image_button)
        layout.addWidget(self.video_button)
        layout.addWidget(self.camera_button)
        layout.addWidget(self.archive_button)

        self.setLayout(layout)
