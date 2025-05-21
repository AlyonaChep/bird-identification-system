from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt


class HomeView(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("🕊 Bird Identification System")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)

        self.image_button = QPushButton("📷 Open Image")
        self.video_button = QPushButton("🎞 Open Video")
        self.camera_button = QPushButton("📡 Open Camera")

        self.image_button.clicked.connect(self.main_window.show_image_view)
        self.video_button.clicked.connect(self.main_window.show_video_view)
        self.camera_button.clicked.connect(self.main_window.show_camera_view)

        layout.addWidget(title)
        layout.addSpacing(40)
        layout.addWidget(self.image_button)
        layout.addWidget(self.video_button)
        layout.addWidget(self.camera_button)

        self.setLayout(layout)
