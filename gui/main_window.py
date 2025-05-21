import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QStackedLayout, QVBoxLayout
)
from gui.home_view import HomeView
from gui.image_view import ImageView
from gui.video_view import VideoView
from gui.camera_view import CameraView


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bird Identification System")
        self.setGeometry(100, 100, 900, 700)

        self.layout = QVBoxLayout()
        self.stack = QStackedLayout()

        # ініціалізуємо всі сторінки
        self.home_view = HomeView(self)
        self.image_view = ImageView(self)
        self.video_view = VideoView(self)
        self.camera_view = CameraView(self)

        # додаємо до стеку
        self.stack.addWidget(self.home_view)
        self.stack.addWidget(self.image_view)
        self.stack.addWidget(self.video_view)
        self.stack.addWidget(self.camera_view)

        self.layout.addLayout(self.stack)
        self.setLayout(self.layout)

        self.show_home()

    def show_home(self):
        self.stack.setCurrentWidget(self.home_view)

    def show_image_view(self):
        self.stack.setCurrentWidget(self.image_view)

    def show_video_view(self):
        self.stack.setCurrentWidget(self.video_view)

    def show_camera_view(self):
        self.stack.setCurrentWidget(self.camera_view)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
