from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout, QMessageBox, QComboBox, QSizePolicy
)
from pygrabber.dshow_graph import FilterGraph

from src.core.camera_processor import CameraProcessorThread
from gui.ui_helpers import create_label, create_button


class CameraView(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("📹 Live Bird Detection")

        self.camera_running = False
        graph = FilterGraph()
        device_names = graph.get_input_devices()  # список назв камер
        if not device_names:
            QMessageBox.warning(self, "Warning", "No camera devices found")
            device_names = ["No cameras found"]

        self.worker = None

        self.label = create_label("📹 Real-time bird identification", bold=True, size=16)
        self.label.setAlignment(Qt.AlignCenter)

        self.image_label = create_label("Camera not started", bold=True, size=18)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ccc;")
        self.image_label.setScaledContents(False)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setMaximumSize(1280, 720)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.log_output = QTextEdit()
        self.log_output.setStyleSheet("font-size: 16px;")
        self.log_output.clear()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Bird sightings will appear here...")

        self.camera_selector = QComboBox()
        self.camera_selector.addItems(device_names)
        self.camera_selector.setCurrentIndex(0)
        self.camera_selector.currentIndexChanged.connect(self.change_camera)

        self.start_stop_button = create_button("▶️ Start", self.toggle_camera)
        self.back_button = create_button("⬅️ Back", self.go_back)

        button_style = "font-size: 16px; padding: 8px 16px;"
        self.start_stop_button.setStyleSheet(button_style)
        self.back_button.setStyleSheet(button_style)

        combo_style = "font-size: 16px; padding: 7px;"
        self.camera_selector.setStyleSheet(combo_style)

        top_layout = QHBoxLayout()
        label_camera = QLabel("Select Camera:")
        label_camera.setStyleSheet("font-size: 16px;")
        top_layout.addWidget(label_camera)
        top_layout.addWidget(self.camera_selector)
        top_layout.addWidget(self.start_stop_button)
        top_layout.addWidget(self.back_button)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(top_layout)
        layout.addWidget(self.image_label)
        layout.addWidget(self.log_output)
        self.setLayout(layout)

    def start_camera(self):
        if self.worker and self.worker.isRunning():
            return

        self.camera_running = True
        self.image_label.setText("Starting the camera...")
        self.worker = CameraProcessorThread(self.camera_index)
        self.worker.frame_signal.connect(self.update_frame)
        self.worker.log_signal.connect(self.append_log)
        self.worker.start()
        self.start_stop_button.setText("⏹️ Stop")
        self.log_output.append(f"Camera {self.camera_selector.currentText()} started.")

    def stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

        self.camera_running = False
        self.start_stop_button.setText("▶️ Start")
        self.log_output.append("Camera stopped.")

        self.image_label.clear()
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("Camera not started")

    def toggle_camera(self):
        if self.worker and self.worker.isRunning():
            self.stop_camera()
        else:
            self.start_camera()

    def change_camera(self, index):
        self.camera_index = index

        if self.worker and self.worker.isRunning():
            self.stop_camera()

        self.log_output.append(f"Camera changed to {self.camera_selector.currentText()}")

        # Автоматично запускати нову камеру після вибору
        self.start_camera()

    def update_frame(self, qt_image):
        if not self.camera_running:
            return  # Ігнорувати кадри, коли камера не працює

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),  # поточний розмір QLabel
            Qt.KeepAspectRatio  # зберігати пропорції
        )
        self.image_label.setPixmap(scaled_pixmap)

    def append_log(self, message):
        self.log_output.append(message)

    def go_back(self):
        self.stop_camera()
        self.log_output.clear()  # Очистити всі попередні логи
        self.image_label.clear()
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("Camera not started")
        self.main_window.show_home()

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()
