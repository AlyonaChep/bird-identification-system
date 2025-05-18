import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt5.QtGui import QPixmap
from src.bird_detector import detect_birds
from src.frame_classifier import classify_bird
import time


class CameraProcessorThread(QThread):
    frame_signal = pyqtSignal(QImage)
    log_signal = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.last_seen = {}  # {class_name: (timestamp, (x1, y1))}

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.log_signal.emit("Failed to access the camera")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = detect_birds(frame)
            for (x1, y1, x2, y2, _) in boxes:
                bird_img = frame[y1:y2, x1:x2]
                predicted_class, conf = classify_bird(bird_img)

                # Простий фільтр на повторне виявлення
                current_time = time.time()
                seen = False
                if predicted_class in self.last_seen:
                    last_time, (lx, ly) = self.last_seen[predicted_class]
                    if current_time - last_time < 5 and abs(x1 - lx) < 30 and abs(y1 - ly) < 30:
                        seen = True

                if not seen:
                    self.last_seen[predicted_class] = (current_time, (x1, y1))
                    self.log_signal.emit(f"🕊️ {predicted_class} ({conf:.2f}) spotted")

                label = f"{predicted_class} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_signal.emit(qt_image)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class CameraViewer(QWidget):
    def __init__(self, camera_index=0):
        super().__init__()
        self.setWindowTitle("Live Camera Bird Detection")
        self.setGeometry(200, 200, 800, 600)

        self.image_label = QLabel("Starting camera...")
        self.image_label.setStyleSheet("border: 1px solid black")
        self.image_label.setScaledContents(True)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Bird sightings will appear here...")

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.log_output)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        self.worker = CameraProcessorThread(camera_index)
        self.worker.frame_signal.connect(self.update_frame)
        self.worker.log_signal.connect(self.append_log)
        self.worker.start()

    def update_frame(self, qt_image):
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def append_log(self, message):
        self.log_output.append(message)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()