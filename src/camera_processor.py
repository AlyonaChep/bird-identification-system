import cv2
import time
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from src.bird_detector import detect_birds
from src.frame_classifier import classify_bird


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
            self.log_signal.emit("❌ Failed to access the camera.")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = detect_birds(frame)
            for (x1, y1, x2, y2, _) in boxes:
                bird_img = frame[y1:y2, x1:x2]
                predicted_class, conf = classify_bird(bird_img)

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
