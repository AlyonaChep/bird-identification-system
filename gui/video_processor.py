import os
import cv2
import json
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal
from src.bird_detector import detect_birds
from src.frame_classifier import classify_bird


class VideoProcessorThread(QThread):
    log_signal = pyqtSignal(str)
    summary_signal = pyqtSignal(dict)
    progress_signal = pyqtSignal(int)  # від 0 до 100

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log_signal.emit("Failed to open video")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        filename = os.path.splitext(os.path.basename(self.video_path))[0]
        output_dir = os.path.join("snapshots", filename)
        os.makedirs(output_dir, exist_ok=True)

        frame_interval = 5  # Аналізуємо кожен N-ий кадр
        MAX_GAP_FRAMES = 30  # Максимальна пауза для продовження "однієї появи"

        frame_count = 0

        # {class_name: [(start_frame, end_frame)]}
        bird_appearances = {}

        # Для збереження останніх кадрів та координат для кожного класу
        last_seen_frame = {}  # {class_name: last_seen_frame_number}
        last_seen_box = {}    # {class_name: (x1, y1, x2, y2)}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                boxes = detect_birds(frame)
                current_classes = set()

                for (x1, y1, x2, y2, conf) in boxes:
                    bird_img = frame[y1:y2, x1:x2]
                    predicted_class, conf_score = classify_bird(bird_img)
                    current_classes.add(predicted_class)

                    intervals = bird_appearances.get(predicted_class, [])

                    if predicted_class in last_seen_frame:
                        gap = frame_count - last_seen_frame[predicted_class]
                    else:
                        gap = None

                    # Якщо є інтервал і не було довгої паузи - оновлюємо інтервал
                    if intervals and gap is not None and gap <= MAX_GAP_FRAMES:
                        # Оновлюємо кінець останнього інтервалу
                        start, _ = intervals[-1]
                        intervals[-1] = (start, frame_count)
                        bird_appearances[predicted_class] = intervals
                    else:
                        # Новий інтервал
                        intervals.append((frame_count, frame_count))
                        bird_appearances[predicted_class] = intervals
                        # Зберігаємо снапшот для нового інтервалу
                        snapshot_name = f"{frame_count}_{predicted_class}.jpg"
                        cv2.imwrite(os.path.join(output_dir, snapshot_name), bird_img)
                        self.log_signal.emit(f"[Frame {frame_count}] {predicted_class} appeared")

                    last_seen_frame[predicted_class] = frame_count
                    last_seen_box[predicted_class] = (x1, y1, x2, y2)

                # Перевірка, які пташки зникли (їх немає в цьому кадрі)
                for cls in list(last_seen_frame.keys()):
                    if cls not in current_classes:
                        # Якщо занадто довго не бачили, видаляємо з last_seen
                        if frame_count - last_seen_frame[cls] > MAX_GAP_FRAMES:
                            del last_seen_frame[cls]
                            del last_seen_box[cls]

            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            self.progress_signal.emit(progress)

        cap.release()
        self.log_signal.emit("--- Processing complete ---")

        # Формуємо підсумковий звіт
        summary = {cls: len(intervals) for cls, intervals in bird_appearances.items()}
        self.summary_signal.emit(summary)


class VideoProcessorViewer(QWidget):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("Video Analysis Log")
        self.setGeometry(200, 200, 600, 400)

        layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        layout.addWidget(QLabel(f"Analyzing: {os.path.basename(video_path)}"))
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_output)
        self.setLayout(layout)

        self.worker = VideoProcessorThread(video_path)
        self.worker.log_signal.connect(self.append_log)
        self.worker.summary_signal.connect(self.display_summary)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.start()

    def append_log(self, text):
        self.log_output.append(text)

    def display_summary(self, sightings):
        self.log_output.append("\nSummary:")
        for bird, count in sightings.items():
            self.log_output.append(f"- {bird}: {count} unique appearance(s)")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
