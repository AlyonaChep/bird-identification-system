import os
from datetime import datetime
from pathlib import Path

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from src.config import VIDEO_OBS_DIR
from src.core.bird_detector import detect_birds
from src.core.frame_classifier import classify_bird


class VideoProcessorThread(QThread):
    log_signal = pyqtSignal(str)
    summary_signal = pyqtSignal(dict)
    progress_signal = pyqtSignal(int)  # від 0 до 100
    snapshot_dir_signal = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log_signal.emit("Failed to open video")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        filename = Path(self.video_path).stem
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{filename}_{timestamp}"
        output_dir = VIDEO_OBS_DIR / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_dir_signal.emit(str(output_dir))

        frame_interval = 5  # Аналізуємо кожен N-ий кадр
        MAX_GAP_FRAMES = 30  # Максимальна пауза для продовження "однієї появи"

        frame_count = 0

        # {class_name: [(start_frame, end_frame)]}
        bird_appearances = {}

        # Для збереження останніх кадрів та координат для кожного класу
        last_seen_frame = {}  # {class_name: last_seen_frame_number}
        last_seen_box = {}  # {class_name: (x1, y1, x2, y2)}

        while True:
            if self._abort:
                self.log_signal.emit("❌ Analysis aborted by user.")
                break  # виходимо з циклу

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
                        frame_id = f"{frame_count:04d}"
                        snapshot_name = f"{frame_id}_{predicted_class}.jpg"
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
