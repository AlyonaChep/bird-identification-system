import os
import cv2
from src.bird_detector import detect_birds
from src.frame_classifier import classify_bird


def process_video(video_path, snapshot_interval=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video.")
        return

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("snapshots", base_name)
    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    saved_snapshots = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_birds(frame)

        if boxes:
            if frame_idx % snapshot_interval == 0:
                for i, (x1, y1, x2, y2, _) in enumerate(boxes):
                    bird_crop = frame[y1:y2, x1:x2]
                    predicted_class, conf = classify_bird(bird_crop)

                    filename = f"{predicted_class}_{frame_idx}_{i}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, bird_crop)
                    print(f"Saved snapshot: {filepath}")
                    saved_snapshots += 1

        frame_idx += 1

    cap.release()
    print(f"Finished. Total snapshots saved: {saved_snapshots}")
