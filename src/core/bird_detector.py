from ultralytics import YOLO

from src.config import BASE_DIR

# Завантаження моделі YOLO
model = YOLO(BASE_DIR / "model" / "yolov8n.pt")


def detect_birds(frame):
    results = model(frame, verbose=False)[0]
    boxes = []

    # Збираємо координати пташок, якщо вони були знайдені
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])

        if class_name == "bird" and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2, conf))

    return boxes
