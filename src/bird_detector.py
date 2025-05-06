from ultralytics import YOLO
import cv2

# Завантаження моделі (використовуємо найменшу для швидкості)
model = YOLO('../model/yolov8n.pt')  # або 'yolov8s.pt' для більш точного

def detect_birds(frame):
    results = model(frame)[0]
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