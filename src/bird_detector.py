from ultralytics import YOLO
import cv2

# Завантаження моделі (використовуємо найменшу для швидкості)
model = YOLO('../model/yolov8n.pt')  # або 'yolov8s.pt' для більш точного

# Відкриття відео або камери
cap = cv2.VideoCapture("../dataset/videosamples/blackbird_roof.mp4")  # або 0 для камери

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Детекція
    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])

        if class_name == "bird" and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Показ результату
    cv2.imshow("Bird Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
