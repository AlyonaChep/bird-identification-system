from ultralytics import YOLO
import cv2

# Завантаження моделі (використовуємо найменшу для швидкості)
model = YOLO('../model/yolov8n.pt')  # або 'yolov8s.pt' для більш точного

# Відкриття відео або камери
cap = cv2.VideoCapture("../dataset/videosamples/blackbird_roof.mp4")  # або 0 для камери

# Спроба зчитати перший кадр
ret, frame = cap.read()
if not ret:
    print("Failed to read the frame")
    exit()

# Отримуємо розмір першого кадру
h, w = frame.shape[:2]
window_name = "Bird Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Максимальні розміри вікна
max_width = 1280
max_height = 720

# Обмежуємо розмір вікна, щоб воно не перевищувало 1280x720
w = min(w, max_width)
h = min(h, max_height)

cv2.resizeWindow(window_name, w, h)  # Встановлюємо початковий розмір вікна

# Повертаємо кадр назад, щоб обробити його у циклі
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()