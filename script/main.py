import cv2
from src.bird_detector import detect_birds
from src.frame_classifier import classify_bird

# Відкриваємо відео
cap = cv2.VideoCapture("../dataset/videosamples/blackbird_roof.mp4")

# Спроба зчитати перший кадр
ret, frame = cap.read()
if not ret:
    print("Failed to read the frame")
    exit()

# Отримуємо розмір першого кадру
h, w = frame.shape[:2]
window_name = "Bird Identification"
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

    # Детекція пташок на кадрі
    boxes = detect_birds(frame)

    # Класифікація кожної пташки
    for (x1, y1, x2, y2, conf) in boxes:
        bird_image = frame[y1:y2, x1:x2]

        # Класифікація пташки
        predicted_class, classification_conf = classify_bird(bird_image)

        # Виведення результату на екран
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{predicted_class} ({classification_conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Відображення результату
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
