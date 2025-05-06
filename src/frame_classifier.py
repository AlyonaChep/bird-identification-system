import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model

# Завантажуємо модель класифікації
classification_model = load_model('../model/bird_identification_model.h5')

# Завантажуємо мапу класів
with open('../model/class_names.json') as f:
    index_to_class = json.load(f)


def classify_bird(bird_image):
    # Підготовка зображення
    bird_image_resized = cv2.resize(bird_image, (224, 224))
    bird_image_resized = np.expand_dims(bird_image_resized, axis=0)  # [1, 224, 224, 3]
    bird_image_resized = bird_image_resized / 255.0  # нормалізація до [0, 1]

    # Передбачення
    predictions = classification_model.predict(bird_image_resized)
    predicted_index = int(np.argmax(predictions))
    predicted_class = index_to_class[str(predicted_index)]
    confidence = float(np.max(predictions))  # Ймовірність найбільш впевненого класу

    return predicted_class, confidence