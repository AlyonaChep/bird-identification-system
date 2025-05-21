import os
import cv2
import datetime
import json


def save_image(image, class_name, folder="../dataset/observations"):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if class_name.lower() == "unknown":
        folder = os.path.join(folder, "unknown")
    os.makedirs(folder, exist_ok=True)
    filename = f"{class_name}_{now}.jpg"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, image)
    return path


def get_bird_classes():
    try:
        with open('../model/class_names.json') as f:
            index_to_class = json.load(f)
        return list(index_to_class.values())
    except Exception as e:
        print(f"[ERROR] Could not load class names: {e}")
        # fallback — якщо файл не знайдено
        return ["blackbird", "blue_tit", "great_tit", "robin", "sparrow"]


def handle_user_feedback(image, predicted_class, user_choice, corrected_class=None):
    if user_choice == "yes":
        return save_image(image, predicted_class)
    elif user_choice == "no" and corrected_class:
        return save_image(image, corrected_class)
    elif user_choice == "unsure":
        return save_image(image, "unknown")
    return None

