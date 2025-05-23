import datetime
import json
import os

import cv2

from src.config import CLASS_NAMES_PATH
from src.config import IMAGES_OBS_DIR
from src.config import VIDEO_OBS_DIR


def save_image(image, class_name, context=None):
    if context and context.get("source") == "video":
        video_name = context.get("video_name", "unknown_video")
        frame_id_raw = context.get("frame_id", "0")
        try:
            frame_id = f"{int(frame_id_raw):04d}"
        except ValueError:
            frame_id = frame_id_raw

        folder = VIDEO_OBS_DIR / video_name
        filename = f"{frame_id}_{class_name}_confirmed.jpg"
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = IMAGES_OBS_DIR
        filename = f"{class_name}_{now}.jpg"

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    cv2.imwrite(str(path), image)
    return str(path)


def get_bird_classes():
    try:
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            index_to_class = json.load(f)
        return list(index_to_class.values())
    except Exception as e:
        print(f"[ERROR] Could not load class names: {e}")
        # fallback — якщо файл не знайдено
        return ["blackbird", "blue_tit", "great_tit", "robin", "sparrow"]


def handle_user_feedback(image, predicted_class, user_choice, corrected_class=None, context=None):
    if user_choice == "yes":
        class_name = predicted_class
    elif user_choice == "no" and corrected_class:
        class_name = corrected_class
    elif user_choice == "unsure":
        class_name = "unknown"
    else:
        return None

    if context and context.get("source") == "video":
        video_name = context["video_name"]
        frame_id_raw = context["frame_id"]
        try:
            frame_id = f"{int(frame_id_raw):04d}"
        except ValueError:
            frame_id = frame_id_raw

        base_folder = VIDEO_OBS_DIR / video_name
        base_folder.mkdir(parents=True, exist_ok=True)

        for fname in os.listdir(base_folder):
            if fname.startswith(f"{frame_id}_") and "_confirmed" not in fname:
                os.remove(os.path.join(base_folder, fname))

        new_filename = f"{frame_id}_{class_name}_confirmed.jpg"
        new_path = os.path.join(base_folder, new_filename)

        cv2.imwrite(new_path, image)
        return new_path

    return save_image(image, class_name, context)
