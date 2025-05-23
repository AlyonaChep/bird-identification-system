from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"
MODEL_PATH = BASE_DIR / "model" / "bird_identification_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"
OBSERVATIONS_DIR = BASE_DIR / "dataset" / "observations"
IMAGES_OBS_DIR = OBSERVATIONS_DIR / "images"
VIDEO_OBS_DIR = OBSERVATIONS_DIR / "videos"
