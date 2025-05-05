from pathlib import Path

# Корінь проєкту
BASE_DIR = Path(__file__).resolve().parent.parent

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"

MODEL_PATH = BASE_DIR / "model" / "bird_identification_model.h5"
