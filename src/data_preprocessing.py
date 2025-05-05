from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE

def create_data_generators():
    # Тренувальна аугментація
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Для валідації та тесту – тільки rescale
    val_test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = val_test_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    class_names = list(train_generator.class_indices.keys())

    return train_generator, val_generator, test_generator, class_names