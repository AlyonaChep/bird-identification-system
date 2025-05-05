from tensorflow.keras.optimizers import Adam
from .model_builder import build_model

def train_model(train_gen, val_gen, num_classes, model_save_path, epochs=10):
    model = build_model(num_classes)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save(model_save_path)
    return model
