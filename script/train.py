from src.data_preprocessing import create_data_generators
from src.train_model import train_model
from src.config import MODEL_PATH

if __name__ == '__main__':
    train_gen, val_gen, test_gen, class_names = create_data_generators()
    model = train_model(train_gen, val_gen, num_classes=len(class_names), model_save_path=MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')
