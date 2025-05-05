from src.data_preprocessing import create_data_generators
from src.config import MODEL_PATH
from src.test_model import evaluate_and_report

if __name__ == '__main__':
    # Створюємо лише test_generator
    _, _, test_gen, _ = create_data_generators()

    # Запускаємо оцінку
    evaluate_and_report(MODEL_PATH, test_gen)
