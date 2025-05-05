from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_and_report(model_path, test_generator):
    # Завантажуємо модель
    model = load_model(model_path)
    print(f'Model loaded from {model_path}')

    # Оцінка
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')

    # Отримуємо передбачення
    preds = model.predict(test_generator)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Матриця плутанини
    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Звіт класифікації
    report = classification_report(true_classes, pred_classes, target_names=class_labels)
    print(report)