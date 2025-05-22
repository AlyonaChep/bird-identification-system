from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QHBoxLayout
from src.feedback_handler import get_bird_classes


class MultiEditDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Class for Selected Images")
        self.setFixedSize(400, 150)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Select new class:"))

        self.combo = QComboBox()
        self.combo.addItems(get_bird_classes() + ["other"])
        layout.addWidget(self.combo)

        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("✅ Apply")
        self.cancel_btn = QPushButton("❌ Cancel")
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def selected_class(self):
        return self.combo.currentText()
