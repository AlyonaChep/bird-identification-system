from PyQt5.QtWidgets import QDialog, QVBoxLayout, QComboBox, QHBoxLayout

from gui.ui_helpers import create_label, create_button
from src.feedback_handler import get_bird_classes


class MultiEditDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Class for Selected Images")
        self.setFixedSize(400, 150)

        layout = QVBoxLayout()

        layout.addWidget(create_label("Select new class:", bold=True, size=16))

        self.combo = QComboBox()
        self.combo.addItems(get_bird_classes() + ["other"])
        self.combo.setStyleSheet("font-size: 16px; padding: 4px;")
        layout.addWidget(self.combo)

        btn_layout = QHBoxLayout()
        self.ok_btn = create_button("✅ Apply", self.accept)
        self.cancel_btn = create_button("❌ Cancel", self.reject)
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def selected_class(self):
        return self.combo.currentText()
