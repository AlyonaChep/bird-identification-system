from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QComboBox, QMessageBox
from PyQt5.QtCore import pyqtSignal

from src.feedback_handler import get_bird_classes


class FeedbackWidget(QWidget):
    feedback_given = pyqtSignal(str, str)  # (user_choice, corrected_class)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.layout.addWidget(QLabel("Is it correct?"))
        self.yes_btn = QPushButton("✅ Yes")
        self.no_btn = QPushButton("❌ No")
        self.unsure_btn = QPushButton("❓ Not sure")

        self.yes_btn.clicked.connect(self.yes_clicked)
        self.no_btn.clicked.connect(self.show_class_selector)
        self.unsure_btn.clicked.connect(self.unsure_clicked)

        self.layout.addWidget(self.yes_btn)
        self.layout.addWidget(self.no_btn)
        self.layout.addWidget(self.unsure_btn)

        self.combo_layout = None

    def yes_clicked(self):
        self.feedback_given.emit("yes", None)
        self.hide()

    def unsure_clicked(self):
        self.feedback_given.emit("unsure", None)
        self.hide()

    def show_class_selector(self):
        if self.combo_layout:
            return

        self.combo_layout = QHBoxLayout()
        self.class_selector = QComboBox()
        self.class_selector.addItems(get_bird_classes() + ["other"])

        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(self.send_corrected_class)

        self.combo_layout.addWidget(self.class_selector)
        self.combo_layout.addWidget(confirm_btn)
        self.layout.addLayout(self.combo_layout)

    def send_corrected_class(self):
        selected = self.class_selector.currentText()
        self.feedback_given.emit("no", selected)
        self.hide()

    def clear(self):
        if self.combo_layout:
            for i in reversed(range(self.combo_layout.count())):
                widget = self.combo_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            self.combo_layout = None
