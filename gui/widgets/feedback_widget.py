from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox

from src.feedback_handler import get_bird_classes
from gui.ui_helpers import create_button, create_label  # Імпорт хелперів


class FeedbackWidget(QWidget):
    feedback_given = pyqtSignal(str, str)  # (user_choice, corrected_class)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.layout.addWidget(create_label("Is it correct?", bold=True, size=16))

        self.yes_btn = create_button("✅ Yes", self.yes_clicked)
        self.no_btn = create_button("❌ No", self.show_class_selector)
        self.unsure_btn = create_button("❓ Not sure", self.unsure_clicked)

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
        self.class_selector.setStyleSheet("font-size: 16px; padding: 4px;")

        confirm_btn = create_button("Confirm", self.send_corrected_class, size=16)

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
