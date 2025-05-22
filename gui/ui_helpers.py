from PyQt5.QtWidgets import QPushButton, QLabel
from PyQt5.QtCore import Qt


def create_button(text, callback):
    btn = QPushButton(text)
    btn.clicked.connect(callback)
    return btn


def create_label(text, bold=False, center=True, size=14, color=None):
    label = QLabel(text)
    if center:
        label.setAlignment(Qt.AlignCenter)
    style = f"font-size: {size}px;"
    if bold:
        style += " font-weight: bold;"
    if color:
        style += f" color: {color};"
    label.setStyleSheet(style)
    return label
