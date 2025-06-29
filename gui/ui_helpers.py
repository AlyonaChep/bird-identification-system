from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QLabel


def create_button(text, callback, size=16, style=None):
    btn = QPushButton(text)
    btn.clicked.connect(callback)
    default_style = f"font-size: {size}px; padding: 8px;"
    if style:
        btn.setStyleSheet(default_style + style)
    else:
        btn.setStyleSheet(default_style)
    return btn



def create_label(text, bold=False, center=True, size=16, color=None):
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
