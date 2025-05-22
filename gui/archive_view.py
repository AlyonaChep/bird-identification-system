import os
import re
from PyQt5.QtWidgets import (
    QWidget, QLabel, QScrollArea, QVBoxLayout, QGridLayout,
    QFrame, QPushButton, QHBoxLayout, QSizePolicy, QComboBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from ui_helpers import create_button, create_label


class ArchiveView(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.label = create_label("🗃 Observations archive", bold=True, size=16)
        self.path_label = create_label("", center=True)

        # --- Buttons ---
        btn_layout = QHBoxLayout()

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Images", "Videos"])
        self.filter_combo.currentTextChanged.connect(self.load_observations)

        self.multi_select_button = create_button("🖼️ Multi-select edit", self.toggle_multi_select_mode)
        self.back_button = create_button("← Back", self.main_window.show_home)

        btn_layout.addWidget(self.filter_combo)
        btn_layout.addWidget(self.multi_select_button)
        btn_layout.addWidget(self.back_button)

        # --- Scrollable grid area ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.grid = QGridLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.path_label)
        self.layout.addLayout(btn_layout)
        self.layout.addWidget(self.scroll_area)

        self.tiles = []  # keep track of tile metadata for future extensions
        self.multi_select_mode = False
        self.selected_tiles = set()

        self.load_observations()

    def extract_bird_name_and_datetime(self, filename):
        name_part = filename.rsplit('_', 2)[0]
        date_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
        date_part = date_match.group(1) if date_match else "Unknown date"
        bird_name = name_part.replace('_', ' ').title()
        return bird_name, date_part

    def clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.tiles.clear()
        self.selected_tiles.clear()

    def load_observations(self):
        self.clear_grid()

        selected_filter = self.filter_combo.currentText() if hasattr(self, "filter_combo") else "All"

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'observations'))
        self.path_label.setText(f"Observations folder: bird_identification_system/dataset/observations")
        image_dir = os.path.join(base_path, 'image')
        videos_dir = os.path.join(base_path, 'videos')
        row, col = 0, 0

        def add_tile(text, image_path, tag=None):
            nonlocal row, col
            frame = QFrame()
            layout = QVBoxLayout()

            display_text = text.replace("Confirmed", "✅")

            label = QLabel(display_text)
            label.setAlignment(Qt.AlignCenter)

            pix = QPixmap(image_path)
            if not pix.isNull():
                image_label = QLabel()
                image_label.setPixmap(pix.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                image_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(image_label)

            layout.addWidget(label)
            frame.setLayout(layout)
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setStyleSheet("padding: 5px;")

            frame.mousePressEvent = lambda event, f=frame, p=image_path: self.toggle_tile_selection(f, p)

            self.grid.addWidget(frame, row, col)
            self.tiles.append((frame, tag or text, image_path))

            col += 1
            if col >= 3:
                col = 0
                row += 1

        if selected_filter in ["All", "Images"] and os.path.exists(image_dir):
            for filename in os.listdir(image_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name, dt = self.extract_bird_name_and_datetime(filename)
                    full_path = os.path.join(image_dir, filename)
                    add_tile(f"{name} — {dt}", full_path, tag=name)

        if selected_filter in ["All", "Videos"] and os.path.exists(videos_dir):
            for video_folder in os.listdir(videos_dir):
                folder_path = os.path.join(videos_dir, video_folder)
                if os.path.isdir(folder_path):
                    for filename in os.listdir(folder_path):
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            parts = filename.split('_')
                            frame_num = parts[0]
                            bird_name = '_'.join(parts[1:]).replace('.jpg', '').replace('_', ' ').title()
                            full_path = os.path.join(folder_path, filename)
                            add_tile(f"{bird_name} (from video: {video_folder}, frame №{frame_num})", full_path,
                                     tag=bird_name)

    def group_by_class(self):
        # Sort tiles alphabetically by class name (tag)
        self.tiles.sort(key=lambda t: t[1])
        self.clear_grid()
        row, col = 0, 0
        for frame, tag, image_path in self.tiles:
            self.grid.addWidget(frame, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1

    def toggle_multi_select_mode(self):
        self.multi_select_mode = not self.multi_select_mode
        self.load_observations()

    def toggle_tile_selection(self, frame, path):
        if path in self.selected_tiles:
            frame.setStyleSheet("padding: 5px;")  # deselect
            self.selected_tiles.remove(path)
        else:
            frame.setStyleSheet("padding: 2px; border: 2px solid green;")
            self.selected_tiles.add(path)
