import os
import re
import shutil
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QLabel, QScrollArea, QVBoxLayout, QGridLayout,
    QFrame, QHBoxLayout, QComboBox, QMessageBox, QDialog
)

from gui.ui_helpers import create_button, create_label
from gui.widgets.multi_edit_dialog import MultiEditDialog
from src.config import VIDEO_OBS_DIR, IMAGES_OBS_DIR


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
        combo_style = "font-size: 16px; padding: 7px;"
        self.filter_combo.setStyleSheet(combo_style)

        self.multi_select_button = create_button("🖼️ Multi-select edit", self.toggle_multi_select_mode)
        self.back_button = create_button("⬅️ Back", self.main_window.show_home)

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
        # Витягує назву птаха та дату/час із імені файлу на основі шаблону
        name_without_ext = os.path.splitext(filename)[0]

        # Виділяємо дату
        date_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', name_without_ext)
        date_part = date_match.group(1) if date_match else "Unknown date"

        # Видаляємо дату та суфікс _confirmed з імені
        name_cleaned = name_without_ext.replace('_confirmed', '')
        name_cleaned = name_cleaned.replace(date_part, '')
        name_cleaned = name_cleaned.rstrip('_')

        bird_name = name_cleaned.replace('_', ' ').title()
        return bird_name, date_part

    def clear_grid(self):
        # Очищає сітку зображень в архіві: видаляє всі віджети та обнуляє вибір
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.tiles.clear()
        self.selected_tiles.clear()

    def load_observations(self):
        # Завантажує спостереження (зображення або відео) з відповідних директорій та
        # відображає їх у вигляді плиток із можливістю фільтрації за типом
        self.clear_grid()

        selected_filter = self.filter_combo.currentText() if hasattr(self, "filter_combo") else "All"

        self.path_label.setText("Observations folder: bird_identification_system/dataset/observations")
        image_dir = IMAGES_OBS_DIR
        videos_dir = VIDEO_OBS_DIR
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

            layout.addStretch()
            layout.addWidget(label)
            frame.setLayout(layout)
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setStyleSheet("padding: 0px;")

            frame.mousePressEvent = lambda event, f=frame, p=image_path: self.toggle_tile_selection(f, p)

            self.grid.addWidget(frame, row, col)
            self.tiles.append((frame, tag or text, image_path))

            col += 1
            if col >= 3:
                col = 0
                row += 1

        image_files = []
        if selected_filter in ["All", "Images"] and os.path.exists(image_dir):
            for filename in os.listdir(image_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(image_dir, filename)
                    name, dt_str = self.extract_bird_name_and_datetime(filename)

                    try:
                        dt = datetime.strptime(dt_str, "%Y-%m-%d_%H-%M-%S")
                    except Exception:
                        dt = datetime.min  # если не удалось - ставим минимальное значение для сортировки вниз

                    confirmed = "✅" if "_confirmed" in filename else ""
                    display_text = f"{name} {confirmed}".strip()
                    image_files.append((dt, display_text, full_path, name))

        image_files.sort(key=lambda x: x[0], reverse=True)

        for dt, display_text, full_path, name in image_files:
            add_tile(f"{display_text}\n{dt.strftime('%Y-%m-%d %H:%M:%S')}", full_path, tag=name)

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

                            # Витягуємо дату з назви папки (остання частина — timestamp)
                            try:
                                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$', video_folder)
                                dt = datetime.strptime(timestamp_match.group(1),
                                                       "%Y-%m-%d_%H-%M-%S") if timestamp_match else datetime.min
                            except Exception:
                                dt = datetime.min

                            # Витягуємо назву відео (все до останнього підкреслення)
                            video_base_name = video_folder.rsplit("_", 1)[0]

                            # Відображення
                            display_text = f"{bird_name}\n{dt.strftime('%Y-%m-%d %H:%M:%S')}\nfrom video: {video_base_name}, frame №{frame_num}"
                            add_tile(display_text.strip(), full_path, tag=bird_name)

    def group_by_class(self):
        # Групує зображення у сітці за класами птахів (сортує за назвою)
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
        # Активує/деактивує режим мультивибору для групового редагування зображень
        self.multi_select_mode = not self.multi_select_mode

        if self.multi_select_mode:
            if not self.selected_tiles:
                QMessageBox.information(self, "No selection", "Please select images to edit.")
                self.multi_select_mode = False
                return

            dialog = MultiEditDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                new_class = dialog.selected_class().replace(" ", "_").lower()

                updated_files = 0
                for path in list(self.selected_tiles):
                    dir_name = os.path.dirname(path)
                    filename = os.path.basename(path)
                    ext = os.path.splitext(filename)[1]

                    # ---- Визначаємо тип файлу ----
                    if re.match(r"^\d+_", filename):  # починається з номера кадру
                        # Відео кадр: 23_crow.jpg → 23_newclass_confirmed.jpg
                        frame_num = filename.split('_')[0]
                        new_filename = f"{frame_num}_{new_class}_confirmed{ext}"

                    elif re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', filename):
                        # Фото з датою: sparrow_2024-05-20_15-30-00.jpg → newclass_2024-05-20_15-30-00_confirmed.jpg
                        date_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
                        datetime_part = date_match.group(1) if date_match else "unknown_time"
                        new_filename = f"{new_class}_{datetime_part}_confirmed{ext}"

                    else:
                        # fallback
                        new_filename = f"{new_class}_renamed{ext}"

                    new_path = os.path.join(dir_name, new_filename)

                    if not os.path.exists(new_path):
                        shutil.move(path, new_path)
                        updated_files += 1
                    else:
                        print(f"⚠️ File already exists: {new_path}")

                QMessageBox.information(self, "Done", f"Renamed {updated_files} files to class '{new_class}'.")
            else:
                self.multi_select_mode = False

        self.load_observations()

    def toggle_tile_selection(self, frame, path):
        # Додає або прибирає зображення зі списку обраних при кліку мишею
        if path in self.selected_tiles:
            frame.setStyleSheet("padding: 0px; background-color: none;")  # deselect
            self.selected_tiles.remove(path)
        else:
            frame.setStyleSheet("padding: 0px; background-color: #e0e0e0;")  # light gray for selection
            self.selected_tiles.add(path)
