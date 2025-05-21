import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QTextEdit, QProgressBar, QHBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt

from src.video_processor import VideoProcessorThread

class VideoView(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.video_path = None
        self.worker = None

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("🎥 Bird identification by video")
        self.label.setAlignment(Qt.AlignCenter)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.select_button = QPushButton("📂 Select video")
        self.analyse_button = QPushButton("🧠 analyse")
        self.back_button = QPushButton("← Back")

        self.select_button.clicked.connect(self.select_video)
        self.analyse_button.clicked.connect(self.analyse_video)
        self.back_button.clicked.connect(self.go_back)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.analyse_button)
        button_layout.addWidget(self.back_button)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.video_label)
        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.log_output)

    def select_video(self):
        self.log_output.clear()
        self.video_label.setText("")
        file_path, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Videos (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path = file_path
            filename = os.path.basename(file_path)
            self.video_label.setText(f"Selected video: {filename}")
            self.log_output.setText("Video selected. Ready for analysis.")

    def analyse_video(self):
        if not self.video_path:
            QMessageBox.warning(self, "No video selected", "Please select a video file before analysing.")
            return

        self.worker = VideoProcessorThread(self.video_path)
        self.worker.log_signal.connect(self.append_log)
        self.worker.summary_signal.connect(self.display_summary)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.start()
        self.log_output.append("🔄 Starting analysis...")

    def append_log(self, text):
        self.log_output.append(text)

    def display_summary(self, summary):
        self.log_output.append("\n✅ Summary:")
        for bird, count in summary.items():
            self.log_output.append(f"• {bird}: {count} appearance(s)")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def go_back(self):
        self.video_path = None
        self.video_label.setText("")
        self.log_output.clear()
        self.progress_bar.setValue(0)
        self.main_window.show_home()
