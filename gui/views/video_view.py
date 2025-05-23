import os
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QProgressBar, QHBoxLayout, QMessageBox, QFileDialog

from gui.ui_helpers import create_button, create_label
from gui.widgets.snapshot_review_dialog import SnapshotReviewDialog
from src.config import VIDEO_OBS_DIR
from src.core.video_processor import VideoProcessorThread


class VideoView(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.video_path = None
        self.worker = None
        self.is_running = False

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.label = create_label("🎥 Bird identification by video", bold=True, size=16)
        self.setGeometry(200, 200, 800, 600)
        self.path_label = create_label("", center=True)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.feedback_widget = None

        btn_layout = QHBoxLayout()
        self.select_button = create_button("📂 Select video", self.select_video)
        self.analyse_button = create_button("🧠 Analyse", self.analyse_video)
        self.back_button = create_button("← Back", self.go_back)

        btn_layout.addWidget(self.select_button)
        btn_layout.addWidget(self.analyse_button)
        btn_layout.addWidget(self.back_button)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.path_label)
        self.layout.addLayout(btn_layout)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.log_output)

    def select_video(self):
        self.log_output.clear()
        self.path_label.setText("")
        self.progress_bar.setValue(0)
        self.analyse_button.setText("🧠 Analyse")
        self.analyse_button.clicked.disconnect()
        self.analyse_button.clicked.connect(self.analyse_video)
        file_path, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Videos (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path = file_path
            filename = os.path.basename(file_path)
            self.path_label.setText(f"Selected video: {filename}")
            self.log_output.setText("Video selected. Ready for analysis.")

    def analyse_video(self):
        if self.is_running:
            if self.worker:
                self.worker.abort()
                self.log_output.append("⏹️ Aborting analysis...")
                self.analyse_button.setEnabled(False)  # Щоб не тицяли ще раз
            return

        if not self.video_path:
            QMessageBox.warning(self, "No video selected", "Please select a video file before analysing.")
            return

        self.worker = VideoProcessorThread(self.video_path)
        self.worker.log_signal.connect(self.append_log)
        self.worker.summary_signal.connect(self.display_summary)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.start()

        self.is_running = True
        self.analyse_button.setText("⏹️ Stop")
        self.log_output.append("🔄 Starting analysis...")

    def analysis_finished(self):
        self.is_running = False
        self.analyse_button.setText("🔍 Review snapshots")
        self.analyse_button.clicked.disconnect()
        self.analyse_button.clicked.connect(self.review_snapshots)
        self.analyse_button.setEnabled(True)

    def append_log(self, text):
        self.log_output.append(text)

    def display_summary(self, summary):
        self.log_output.append("\n✅ Summary:")
        for bird, count in summary.items():
            self.log_output.append(f"• {bird}: {count} appearance(s)")

        self.review_snapshots()

    def review_snapshots(self):
        filename = Path(self.video_path).stem
        snapshot_dir = VIDEO_OBS_DIR / filename

        if not os.path.exists(snapshot_dir):
            self.log_output.append("⚠️ No snapshot directory found.")
            return

        files_to_review = [
            f for f in os.listdir(snapshot_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and "_confirmed" not in f
        ]

        if not files_to_review:
            QMessageBox.information(self, "No snapshots", "All snapshots have already been confirmed.")
            return

        dialog = SnapshotReviewDialog(snapshot_dir)
        dialog.exec_()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def go_back(self):
        self.video_path = None
        self.path_label.setText("")
        self.log_output.clear()
        self.progress_bar.setValue(0)

        self.analyse_button.setText("🧠 Analyse")
        self.analyse_button.clicked.disconnect()
        self.analyse_button.clicked.connect(self.analyse_video)

        self.main_window.show_home()
