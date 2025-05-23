import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

def run():
    """Запуск графічної версії програми."""
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()