import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.gui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
