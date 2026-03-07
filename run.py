import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Add venv site-packages manually if missing
venv_site = os.path.join(os.path.dirname(__file__), '.venv', 'Lib', 'site-packages')
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.append(venv_site)

# Pre-import torch to avoid DLL conflict with PyQt6 (WinError 1114)
import torch

from src.gui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
