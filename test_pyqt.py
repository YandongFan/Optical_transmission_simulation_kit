import sys
try:
    from PyQt6.QtWidgets import QApplication, QMainWindow
    print("PyQt6.QtWidgets imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
