import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 尽量避免 Windows 下 OpenMP/MKL 与 Qt 的 DLL 冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# 仅在“非 venv 解释器启动”时，补充 .venv 的 site-packages（避免混用导致 native 崩溃）
if sys.prefix == sys.base_prefix:
    venv_site = os.path.join(os.path.dirname(__file__), ".venv", "Lib", "site-packages")
    if os.path.exists(venv_site) and venv_site not in sys.path:
        sys.path.append(venv_site)

# PyTorch 与 Qt 在 Windows 上存在 DLL 初始化顺序问题：
# - torch 在 Qt 之后导入：可能触发 WinError 1114
# - torch 在 Qt 之前导入：若 OpenMP 冲突未处理，可能触发进程异常退出
# 这里在设置环境变量后预加载 torch，最大化兼容性
import torch

from src.gui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
