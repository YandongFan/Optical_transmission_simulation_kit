import os
import sys

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout  # noqa: E402

from src.gui.data_preview_widget import DataPreviewWidget  # noqa: E402


def _make_preview(title: str, allow_3d: bool, cmap: str, complex_policy: str):
    w = DataPreviewWidget(
        title=title,
        allow_3d=allow_3d,
        default_cmap=cmap,
        complex_policy=complex_policy,
    )
    w.set_extent_um(-10.0, 10.0, -10.0, 10.0)
    return w


def _print_state(label: str, w: DataPreviewWidget):
    canvas_w = int(w.canvas.width())
    canvas_h = int(w.canvas.height())
    ax = w.fig.axes[0] if w.fig.axes else None
    if ax is None:
        print(f"{label}: canvas={canvas_w}x{canvas_h}, ax=None")
        return
    l, b, ww, hh = ax.get_position().bounds
    print(f"{label}: canvas={canvas_w}x{canvas_h}, ax_bounds={l:.3f},{b:.3f},{ww:.3f},{hh:.3f}, aspect={ax.get_aspect()}")


def main():
    app = QApplication.instance() or QApplication([])

    win = QMainWindow()
    root = QWidget()
    lay = QVBoxLayout(root)

    w_phase = _make_preview("相位预览 (Phase Preview)", True, "twilight", "angle")
    w_amp = _make_preview("透射率预览 (Transmission Preview)", False, "viridis", "abs")
    lay.addWidget(w_phase, 1)
    lay.addWidget(w_amp, 1)

    win.setCentralWidget(root)
    win.resize(1100, 860)
    win.show()

    cases = [
        ("square_730", (730, 730)),
        ("square_2048", (2048, 2048)),
        ("rect_600x900", (600, 900)),
    ]
    for name, shape in cases:
        a = np.random.rand(*shape).astype(np.float64)
        w_phase.set_data(a)
        w_amp.set_data(a)
        app.processEvents()
        _print_state(f"{name}/phase", w_phase)
        _print_state(f"{name}/amp", w_amp)

    win.close()
    app.processEvents()


if __name__ == "__main__":
    main()

