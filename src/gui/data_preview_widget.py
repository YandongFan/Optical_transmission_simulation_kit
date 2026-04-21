import time
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSizePolicy

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class DataPreviewWidget(QWidget):
    def __init__(
        self,
        title: str,
        allow_3d: bool,
        default_cmap: str,
        complex_policy: str,
        parent=None,
    ):
        super().__init__(parent)
        self._title = title
        self._allow_3d = allow_3d
        self._complex_policy = complex_policy
        self._data = None
        self._last_error = None
        self._cbar = None
        self._extent_um = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        header = QHBoxLayout()
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("font-weight: 600;")
        header.addWidget(self.lbl_title)

        header.addStretch()

        self.combo_mode = QComboBox()
        if allow_3d:
            self.combo_mode.addItems(["二维 (2D)", "三维 (3D)"])
        else:
            self.combo_mode.addItems(["二维 (2D)"])
        header.addWidget(self.combo_mode)

        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(
            [
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "cividis",
                "gray",
                "jet",
                "hsv",
                "twilight",
                "twilight_shifted",
            ]
        )
        if default_cmap in [self.combo_cmap.itemText(i) for i in range(self.combo_cmap.count())]:
            self.combo_cmap.setCurrentText(default_cmap)
        header.addWidget(self.combo_cmap)

        self.cb_colorbar = QCheckBox("显示色标")
        self.cb_colorbar.setChecked(True)
        header.addWidget(self.cb_colorbar)

        self.lbl_meta = QLabel("")
        self.lbl_meta.setStyleSheet("color: gray;")
        header.addWidget(self.lbl_meta)

        root.addLayout(header)

        self.err_label = QLabel("")
        self.err_label.setWordWrap(True)
        self.err_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.err_label.setVisible(False)
        root.addWidget(self.err_label)

        self.fig = Figure(figsize=(5.6, 4.2), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.canvas.setMinimumHeight(260)
        root.addWidget(self.toolbar, 0)
        root.addWidget(self.canvas, 1)

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(120)
        self._resize_timer.timeout.connect(self._render_if_ready)

        self.combo_mode.currentIndexChanged.connect(self._render_if_ready)
        self.combo_cmap.currentIndexChanged.connect(self._render_if_ready)
        self.cb_colorbar.toggled.connect(self._render_if_ready)

        self.set_placeholder("未加载 (Not Loaded)")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._data is None:
            return
        self._resize_timer.start()

    def set_placeholder(self, message: str):
        self._data = None
        self._last_error = None
        self.err_label.setText(message)
        self.err_label.setStyleSheet("color: gray; font-size: 10pt;")
        self.err_label.setVisible(True)
        self.toolbar.setVisible(False)
        self.canvas.setVisible(False)

    def set_error(self, message: str):
        self._data = None
        self._last_error = message
        self.err_label.setText(message)
        self.err_label.setStyleSheet("color: red; font-size: 10pt;")
        self.err_label.setVisible(True)
        self.toolbar.setVisible(False)
        self.canvas.setVisible(False)

    def set_data(self, data: np.ndarray):
        try:
            arr = self._to_2d_array(data)
            self._data = arr
            self._last_error = None
            self.err_label.setVisible(False)
            self.toolbar.setVisible(True)
            self.canvas.setVisible(True)
            self._render()
        except Exception as e:
            self.set_error(f"预览失败：{e}")

    def set_extent_um(self, x_min: float, x_max: float, y_min: float, y_max: float):
        self._extent_um = (float(x_min), float(x_max), float(y_min), float(y_max))
        if self._data is not None:
            self._render()

    def _to_2d_array(self, data: np.ndarray) -> np.ndarray:
        if data is None:
            raise ValueError("数据为空")
        arr = np.asarray(data)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"需要二维数组，当前维度: {arr.shape}")
        if np.iscomplexobj(arr):
            if self._complex_policy == "angle":
                arr = np.angle(arr)
            elif self._complex_policy == "abs":
                arr = np.abs(arr)
            else:
                raise ValueError("不支持复数数组")
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"数据类型不支持: {arr.dtype}")
        arr = arr.astype(np.float64, copy=False)
        if not np.isfinite(arr).any():
            raise ValueError("数据全为 NaN/Inf")
        return arr

    def _downsample(self, arr: np.ndarray, max_size: int) -> np.ndarray:
        ny, nx = arr.shape
        sy = max(1, int(np.ceil(ny / max_size)))
        sx = max(1, int(np.ceil(nx / max_size)))
        return arr[::sy, ::sx]

    def _render_if_ready(self):
        if self._data is None:
            return
        self._render()

    def _render(self):
        t0 = time.perf_counter()
        arr = self._data

        mode = self.combo_mode.currentText()
        is_3d = "3D" in mode
        if is_3d:
            max_size = 128
        else:
            try:
                w, h = self.canvas.get_width_height()
                max_size = int(min(2048, max(512, max(w, h))))
            except Exception:
                max_size = 512
        view = self._downsample(arr, max_size=max_size)

        try:
            w, h = self.canvas.get_width_height()
            if w > 0 and h > 0:
                self.fig.set_size_inches(w / self.fig.dpi, h / self.fig.dpi, forward=False)
        except Exception:
            pass

        self.fig.clear()
        self._cbar = None

        cmap = self.combo_cmap.currentText()
        show_cbar = self.cb_colorbar.isChecked()

        if is_3d and self._allow_3d:
            ax = self.fig.add_subplot(111, projection="3d")
            ny, nx = view.shape
            if self._extent_um is not None:
                x0, x1, y0, y1 = self._extent_um
                x = np.linspace(x0, x1, nx)
                y = np.linspace(y0, y1, ny)
            else:
                x = np.arange(nx)
                y = np.arange(ny)
            X, Y = np.meshgrid(x, y)
            surf = ax.plot_surface(X, Y, view, cmap=cmap, linewidth=0, antialiased=False)
            ax.set_title(self._title, fontsize=10)
            ax.set_xlabel("X (μm)")
            ax.set_ylabel("Y (μm)")
            ax.set_zlabel("Value")
            if show_cbar:
                self._cbar = self.fig.colorbar(surf, ax=ax, fraction=0.046, pad=0.04, shrink=0.75)
        else:
            ax = self.fig.add_subplot(111)
            vmin = np.nanpercentile(view, 1)
            vmax = np.nanpercentile(view, 99)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin = np.nanmin(view)
                vmax = np.nanmax(view)
            imshow_kwargs = {"cmap": cmap, "origin": "lower", "vmin": vmin, "vmax": vmax}
            if self._extent_um is not None:
                imshow_kwargs["extent"] = self._extent_um
            im = ax.imshow(view, **imshow_kwargs)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(self._title, fontsize=10)
            ax.set_xlabel("X (μm)")
            ax.set_ylabel("Y (μm)")
            if show_cbar:
                self._cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        try:
            self.fig.tight_layout(pad=0.6)
        except Exception:
            pass
        self.canvas.draw_idle()

        ny0, nx0 = arr.shape
        ny1, nx1 = view.shape
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if (ny0, nx0) == (ny1, nx1):
            ds_text = f"{ny0}×{nx0}"
        else:
            ds_text = f"{ny0}×{nx0} → {ny1}×{nx1}"
        self.lbl_meta.setText(f"{ds_text} | {elapsed_ms:.0f} ms")
