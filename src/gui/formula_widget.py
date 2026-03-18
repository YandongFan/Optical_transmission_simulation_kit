
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTextEdit, QTableWidget, QTableWidgetItem, 
                             QPushButton, QHeaderView, QSplitter, QFrame,
                             QToolTip)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import re
from src.core.modulator import evaluate_formula

class FormulaPreview(FigureCanvas):
    def __init__(self, parent=None, width=4, height=3, dpi=100, is_intensity=False):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        self.ax.axis('off')
        super().__init__(fig)
        self.setParent(parent)
        self.cbar = None
        self.is_intensity = is_intensity

    def update_plot(self, data, title="Preview"):
        self.ax.clear()
        self.ax.set_title(title, fontsize=10)
        self.ax.axis('off')
        
        if data is None:
            self.ax.text(0.5, 0.5, "No Data / Error", 
                         ha='center', va='center', transform=self.ax.transAxes)
            # Remove old colorbar if exists
            if self.cbar:
                try:
                    self.cbar.remove()
                except Exception:
                    pass
                self.cbar = None
            self.draw()
            return

        cmap = 'gray' if self.is_intensity else 'viridis'
        im = self.ax.imshow(data, cmap=cmap, origin='lower')
        
        # Recreate colorbar safely
        if self.cbar:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
            
        self.cbar = self.figure.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
        # Fix colorbar tick labels order (flip if inverted)
        self.ax.invert_yaxis() # Fix Y axis to match matrix orientation (row 0 at top)
        
        # Optionally mark min/max on title or axes for intensity
        if self.is_intensity:
            d_min, d_max = np.min(data), np.max(data)
            self.ax.set_title(f"{title} [Min: {d_min:.2f}, Max: {d_max:.2f}]", fontsize=9)
            
        self.draw()

class FormulaWidget(QWidget):
    """
    Reusable widget for custom formula editing and preview.
    """
    formulaChanged = pyqtSignal(str) # Emits validation status or formula
    
    def __init__(self, formula_type='trans', parent=None):
        """
        :param formula_type: 'trans' (0-1) or 'phase' (real)
        """
        super().__init__(parent)
        self.formula_type = formula_type
        self.last_valid_formula = ""
        self.custom_vars = {}
        
        self.init_ui()
        
        # Debounce timer
        self.check_timer = QTimer()
        self.check_timer.setSingleShot(True)
        self.check_timer.setInterval(500)
        self.check_timer.timeout.connect(self.validate_and_preview)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Splitter for Editor vs Preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Editor
        editor_widget = QWidget()
        editor_layout = QVBoxLayout(editor_widget)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        
        # Help Label
        help_text = "支持变量 (Vars): x, y, lambda, theta, phi (builtin) + custom"
        if self.formula_type == 'trans':
            help_text += "\n输出范围 (Output): [0, 1]"
        else:
            help_text += "\n输出单位 (Unit): Radian/Degree (User def)"
        
        self.lbl_help = QLabel(help_text)
        self.lbl_help.setStyleSheet("color: gray; font-size: 9pt;")
        editor_layout.addWidget(self.lbl_help)
        
        # Formula Input
        self.txt_formula = QTextEdit()
        self.txt_formula.setPlaceholderText("e.g. np.sin(x/10) * 0.5 + 0.5")
        self.txt_formula.setMaximumHeight(80)
        self.txt_formula.textChanged.connect(self.on_text_changed)
        editor_layout.addWidget(self.txt_formula)
        
        # Error Label
        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet("color: red; font-size: 9pt;")
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setVisible(False)
        editor_layout.addWidget(self.lbl_error)
        
        # Variables Table
        self.table_vars = QTableWidget(0, 2)
        self.table_vars.setHorizontalHeaderLabels(["Name", "Value"])
        self.table_vars.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_vars.setMaximumHeight(100)
        self.table_vars.cellChanged.connect(self.on_var_changed)
        editor_layout.addWidget(self.table_vars)
        
        # Var Buttons
        btn_layout = QHBoxLayout()
        self.btn_add_var = QPushButton("+ Var")
        self.btn_add_var.clicked.connect(self.add_variable)
        self.btn_del_var = QPushButton("- Var")
        self.btn_del_var.clicked.connect(self.del_variable)
        btn_layout.addWidget(self.btn_add_var)
        btn_layout.addWidget(self.btn_del_var)
        editor_layout.addStretch()
        editor_layout.addLayout(btn_layout)
        
        splitter.addWidget(editor_widget)
        
        # Right: Preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        # We need two previews if phase, else one?
        # Requirement: "In the same Parameter Definition interface, add an Intensity Preview next to Phase"
        # Since FormulaWidget is used for both 'trans' and 'phase', we can show both for both, 
        # or conditionally. Let's just stack them vertically or horizontally.
        preview_split = QSplitter(Qt.Orientation.Vertical)
        
        self.preview_canvas = FormulaPreview(width=3, height=2)
        preview_split.addWidget(self.preview_canvas)
        
        # Intensity Canvas
        self.intensity_canvas = FormulaPreview(width=3, height=2, is_intensity=True)
        preview_split.addWidget(self.intensity_canvas)
        
        preview_layout.addWidget(preview_split)
        
        splitter.addWidget(preview_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)

    def on_text_changed(self):
        self.check_timer.start()
        
    def on_var_changed(self, row, col):
        self.update_custom_vars_from_table()
        self.check_timer.start()

    def add_variable(self):
        row = self.table_vars.rowCount()
        self.table_vars.insertRow(row)
        self.table_vars.setItem(row, 0, QTableWidgetItem(f"k{row}"))
        self.table_vars.setItem(row, 1, QTableWidgetItem("1.0"))

    def del_variable(self):
        row = self.table_vars.currentRow()
        if row >= 0:
            self.table_vars.removeRow(row)
            self.update_custom_vars_from_table()
            self.check_timer.start()

    def update_custom_vars_from_table(self):
        self.custom_vars = {}
        for r in range(self.table_vars.rowCount()):
            name_item = self.table_vars.item(r, 0)
            val_item = self.table_vars.item(r, 1)
            if name_item and val_item:
                try:
                    self.custom_vars[name_item.text()] = float(val_item.text())
                except ValueError:
                    pass

    def get_formula(self):
        return self.txt_formula.toPlainText().strip()

    def set_formula(self, formula):
        self.txt_formula.blockSignals(True)
        self.txt_formula.setPlainText(formula)
        self.txt_formula.blockSignals(False)
        self.validate_and_preview()

    def set_variables(self, vars_dict):
        self.table_vars.blockSignals(True)
        self.table_vars.setRowCount(0)
        for i, (k, v) in enumerate(vars_dict.items()):
            self.table_vars.insertRow(i)
            self.table_vars.setItem(i, 0, QTableWidgetItem(str(k)))
            self.table_vars.setItem(i, 1, QTableWidgetItem(str(v)))
        self.update_custom_vars_from_table()
        self.table_vars.blockSignals(False)
        self.validate_and_preview()

    def validate_and_preview(self):
        try:
            formula = self.get_formula()
            if not formula:
                self.lbl_error.setText("")
                self.lbl_error.setVisible(False)
                self.preview_canvas.update_plot(None)
                self.intensity_canvas.update_plot(None)
                return

            # Create dummy grid (meters, e.g. -50um to 50um)
            N = 256
            x = np.linspace(-50e-6, 50e-6, N)
            y = np.linspace(-50e-6, 50e-6, N)
            X, Y = np.meshgrid(x, y)
            
            # Wavelength 0.532 um in meters
            res = evaluate_formula(formula, self.custom_vars, X, Y, 0.532e-6)
            
            if res is None:
                self.lbl_error.setText("Error: Evaluation Failed")
                self.lbl_error.setVisible(True)
                self.preview_canvas.update_plot(None)
                self.intensity_canvas.update_plot(None)
                self.formulaChanged.emit("invalid")
                return
                
            # Validation for Trans (0-1)
            intensity_res = None
            preview_res = res
            
            if self.formula_type == 'trans':
                if np.any(res < 0) or np.any(res > 1):
                    self.lbl_error.setText("Warning: Values out of range [0, 1] (clamped for preview)")
                    self.lbl_error.setVisible(True)
                    res = np.clip(res, 0, 1)
                    preview_res = res
                else:
                    self.lbl_error.setText("")
                    self.lbl_error.setVisible(False)
                    
                # Intensity for trans is |T|^2
                intensity_res = np.abs(res)**2
                
            elif self.formula_type == 'phase':
                self.lbl_error.setText("")
                self.lbl_error.setVisible(False)
                # Phase wrapping for preview (mod 360)
                # We assume the result is in radians? 
                # Requirement: "包裹算法采用模 360 运算：wrapped_phase = mod(original_phase, 360)"
                # If the formula outputs degrees, mod 360. If radians, mod 2pi.
                # Let's assume user expects degrees for preview wrapping as requested,
                # or just directly wrap the numerical result by 360 if they input degrees.
                # "wrapped_phase = mod(original_phase, 360)"
                preview_res = np.mod(res, 360)
                
                # Intensity for pure phase is |exp(i*phi)|^2 = 1.0 everywhere
                intensity_res = np.ones_like(res)
                
            self.preview_canvas.update_plot(preview_res, title=f"Preview ({self.formula_type.capitalize()})")
            self.intensity_canvas.update_plot(intensity_res, title="Intensity (|E|^2)")
            
            self.last_valid_formula = formula
            self.formulaChanged.emit("valid")
        except Exception as e:
            # Catch all GUI-level exceptions during validation
            print(f"Validation crash prevented: {e}")
            
            # The exception from evaluate_formula already has line number formatted
            err_msg = str(e)
            
            self.lbl_error.setText(f"System Error: {err_msg}")
            self.lbl_error.setVisible(True)
            self.preview_canvas.update_plot(None)
            self.intensity_canvas.update_plot(None)
            self.formulaChanged.emit("invalid")

