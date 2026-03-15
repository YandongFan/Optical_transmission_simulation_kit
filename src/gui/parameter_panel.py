from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QGroupBox, 
                             QRadioButton, QButtonGroup, QLabel, QDoubleSpinBox, 
                             QComboBox, QPushButton, QFormLayout, QScrollArea, QFileDialog,
                             QListWidget, QHBoxLayout, QMessageBox, QCheckBox, QDialog,
                             QSpinBox, QLineEdit, QTableWidget, QTableWidgetItem, QTextEdit,
                             QStackedWidget, QHeaderView)
from PyQt6.QtGui import QAction, QShortcut, QKeySequence
from PyQt6.QtCore import Qt, pyqtSignal as Signal, QMutex, QMutexLocker
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from .formula_widget import FormulaWidget
from .polygon_widget import PolygonEditorWidget

import json

# Preset file path
PRESET_FILE = os.path.join(os.path.expanduser("~"), ".optical_simulation_kit", "preset.json")

class EquationDisplay(FigureCanvas):
    def __init__(self, parent=None, width=5, height=1, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_alpha(0) # Transparent background
        self.ax = fig.add_subplot(111)
        self.ax.axis('off')
        super().__init__(fig)
        self.setParent(parent)
        self.setFixedHeight(80) # Fixed height
        
    def update_equation(self, latex_str):
        self.ax.clear()
        self.ax.axis('off')
        self.ax.text(0.5, 0.5, f"${latex_str}$", horizontalalignment='center', 
                     verticalalignment='center', fontsize=12)
        self.draw()

class MonitorArrayDialog(QDialog):
    """
    监视器阵列配置对话框 (Monitor Array Configuration Dialog)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加监视器阵列 (Add Monitor Array)")
        layout = QFormLayout(self)
        
        self.sb_count = QSpinBox()
        self.sb_count.setRange(1, 1000)
        self.sb_count.setValue(10)
        
        # Helper for unit spinbox in dialog
        def create_dialog_unit_spinbox(val, unit):
            w = QWidget()
            l = QHBoxLayout(w)
            l.setContentsMargins(0,0,0,0)
            sb = QDoubleSpinBox()
            sb.setRange(0.0, 100000.0)
            sb.setValue(val)
            cmb = QComboBox()
            cmb.addItems(["mm", "um"])
            cmb.setCurrentText(unit)
            l.addWidget(sb)
            l.addWidget(cmb)
            
            # Link for conversion
            cmb.setProperty("last_unit", unit)
            def on_change(new_u):
                old_u = cmb.property("last_unit")
                v = sb.value()
                # to um
                v_um = v * (1000 if old_u == "mm" else 1)
                # to new
                v_new = v_um / (1000 if new_u == "mm" else 1)
                sb.setValue(v_new)
                cmb.setProperty("last_unit", new_u)
            cmb.currentTextChanged.connect(on_change)
            
            return w, sb, cmb

        self.w_start, self.sb_start, self.combo_start_unit = create_dialog_unit_spinbox(10.0, "mm")
        self.w_spacing, self.sb_spacing, self.combo_spacing_unit = create_dialog_unit_spinbox(1.0, "mm")
        
        self.le_prefix = QLineEdit("Array_")
        
        layout.addRow("监视器数量 (Count):", self.sb_count)
        layout.addRow("起始位置 (Start Z):", self.w_start)
        layout.addRow("间距 (Spacing):", self.w_spacing)
        layout.addRow("名称前缀 (Prefix):", self.le_prefix)
        
        btns = QHBoxLayout()
        btn_ok = QPushButton("确定 (OK)")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("取消 (Cancel)")
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addRow(btns)

    def get_values(self):
        return (self.sb_count.value(), 
                self.sb_start.value(), self.combo_start_unit.currentText(),
                self.sb_spacing.value(), self.combo_spacing_unit.currentText(),
                self.le_prefix.text())

class ParameterPanel(QWidget):
    """
    参数配置面板 (Parameter Configuration Panel)
    """
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # Scroll Area for parameter tabs
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        scroll.setWidget(content_widget)
        
        content_layout = QVBoxLayout(content_widget)
        
        # Tab Widget
        self.tabs = QTabWidget()
        content_layout.addWidget(self.tabs)
        
        # Data holders for modulators
        self.mod1_phase = None
        self.mod1_phase_path = None
        self.mod1_amp = None
        self.mod1_amp_path = None
        self.mod2_phase = None
        self.mod2_phase_path = None
        self.mod2_angle_trans = None
        self.mod2_angle_path = None
        
        # Monitor List
        self.monitors = [] # List of dicts: {'name': str, 'z': float, 'plane': int, 'type': int}

        # Add Tabs
        self.tabs.addTab(self.create_grid_tab(), "网格与方向 (Grid & Direction)")
        self.tabs.addTab(self.create_source_tab(), "光源 (Source)")
        self.tabs.addTab(self.create_modulator1_tab(), "调制平面1 (Mod 1)")
        self.tabs.addTab(self.create_modulator2_tab(), "调制平面2 (Mod 2)")
        self.tabs.addTab(self.create_monitor_tab(), "监视器 (Monitors)")
        
        layout.addWidget(scroll)
        
        # Action Buttons
        button_layout = QVBoxLayout()
        self.btn_preview = QPushButton("预览 (Preview)")
        self.btn_run = QPushButton("运行仿真 (Run Simulation)")
        button_layout.addWidget(self.btn_preview)
        button_layout.addWidget(self.btn_run)
        layout.addLayout(button_layout)
        
        # Simulation Configuration Sync
        self.config_mutex = QMutex()
        self.simulation_config = {}
        
        # Debounce timer for saving preset
        from PyQt6.QtCore import QTimer
        self.preset_save_timer = QTimer()
        self.preset_save_timer.setSingleShot(True)
        self.preset_save_timer.setInterval(1000) # Save after 1 second of inactivity
        self.preset_save_timer.timeout.connect(self.save_preset)
        
        # Initial Sync & Load Preset
        self.load_preset()
        self.sync_source_to_config()
        
    def sync_source_to_config(self):
        """
        同步UI参数到配置对象 (Sync UI parameters to config object)
        """
        with QMutexLocker(self.config_mutex):
            self.simulation_config = self.get_project_data()
        
        # Trigger preset save
        self.preset_save_timer.start()
        
    def save_preset(self):
        """
        Save current UI state to preset file
        """
        try:
            data = self.get_project_data()
            os.makedirs(os.path.dirname(PRESET_FILE), exist_ok=True)
            with open(PRESET_FILE, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save preset: {e}")

    def load_preset(self):
        """
        Load UI state from preset file if it exists
        """
        try:
            if os.path.exists(PRESET_FILE):
                with open(PRESET_FILE, 'r') as f:
                    data = json.load(f)
                self.load_project_data(data)
        except Exception as e:
            print(f"Failed to load preset: {e}")
            
    def get_latest_config(self):
        """
        获取最新的线程安全配置 (Get thread-safe latest config)
        """
        with QMutexLocker(self.config_mutex):
            return self.simulation_config.copy()

    def create_grid_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        direction_group = QGroupBox("光路传输方向 (Propagation Direction)")
        d_layout = QVBoxLayout()
        self.rb_x = QRadioButton("X 轴正方向 (+X)")
        self.rb_y = QRadioButton("Y 轴正方向 (+Y)")
        self.rb_z = QRadioButton("Z 轴正方向 (+Z)")
        self.rb_z.setChecked(True)
        
        # Sync signals
        self.rb_x.toggled.connect(self.sync_source_to_config)
        self.rb_y.toggled.connect(self.sync_source_to_config)
        self.rb_z.toggled.connect(self.sync_source_to_config)
        
        btn_group = QButtonGroup(self)
        btn_group.addButton(self.rb_x)
        btn_group.addButton(self.rb_y)
        btn_group.addButton(self.rb_z)
        
        d_layout.addWidget(self.rb_x)
        d_layout.addWidget(self.rb_y)
        d_layout.addWidget(self.rb_z)
        direction_group.setLayout(d_layout)
        layout.addWidget(direction_group)
        
        grid_group = QGroupBox("三维网格参数 (3D Grid Parameters)")
        g_layout = QFormLayout()
        
        self.sb_nx = QDoubleSpinBox()
        self.sb_nx.setRange(1, 10000)
        self.sb_nx.setValue(512)
        self.sb_nx.setDecimals(0)
        
        self.sb_ny = QDoubleSpinBox()
        self.sb_ny.setRange(1, 10000)
        self.sb_ny.setValue(512)
        self.sb_ny.setDecimals(0)
        
        self.sb_dx = QDoubleSpinBox()
        self.sb_dx.setRange(0.001, 1000)
        self.sb_dx.setValue(1.0)
        self.sb_dx.setSuffix(" um")
        
        self.sb_dy = QDoubleSpinBox()
        self.sb_dy.setRange(0.001, 1000)
        self.sb_dy.setValue(1.0)
        self.sb_dy.setSuffix(" um")
        
        self.sb_wavelength = QDoubleSpinBox()
        self.sb_wavelength.setRange(0.1, 100)
        self.sb_wavelength.setValue(0.532)
        self.sb_wavelength.setDecimals(3)
        self.sb_wavelength.setSuffix(" um")
        
        g_layout.addRow("Nx (Grid Points X):", self.sb_nx)
        g_layout.addRow("Ny (Grid Points Y):", self.sb_ny)
        g_layout.addRow("dx (Spacing X):", self.sb_dx)
        g_layout.addRow("dy (Spacing Y):", self.sb_dy)
        g_layout.addRow("Wavelength:", self.sb_wavelength)
        
        # Sync signals
        self.sb_nx.valueChanged.connect(self.sync_source_to_config)
        self.sb_ny.valueChanged.connect(self.sync_source_to_config)
        self.sb_dx.valueChanged.connect(self.sync_source_to_config)
        self.sb_dy.valueChanged.connect(self.sync_source_to_config)
        self.sb_wavelength.valueChanged.connect(self.sync_source_to_config)
        
        grid_group.setLayout(g_layout)
        layout.addWidget(grid_group)
        
        layout.addStretch()
        return tab

    def create_source_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # --- Equation Display ---
        self.eq_display = EquationDisplay()
        layout.addWidget(self.eq_display)
        
        # --- Source Type ---
        type_group = QGroupBox("光源类型 (Source Type)")
        t_layout = QVBoxLayout()
        self.combo_source = QComboBox()
        self.combo_source.addItems(["平面波 (Plane Wave)", "高斯光束 (Gaussian Beam)", 
                                    "拉盖尔-高斯 (Laguerre-Gaussian)", "贝塞尔光束 (Bessel Beam)",
                                    "自定义光源 (Custom Source)"])
        self.combo_source.currentIndexChanged.connect(self.update_source_ui)
        self.combo_source.currentIndexChanged.connect(self.sync_source_to_config)
        t_layout.addWidget(self.combo_source)
        type_group.setLayout(t_layout)
        layout.addWidget(type_group)
        
        # --- Polarization Settings ---
        pol_group = QGroupBox("偏振设置 (Polarization Settings)")
        pol_layout = QFormLayout()
        
        self.combo_pol_type = QComboBox()
        self.combo_pol_type.addItems(["线偏振 (Linear)", "左旋圆偏振 (LCP)", "右旋圆偏振 (RCP)", "无偏振 (Unpolarized)"])
        self.combo_pol_type.currentIndexChanged.connect(self.update_pol_ui)
        self.combo_pol_type.currentIndexChanged.connect(self.sync_source_to_config)
        
        self.sb_pol_angle = QDoubleSpinBox()
        self.sb_pol_angle.setRange(0, 180)
        self.sb_pol_angle.setDecimals(1)
        self.sb_pol_angle.setSuffix(" deg")
        self.sb_pol_angle.setToolTip("线偏振角度 (Linear Polarization Angle, 0-180)")
        self.sb_pol_angle.valueChanged.connect(self.sync_source_to_config)
        
        pol_layout.addRow("偏振类型 (Type):", self.combo_pol_type)
        pol_layout.addRow("线偏振角度 (Angle):", self.sb_pol_angle)
        
        pol_group.setLayout(pol_layout)
        layout.addWidget(pol_group)

        # --- Common Parameters ---
        common_group = QGroupBox("基础参数 (Basic Parameters)")
        c_layout = QFormLayout()
        self.sb_amplitude = QDoubleSpinBox()
        self.sb_amplitude.setValue(1.0)
        self.sb_z_pos = QDoubleSpinBox()
        self.sb_z_pos.setRange(-10000, 10000)
        self.sb_z_pos.setValue(0.0)
        self.sb_z_pos.setSuffix(" um")
        self.cb_normalize = QCheckBox("光源电场归一化 (Normalize E-field)")
        self.cb_normalize.setToolTip("勾选后，将电场最大值归一化为1。")
        
        # Sync signals
        self.sb_amplitude.valueChanged.connect(self.sync_source_to_config)
        self.sb_z_pos.valueChanged.connect(self.sync_source_to_config)
        self.cb_normalize.stateChanged.connect(self.sync_source_to_config)
        
        c_layout.addRow("振幅 (Amplitude):", self.sb_amplitude)
        c_layout.addRow("位置 (Z Position):", self.sb_z_pos)
        c_layout.addRow(self.cb_normalize)
        common_group.setLayout(c_layout)
        layout.addWidget(common_group)
        
        # --- Specific Parameters Stack ---
        self.source_stack = QStackedWidget()
        
        # 0: Plane Wave
        self.page_plane = QWidget()
        # No extra params for now (kx, ky default 0)
        self.source_stack.addWidget(self.page_plane)
        
        # 1: Gaussian
        self.page_gaussian = QWidget()
        pg_layout = QFormLayout(self.page_gaussian)
        self.sb_w0 = QDoubleSpinBox()
        self.sb_w0.setRange(0.1, 10000)
        self.sb_w0.setValue(100.0)
        self.sb_w0.setSuffix(" um")
        self.sb_w0.valueChanged.connect(self.sync_source_to_config)
        pg_layout.addRow("束腰半径 (w0):", self.sb_w0)
        self.source_stack.addWidget(self.page_gaussian)
        
        # 2: LG
        self.page_lg = QWidget()
        plg_layout = QFormLayout(self.page_lg)
        self.sb_lg_w0 = QDoubleSpinBox()
        self.sb_lg_w0.setRange(0.1, 10000)
        self.sb_lg_w0.setValue(100.0)
        self.sb_lg_w0.setSuffix(" um")
        
        self.sb_lg_p = QSpinBox()
        self.sb_lg_p.setRange(0, 50)
        self.sb_lg_p.setValue(0)
        
        self.sb_lg_l = QSpinBox()
        self.sb_lg_l.setRange(-50, 50)
        self.sb_lg_l.setValue(1)
        
        self.sb_lg_w0.valueChanged.connect(self.sync_source_to_config)
        self.sb_lg_p.valueChanged.connect(self.sync_source_to_config)
        self.sb_lg_l.valueChanged.connect(self.sync_source_to_config)
        
        plg_layout.addRow("束腰半径 (w0):", self.sb_lg_w0)
        plg_layout.addRow("径向指数 (p):", self.sb_lg_p)
        plg_layout.addRow("角向指数 (l):", self.sb_lg_l)
        self.source_stack.addWidget(self.page_lg)
        
        # 3: Bessel (Reuse Gaussian params for now or placeholder)
        self.page_bessel = QWidget()
        pb_layout = QFormLayout(self.page_bessel)
        self.sb_bessel_w0 = QDoubleSpinBox()
        self.sb_bessel_w0.setRange(0.1, 10000)
        self.sb_bessel_w0.setValue(100.0)
        self.sb_bessel_w0.setSuffix(" um")
        self.sb_bessel_w0.valueChanged.connect(self.sync_source_to_config)
        pb_layout.addRow("束腰半径 (w0):", self.sb_bessel_w0)
        self.source_stack.addWidget(self.page_bessel)
        
        # 4: Custom
        self.page_custom = QWidget()
        pc_layout = QVBoxLayout(self.page_custom)
        
        # Coordinate System Selection
        coord_group = QGroupBox("坐标系 (Coordinate System)")
        coord_layout = QHBoxLayout()
        self.combo_coord_sys = QComboBox()
        self.combo_coord_sys.addItems(["笛卡尔坐标系 (Cartesian: x, y, z)", "柱坐标系 (Cylindrical: r, phi, z)"])
        self.combo_coord_sys.currentIndexChanged.connect(self.update_custom_help)
        self.combo_coord_sys.currentIndexChanged.connect(self.sync_source_to_config)
        coord_layout.addWidget(self.combo_coord_sys)
        coord_group.setLayout(coord_layout)
        pc_layout.addWidget(coord_group)
        
        # Variables Table
        pc_layout.addWidget(QLabel("变量定义 (Variables):"))
        self.table_vars = QTableWidget(0, 2)
        self.table_vars.setHorizontalHeaderLabels(["Name", "Value"])
        self.table_vars.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_vars.cellChanged.connect(self.sync_source_to_config)
        pc_layout.addWidget(self.table_vars)
        
        h_layout = QHBoxLayout()
        self.btn_add_var = QPushButton("+")
        self.btn_add_var.clicked.connect(self.add_custom_variable)
        self.btn_del_var = QPushButton("-")
        self.btn_del_var.clicked.connect(self.del_custom_variable)
        h_layout.addWidget(self.btn_add_var)
        h_layout.addWidget(self.btn_del_var)
        pc_layout.addLayout(h_layout)
        
        # Equation Editor
        self.lbl_equation_help = QLabel("方程 (Equation): e.g. exp(-(x**2+y**2)/100**2)")
        pc_layout.addWidget(self.lbl_equation_help)
        self.txt_equation = QTextEdit()
        self.txt_equation.setPlaceholderText("Enter numpy expression...")
        self.txt_equation.setMaximumHeight(100)
        self.txt_equation.textChanged.connect(self.sync_source_to_config)
        pc_layout.addWidget(self.txt_equation)
        
        self.source_stack.addWidget(self.page_custom)
        
        layout.addWidget(self.source_stack)
        
        # Initialize
        self.update_source_ui(0)
        self.update_pol_ui(0)
        
        layout.addStretch()
        return tab

    def update_pol_ui(self, index):
        # 0: Linear -> Show Angle
        # 1, 2, 3 -> Hide Angle
        if index == 0:
            self.sb_pol_angle.setEnabled(True)
        else:
            self.sb_pol_angle.setEnabled(False)

    def update_custom_help(self):
        idx = self.combo_coord_sys.currentIndex()
        if idx == 0: # Cartesian
            self.lbl_equation_help.setText("方程 (Equation): e.g. exp(-(x**2+y**2)/w0**2)")
            self.txt_equation.setPlaceholderText("Available: x, y, r, phi, z, np, sqrt, exp...")
        else: # Cylindrical
            self.lbl_equation_help.setText("方程 (Equation): e.g. exp(-r**2/w0**2) * exp(1j*m*phi)")
            self.txt_equation.setPlaceholderText("Available: r, phi, z, np, sqrt, exp... (phi in [-pi, pi])")

    def update_source_ui(self, index):
        self.source_stack.setCurrentIndex(index)
        
        # Update Equation Display
        eqs = [
            r"E(x,y) = A \cdot e^{i(k_x x + k_y y)}",
            r"E(r,z) = A \frac{w_0}{w(z)} e^{-\frac{r^2}{w(z)^2}} e^{-i(kz + \frac{kr^2}{2R(z)} - \psi(z))}",
            r"E(r,\phi) \propto (\frac{r\sqrt{2}}{w_0})^{|l|} L_p^{|l|}(\frac{2r^2}{w_0^2}) e^{-\frac{r^2}{w_0^2}} e^{il\phi}",
            r"E(r,\phi) \propto J_n(k_r r) e^{in\phi}", # Bessel placeholder
            r"E_{custom} = f(x, y, r, \phi, z)"
        ]
        if 0 <= index < len(eqs):
            self.eq_display.update_equation(eqs[index])
            
        if index == 4:
            self.update_custom_help()

    def add_custom_variable(self):
        self.table_vars.blockSignals(True)
        try:
            row = self.table_vars.rowCount()
            self.table_vars.insertRow(row)
            self.table_vars.setItem(row, 0, QTableWidgetItem(f"var{row}"))
            self.table_vars.setItem(row, 1, QTableWidgetItem("1.0"))
        finally:
            self.table_vars.blockSignals(False)
        # Trigger sync once after complete
        self.sync_source_to_config()

    def del_custom_variable(self):
        row = self.table_vars.currentRow()
        if row >= 0:
            self.table_vars.removeRow(row)

    def create_unit_spinbox(self, default_value, default_unit="um"):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        sb = QDoubleSpinBox()
        sb.setRange(0.001, 1e9) # 0.001 um to large number
        sb.setDecimals(3)
        sb.setValue(default_value)
        
        combo = QComboBox()
        combo.addItems(["um", "mm"])
        combo.setCurrentText(default_unit)
        
        layout.addWidget(sb, 1)
        layout.addWidget(combo, 0)
        
        # Store state for conversion
        combo.setProperty("last_unit", default_unit)
        
        def on_unit_changed(new_unit):
            old_unit = combo.property("last_unit")
            val = sb.value()
            
            # Convert to meters
            val_m = 0
            if old_unit == "mm": val_m = val * 1e-3
            elif old_unit == "um": val_m = val * 1e-6
            
            # Convert to new unit
            new_val = 0
            if new_unit == "mm": new_val = val_m * 1e3
            elif new_unit == "um": new_val = val_m * 1e6
            
            sb.setValue(new_val)
            combo.setProperty("last_unit", new_unit)
            
        combo.currentTextChanged.connect(on_unit_changed)
        
        return widget, sb, combo

    def create_geometric_config_widget(self, prefix, comp_type='trans'):
        """
        Create a widget for geometric shape configuration (Trans or Phase)
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Shape Selector
        layout.addWidget(QLabel("选择形状 (Select Shape):"))
        combo_shape = QComboBox()
        combo_shape.addItems(["圆环 (Annular)", "圆形 (Circle)", "矩形 (Rectangle)", "多边形 (Polygon)"])
        layout.addWidget(combo_shape)
        
        # Shape Parameters Stack
        shape_stack = QStackedWidget()
        
        # Value Label/Range
        val_label = "透射率 (Transmission):" if comp_type == 'trans' else "相位 (Phase):"
        val_range = (0.0, 1.0) if comp_type == 'trans' else (-100.0, 100.0) # Phase range loose
        
        # Helper for param row
        def create_param_row(label, default, suffix=" um", min_val=-100000.0, max_val=100000.0):
            sb = QDoubleSpinBox()
            sb.setRange(min_val, max_val)
            sb.setValue(default)
            if suffix: sb.setSuffix(suffix)
            return label, sb
            
        # 0: Annular
        page_annular = QWidget()
        pa_layout = QFormLayout(page_annular)
        _, sb_ann_r_in = create_param_row("内径 (Inner R):", 0.0, min_val=0.0)
        _, sb_ann_r_out = create_param_row("外径 (Outer R):", 100.0, min_val=0.0)
        _, sb_ann_cx = create_param_row("中心 X (Center X):", 0.0)
        _, sb_ann_cy = create_param_row("中心 Y (Center Y):", 0.0)
        _, sb_ann_val = create_param_row(val_label, 1.0 if comp_type=='trans' else 0.0, suffix="", min_val=val_range[0], max_val=val_range[1])
        if comp_type == 'trans': sb_ann_val.setSingleStep(0.1)
        
        pa_layout.addRow("内径 (Inner R):", sb_ann_r_in)
        pa_layout.addRow("外径 (Outer R):", sb_ann_r_out)
        pa_layout.addRow("中心 X (Center X):", sb_ann_cx)
        pa_layout.addRow("中心 Y (Center Y):", sb_ann_cy)
        pa_layout.addRow(val_label, sb_ann_val)
        shape_stack.addWidget(page_annular)
        
        # 1: Circle
        page_circle = QWidget()
        pc_layout = QFormLayout(page_circle)
        _, sb_cir_r = create_param_row("半径 (Radius):", 100.0, min_val=0.0)
        _, sb_cir_cx = create_param_row("中心 X (Center X):", 0.0)
        _, sb_cir_cy = create_param_row("中心 Y (Center Y):", 0.0)
        _, sb_cir_val = create_param_row(val_label, 1.0 if comp_type=='trans' else 0.0, suffix="", min_val=val_range[0], max_val=val_range[1])
        if comp_type == 'trans': sb_cir_val.setSingleStep(0.1)

        pc_layout.addRow("半径 (Radius):", sb_cir_r)
        pc_layout.addRow("中心 X (Center X):", sb_cir_cx)
        pc_layout.addRow("中心 Y (Center Y):", sb_cir_cy)
        pc_layout.addRow(val_label, sb_cir_val)
        shape_stack.addWidget(page_circle)
        
        # 2: Rectangle
        page_rect = QWidget()
        pr_layout = QFormLayout(page_rect)
        _, sb_rect_w = create_param_row("宽度 (Width):", 200.0, min_val=0.0)
        _, sb_rect_h = create_param_row("高度 (Height):", 200.0, min_val=0.0)
        _, sb_rect_cx = create_param_row("中心 X (Center X):", 0.0)
        _, sb_rect_cy = create_param_row("中心 Y (Center Y):", 0.0)
        _, sb_rect_rot = create_param_row("旋转 (Rotation):", 0.0, suffix=" deg", min_val=-360.0, max_val=360.0)
        _, sb_rect_val = create_param_row(val_label, 1.0 if comp_type=='trans' else 0.0, suffix="", min_val=val_range[0], max_val=val_range[1])
        if comp_type == 'trans': sb_rect_val.setSingleStep(0.1)

        pr_layout.addRow("宽度 (Width):", sb_rect_w)
        pr_layout.addRow("高度 (Height):", sb_rect_h)
        pr_layout.addRow("中心 X (Center X):", sb_rect_cx)
        pr_layout.addRow("中心 Y (Center Y):", sb_rect_cy)
        pr_layout.addRow("旋转 (Rotation):", sb_rect_rot)
        pr_layout.addRow(val_label, sb_rect_val)
        shape_stack.addWidget(page_rect)
        
        # 3: Polygon (New)
        page_poly = QWidget()
        pp_layout = QVBoxLayout(page_poly)
        poly_editor = PolygonEditorWidget()
        _, sb_poly_val = create_param_row(val_label, 1.0 if comp_type=='trans' else 0.0, suffix="", min_val=val_range[0], max_val=val_range[1])
        if comp_type == 'trans': sb_poly_val.setSingleStep(0.1)
        
        pp_layout.addWidget(QLabel("多边形顶点 (Polygon Vertices):"))
        pp_layout.addWidget(poly_editor)
        pp_layout.addWidget(QLabel(val_label)) # Form layout mimic
        pp_layout.addWidget(sb_poly_val)
        
        shape_stack.addWidget(page_poly)
        
        layout.addWidget(shape_stack)
        
        combo_shape.currentIndexChanged.connect(shape_stack.setCurrentIndex)
        
        # Connect signals to sync
        combo_shape.currentIndexChanged.connect(self.sync_source_to_config)
        for w in [sb_ann_r_in, sb_ann_r_out, sb_ann_cx, sb_ann_cy, sb_ann_val,
                  sb_cir_r, sb_cir_cx, sb_cir_cy, sb_cir_val,
                  sb_rect_w, sb_rect_h, sb_rect_cx, sb_rect_cy, sb_rect_rot, sb_rect_val,
                  sb_poly_val]:
            w.valueChanged.connect(self.sync_source_to_config)
            
        poly_editor.dataChanged.connect(self.sync_source_to_config)
        
        # Store refs dynamically
        id_prefix = f"{prefix}_{comp_type}"
        setattr(self, f"combo_shape_{id_prefix}", combo_shape)
        setattr(self, f"stack_shape_{id_prefix}", shape_stack)
        
        # Annular
        setattr(self, f"sb_ann_r_in_{id_prefix}", sb_ann_r_in)
        setattr(self, f"sb_ann_r_out_{id_prefix}", sb_ann_r_out)
        setattr(self, f"sb_ann_cx_{id_prefix}", sb_ann_cx)
        setattr(self, f"sb_ann_cy_{id_prefix}", sb_ann_cy)
        setattr(self, f"sb_ann_val_{id_prefix}", sb_ann_val)
        
        # Circle
        setattr(self, f"sb_cir_r_{id_prefix}", sb_cir_r)
        setattr(self, f"sb_cir_cx_{id_prefix}", sb_cir_cx)
        setattr(self, f"sb_cir_cy_{id_prefix}", sb_cir_cy)
        setattr(self, f"sb_cir_val_{id_prefix}", sb_cir_val)
        
        # Rect
        setattr(self, f"sb_rect_w_{id_prefix}", sb_rect_w)
        setattr(self, f"sb_rect_h_{id_prefix}", sb_rect_h)
        setattr(self, f"sb_rect_cx_{id_prefix}", sb_rect_cx)
        setattr(self, f"sb_rect_cy_{id_prefix}", sb_rect_cy)
        setattr(self, f"sb_rect_rot_{id_prefix}", sb_rect_rot)
        setattr(self, f"sb_rect_val_{id_prefix}", sb_rect_val)
        
        # Poly
        setattr(self, f"poly_editor_{id_prefix}", poly_editor)
        setattr(self, f"sb_poly_val_{id_prefix}", sb_poly_val)
        
        return widget

    def create_modulator_tab_generic(self, title, prefix):
        """
        Generic creator for modulator tabs
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Position
        pos_group = QGroupBox("位置 (Position)")
        pos_layout = QFormLayout()
        
        default_z = 10000.0 if prefix == 'mod1' else 20000.0
        w_z, sb_z, combo_z = self.create_unit_spinbox(default_z, "um")
        
        pos_layout.addRow("距离光源 (Distance):", w_z)
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        setattr(self, f"sb_{prefix}_z", sb_z)
        setattr(self, f"combo_{prefix}_z_unit", combo_z)
        
        # Device Type
        type_group = QGroupBox("器件类型 (Device Type)")
        t_layout = QVBoxLayout()
        combo = QComboBox()
        combo.addItems(["自定义掩膜 (Custom Mask)", "理想普通透镜 (Ideal Lens)", 
                        "理想柱透镜-X (Cylindrical Lens X)", "理想柱透镜-Y (Cylindrical Lens Y)"])
        t_layout.addWidget(combo)
        type_group.setLayout(t_layout)
        layout.addWidget(type_group)
        
        setattr(self, f"combo_{prefix}_type", combo)
        
        # Stacked Params
        stack = QStackedWidget()
        
        # 0: Custom Mask
        page_mask = QWidget()
        pm_layout = QVBoxLayout(page_mask)
        
        # Tabs for Definition Method
        mask_tabs = QTabWidget()
        
        # Tab 1: File Import
        tab_file = QWidget()
        tf_layout = QVBoxLayout(tab_file)
        
        # Phase File
        phase_group = QGroupBox("相位调制 (Phase)")
        ph_layout = QVBoxLayout()
        btn_phase = QPushButton("加载相位分布 (Load Phase)")
        lbl_phase = QLabel("未加载 (Not Loaded)")
        btn_phase.clicked.connect(lambda: self.load_data(f'phase{prefix[-1]}'))
        ph_layout.addWidget(btn_phase)
        ph_layout.addWidget(lbl_phase)
        phase_group.setLayout(ph_layout)
        tf_layout.addWidget(phase_group)
        
        setattr(self, f"btn_load_phase{prefix[-1]}", btn_phase)
        setattr(self, f"lbl_phase{prefix[-1]}_status", lbl_phase)
        
        # Amp File
        amp_group = QGroupBox("透射率 (Transmission)")
        am_layout = QVBoxLayout()
        btn_amp = QPushButton("加载透射率 (Load Trans)")
        lbl_amp = QLabel("未加载 (Not Loaded)")
        btn_amp.clicked.connect(lambda: self.load_data(f'amp{prefix[-1]}'))
        am_layout.addWidget(btn_amp)
        am_layout.addWidget(lbl_amp)
        amp_group.setLayout(am_layout)
        tf_layout.addWidget(amp_group)
        
        setattr(self, f"btn_load_amp{prefix[-1]}", btn_amp)
        setattr(self, f"lbl_amp{prefix[-1]}_status", lbl_amp)
        
        tf_layout.addStretch()
        mask_tabs.addTab(tab_file, "文件导入 (File Import)")
        
        # Tab 2: Parameter Definition
        tab_param = QWidget()
        tp_layout = QVBoxLayout(tab_param)
        
        # --- Transmission Distribution ---
        grp_trans = QGroupBox("透射率分布 (Transmittance Distribution)")
        gt_layout = QVBoxLayout(grp_trans)
        
        combo_trans_mode = QComboBox()
        combo_trans_mode.addItems(["公式定义 (Formula)", "几何形状 (Geometric Shape)"])
        gt_layout.addWidget(combo_trans_mode)
        
        stack_trans = QStackedWidget()
        
        # Formula
        fw_trans = FormulaWidget(formula_type='trans')
        fw_trans.formulaChanged.connect(lambda _: self.sync_source_to_config())
        stack_trans.addWidget(fw_trans)
        
        # Geometric
        gw_trans = self.create_geometric_config_widget(prefix, 'trans')
        stack_trans.addWidget(gw_trans)
        
        gt_layout.addWidget(stack_trans)
        combo_trans_mode.currentIndexChanged.connect(stack_trans.setCurrentIndex)
        combo_trans_mode.currentIndexChanged.connect(self.sync_source_to_config)
        
        tp_layout.addWidget(grp_trans)
        
        # --- Phase Distribution ---
        grp_phase = QGroupBox("相位分布 (Phase Distribution)")
        gp_layout = QVBoxLayout(grp_phase)
        
        combo_phase_mode = QComboBox()
        combo_phase_mode.addItems(["公式定义 (Formula)", "几何形状 (Geometric Shape)"])
        gp_layout.addWidget(combo_phase_mode)
        
        stack_phase = QStackedWidget()
        
        # Formula
        fw_phase = FormulaWidget(formula_type='phase')
        fw_phase.formulaChanged.connect(lambda _: self.sync_source_to_config())
        stack_phase.addWidget(fw_phase)
        
        # Geometric
        gw_phase = self.create_geometric_config_widget(prefix, 'phase')
        stack_phase.addWidget(gw_phase)
        
        gp_layout.addWidget(stack_phase)
        combo_phase_mode.currentIndexChanged.connect(stack_phase.setCurrentIndex)
        combo_phase_mode.currentIndexChanged.connect(self.sync_source_to_config)
        
        tp_layout.addWidget(grp_phase)
        
        mask_tabs.addTab(tab_param, "参数定义法 (Param Definition)")
        
        pm_layout.addWidget(mask_tabs)
        
        # Store refs
        setattr(self, f"mask_tabs_{prefix}", mask_tabs)
        setattr(self, f"combo_trans_mode_{prefix}", combo_trans_mode)
        setattr(self, f"fw_trans_{prefix}", fw_trans)
        setattr(self, f"combo_phase_mode_{prefix}", combo_phase_mode)
        setattr(self, f"fw_phase_{prefix}", fw_phase)
        
        mask_tabs.currentChanged.connect(self.sync_source_to_config)
        
        stack.addWidget(page_mask)
        
        # 1: Lens Params
        page_lens = QWidget()
        pl_layout = QFormLayout(page_lens)
        
        w_D, sb_D, combo_D = self.create_unit_spinbox(25400.0, "um")
        w_f, sb_f, combo_f = self.create_unit_spinbox(100000.0, "um")
        
        sb_NA = QDoubleSpinBox()
        sb_NA.setRange(0.001, 0.999)
        sb_NA.setValue(0.127)
        sb_NA.setDecimals(3)
        
        # Inter-dependency logic
        def get_val_m(sb, cmb):
            val = sb.value()
            unit = cmb.currentText()
            if unit == "mm": return val * 1e-3
            elif unit == "um": return val * 1e-6
            return val
            
        def set_val_from_m(sb, cmb, val_m):
            unit = cmb.currentText()
            if unit == "mm": sb.setValue(val_m * 1e3)
            elif unit == "um": sb.setValue(val_m * 1e6)

        def update_na():
            f_m = get_val_m(sb_f, combo_f)
            D_m = get_val_m(sb_D, combo_D)
            if f_m > 0:
                na = D_m / (2 * f_m)
                sb_NA.blockSignals(True)
                sb_NA.setValue(min(na, 0.999))
                sb_NA.blockSignals(False)
        
        def update_f():
            na = sb_NA.value()
            D_m = get_val_m(sb_D, combo_D)
            if na > 0:
                f_m = D_m / (2 * na)
                sb_f.blockSignals(True)
                set_val_from_m(sb_f, combo_f, f_m)
                sb_f.blockSignals(False)
        
        sb_D.valueChanged.connect(update_na)
        sb_f.valueChanged.connect(update_na)
        sb_NA.valueChanged.connect(update_f)
        
        sb_D.valueChanged.connect(self.sync_source_to_config)
        sb_f.valueChanged.connect(self.sync_source_to_config)
        sb_NA.valueChanged.connect(self.sync_source_to_config)
        
        pl_layout.addRow("直径 D (Diameter):", w_D)
        pl_layout.addRow("焦距 f (Focal Length):", w_f)
        pl_layout.addRow("数值孔径 NA:", sb_NA)
        
        stack.addWidget(page_lens) 
        stack.addWidget(QWidget()) # Cyl X placeholder
        
        setattr(self, f"sb_{prefix}_D", sb_D)
        setattr(self, f"combo_{prefix}_D_unit", combo_D)
        setattr(self, f"sb_{prefix}_f", sb_f)
        setattr(self, f"combo_{prefix}_f_unit", combo_f)
        setattr(self, f"sb_{prefix}_NA", sb_NA)
        
        combo.currentIndexChanged.connect(lambda idx: stack.setCurrentIndex(0 if idx == 0 else 1))
        combo.currentIndexChanged.connect(self.sync_source_to_config)
        
        layout.addWidget(stack)
        setattr(self, f"stack_{prefix}", stack)
        
        # Affected Polarization
        pol_group = QGroupBox("作用偏振 (Affected Polarization)")
        pol_layout = QVBoxLayout()
        
        cb_lin = QCheckBox("X° 线偏振 (Linear X)")
        cb_lcp = QCheckBox("左旋圆偏振 (LCP)")
        cb_rcp = QCheckBox("右旋圆偏振 (RCP)")
        cb_unpol = QCheckBox("无偏振 (Unpolarized)")
        cb_unpol.setChecked(True)
        
        cb_lin.stateChanged.connect(self.sync_source_to_config)
        cb_lcp.stateChanged.connect(self.sync_source_to_config)
        cb_rcp.stateChanged.connect(self.sync_source_to_config)
        cb_unpol.stateChanged.connect(self.sync_source_to_config)
        
        pol_layout.addWidget(cb_lin)
        pol_layout.addWidget(cb_lcp)
        pol_layout.addWidget(cb_rcp)
        pol_layout.addWidget(cb_unpol)
        pol_group.setLayout(pol_layout)
        layout.addWidget(pol_group)
        
        setattr(self, f"cb_{prefix}_pol_lin_x", cb_lin)
        setattr(self, f"cb_{prefix}_pol_lcp", cb_lcp)
        setattr(self, f"cb_{prefix}_pol_rcp", cb_rcp)
        setattr(self, f"cb_{prefix}_pol_unpol", cb_unpol)
        
        if prefix == 'mod2':
            angle_group = QGroupBox("角度-透射率 (Angle-Transmission)")
            angle_layout = QVBoxLayout()
            btn_angle = QPushButton("加载曲线 (Load Curve)")
            lbl_angle = QLabel("未加载 (Not Loaded)")
            btn_angle.clicked.connect(lambda: self.load_data('angle2'))
            angle_layout.addWidget(btn_angle)
            angle_layout.addWidget(lbl_angle)
            angle_group.setLayout(angle_layout)
            layout.addWidget(angle_group)
            
            setattr(self, f"btn_load_angle2", btn_angle)
            setattr(self, f"lbl_angle2_status", lbl_angle)
        
        layout.addStretch()
        return tab

    def create_modulator1_tab(self):
        return self.create_modulator_tab_generic("调制平面1 (Mod 1)", "mod1")

    def create_modulator2_tab(self):
        return self.create_modulator_tab_generic("调制平面2 (Mod 2)", "mod2")

    def create_monitor_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        manage_group = QGroupBox("监视器管理 (Monitor Management)")
        manage_layout = QHBoxLayout()
        
        self.monitor_list = QListWidget()
        self.monitor_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.monitor_list.currentRowChanged.connect(self.load_monitor_settings)
        
        # Keyboard shortcut for Select All
        self.shortcut_select_all = QShortcut(QKeySequence("Ctrl+A"), self.monitor_list)
        self.shortcut_select_all.activated.connect(self.monitor_list.selectAll)
        
        btn_layout = QVBoxLayout()
        self.btn_add_monitor = QPushButton("添加监视器 (Add)")
        self.btn_add_monitor.clicked.connect(self.add_monitor)
        self.btn_del_monitor = QPushButton("删除监视器 (Delete)")
        self.btn_del_monitor.clicked.connect(self.delete_monitor)
        self.btn_add_array = QPushButton("添加阵列 (Add Array)")
        self.btn_add_array.clicked.connect(self.add_monitor_array)
        self.btn_batch_edit = QPushButton("批量编辑 (Batch Edit)")
        self.btn_batch_edit.clicked.connect(self.batch_edit_monitors)
        
        btn_layout.addWidget(self.btn_add_monitor)
        btn_layout.addWidget(self.btn_del_monitor)
        btn_layout.addWidget(self.btn_add_array)
        btn_layout.addWidget(self.btn_batch_edit)
        btn_layout.addStretch()
        
        manage_layout.addWidget(self.monitor_list, 2)
        manage_layout.addLayout(btn_layout, 1)
        manage_group.setLayout(manage_layout)
        layout.addWidget(manage_group)
        
        self.settings_group = QGroupBox("监视器设置 (Settings)")
        settings_layout = QFormLayout()
        
        # Position with unit
        self.w_mon_pos, self.sb_mon_pos, self.combo_mon_pos_unit = self.create_unit_spinbox(0.0, "mm")
        self.sb_mon_pos.setRange(-100000, 100000)
        self.sb_mon_pos.valueChanged.connect(self.update_current_monitor)
        self.combo_mon_pos_unit.currentTextChanged.connect(self.update_current_monitor)
        
        self.combo_mon_plane = QComboBox()
        self.combo_mon_plane.addItems(["XY Plane (Normal Z)", "YZ Plane (Normal X)", "XZ Plane (Normal Y)"])
        self.combo_mon_plane.currentIndexChanged.connect(self.update_current_monitor)
        self.combo_mon_plane.currentIndexChanged.connect(self.update_monitor_ui_state)
        
        self.combo_mon_type = QComboBox()
        self.combo_mon_type.addItems(["Intensity (|E|^2)", "Complex Field (E)"])
        self.combo_mon_type.currentIndexChanged.connect(self.update_current_monitor)
        
        self.lbl_mon_pos_label = QLabel("位置 (Position):")
        self.lbl_mon_fixed_dim = QLabel("固定维度 (Fixed Dim): Z")
        self.lbl_mon_fixed_dim.setStyleSheet("color: gray; font-style: italic;")
        
        # Range inputs
        self.range_group = QGroupBox("范围设置 (Range Settings)")
        range_layout = QFormLayout()
        
        self.lbl_range1 = QLabel("Range 1:")
        self.sb_range1_min = QDoubleSpinBox()
        self.sb_range1_min.setRange(-100000, 100000)
        self.sb_range1_min.setSuffix(" um")
        self.sb_range1_max = QDoubleSpinBox()
        self.sb_range1_max.setRange(-100000, 100000)
        self.sb_range1_max.setSuffix(" um")
        
        r1_layout = QHBoxLayout()
        r1_layout.addWidget(QLabel("Min:"))
        r1_layout.addWidget(self.sb_range1_min)
        r1_layout.addWidget(QLabel("Max:"))
        r1_layout.addWidget(self.sb_range1_max)
        
        self.lbl_range2 = QLabel("Range 2:")
        self.sb_range2_min = QDoubleSpinBox()
        self.sb_range2_min.setRange(-100000, 100000)
        self.sb_range2_min.setSuffix(" um")
        self.sb_range2_max = QDoubleSpinBox()
        self.sb_range2_max.setRange(-100000, 100000)
        self.sb_range2_max.setSuffix(" um")
        
        r2_layout = QHBoxLayout()
        r2_layout.addWidget(QLabel("Min:"))
        r2_layout.addWidget(self.sb_range2_min)
        r2_layout.addWidget(QLabel("Max:"))
        r2_layout.addWidget(self.sb_range2_max)
        
        range_layout.addRow(self.lbl_range1, r1_layout)
        range_layout.addRow(self.lbl_range2, r2_layout)
        self.range_group.setLayout(range_layout)
        
        # Connect range changes
        self.sb_range1_min.valueChanged.connect(self.update_current_monitor)
        self.sb_range1_max.valueChanged.connect(self.update_current_monitor)
        self.sb_range2_min.valueChanged.connect(self.update_current_monitor)
        self.sb_range2_max.valueChanged.connect(self.update_current_monitor)
        
        settings_layout.addRow(self.lbl_mon_pos_label, self.w_mon_pos)
        settings_layout.addRow("", self.lbl_mon_fixed_dim) # Indented label
        settings_layout.addRow("切面 (Plane):", self.combo_mon_plane)
        settings_layout.addRow("数据类型 (Data Type):", self.combo_mon_type)
        settings_layout.addRow(self.range_group)
        
        # Output Components
        comp_group = QGroupBox("额外输出分量 (Extra Components)")
        comp_layout = QHBoxLayout()
        
        self.cb_mon_ex = QCheckBox("Ex")
        self.cb_mon_ey = QCheckBox("Ey")
        self.cb_mon_ez = QCheckBox("Ez")
        
        # Update visibility based on plane type
        # Logic: 
        # XY (0): Ex, Ey (Ez usually small but can be calc) -> Requirement 3a: "XY: Ex, Ey"
        # XZ (2): Ex, Ez
        # YZ (1): Ey, Ez
        # Actually user requirement 3a says:
        # "XY plane: only show Ex Ey boxes"
        # "XZ plane: only show Ex Ez boxes"
        # "YZ plane: only show Ey Ez boxes"
        
        comp_layout.addWidget(self.cb_mon_ex)
        comp_layout.addWidget(self.cb_mon_ey)
        comp_layout.addWidget(self.cb_mon_ez)
        comp_group.setLayout(comp_layout)
        settings_layout.addRow(comp_group)
        
        self.cb_mon_ex.stateChanged.connect(self.update_current_monitor)
        self.cb_mon_ey.stateChanged.connect(self.update_current_monitor)
        self.cb_mon_ez.stateChanged.connect(self.update_current_monitor)
        
        # Also connect to sync config!
        self.cb_mon_ex.stateChanged.connect(self.sync_source_to_config)
        self.cb_mon_ey.stateChanged.connect(self.sync_source_to_config)
        self.cb_mon_ez.stateChanged.connect(self.sync_source_to_config)
        
        self.btn_export_data = QPushButton("导出监视器数据 (Export Data)")
        self.btn_export_data.clicked.connect(self.export_monitor_data)
        settings_layout.addRow(self.btn_export_data)
        
        self.settings_group.setLayout(settings_layout)
        self.settings_group.setEnabled(False)
        layout.addWidget(self.settings_group)
        
        layout.addStretch()
        return tab

    def add_monitor(self):
        name = f"Monitor {len(self.monitors) + 1}"
        monitor = {
            'name': name,
            'z': 0.0, # Legacy
            'pos': 0.0,
            'pos_unit': 'mm',
            'plane': 0,
            'type': 0,
            'output_components': []
        }
        if self.detect_conflict(monitor):
            QMessageBox.warning(self, "Conflict", "A monitor already exists at this position.")
            return
            
        self.monitors.append(monitor)
        self.monitor_list.addItem(name)
        self.monitor_list.setCurrentRow(len(self.monitors) - 1)

    def detect_conflict(self, monitor, exclude_index=-1):
        """
        冲突检测 (Conflict Detection)
        """
        # Convert monitor pos to um for comparison
        mon_unit = monitor.get('pos_unit', 'um')
        mon_pos = monitor.get('pos', monitor.get('z', 0))
        mon_pos_um = mon_pos * (1000 if mon_unit == 'mm' else 1)
        
        for i, m in enumerate(self.monitors):
            if i == exclude_index:
                continue
            
            m_unit = m.get('pos_unit', 'um')
            m_pos = m.get('pos', m.get('z', 0))
            m_pos_um = m_pos * (1000 if m_unit == 'mm' else 1)
            
            if abs(m_pos_um - mon_pos_um) < 1e-3 and m['plane'] == monitor['plane']: # 1e-3 um tolerance
                return True
        return False

    def delete_monitor(self):
        selected_items = self.monitor_list.selectedItems()
        if not selected_items:
            return
            
        count = len(selected_items)
        msg = f"确定要删除选中的 {count} 个监视器吗？\n(Are you sure you want to delete {count} selected monitors?)"
        reply = QMessageBox.question(self, "Delete Monitors", msg, 
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # Get rows and sort descending to avoid index shift issues
            rows = []
            for item in selected_items:
                rows.append(self.monitor_list.row(item))
            
            rows.sort(reverse=True)
            
            for row in rows:
                self.monitor_list.takeItem(row)
                del self.monitors[row]
            
            if not self.monitors:
                self.settings_group.setEnabled(False)
            
            self.window().statusBar().showMessage(f"已删除 {count} 个监视器 (Deleted {count} monitors).")

    def add_monitor_array(self):
        dialog = MonitorArrayDialog(self)
        if dialog.exec():
            count, start, start_unit, spacing, spacing_unit, prefix = dialog.get_values()
            
            # Convert to um for calculation
            start_um = start * (1000 if start_unit == 'mm' else 1)
            spacing_um = spacing * (1000 if spacing_unit == 'mm' else 1)
            
            for i in range(count):
                z_um = start_um + i * spacing_um
                name = f"{prefix}{len(self.monitors) + 1}"
                
                # Store in mm if that was preferred start unit, or um?
                # Let's default to mm if start was mm, else um
                unit = start_unit
                pos = z_um / 1000 if unit == 'mm' else z_um
                
                monitor = {
                    'name': name, 
                    'pos': pos, 
                    'pos_unit': unit,
                    'z': pos, # Legacy
                    'plane': 0, 
                    'type': 0,
                    'output_components': []
                }
                
                if not self.detect_conflict(monitor):
                    self.monitors.append(monitor)
                    self.monitor_list.addItem(name)

    def batch_edit_monitors(self):
        """
        批量编辑 (Batch Edit)
        """
        # For simplicity, set all monitors to current settings
        row = self.monitor_list.currentRow()
        if row < 0: return
        
        reply = QMessageBox.question(self, "Batch Edit", 
                                   "Apply current monitor settings (Plane, Type) to ALL monitors?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            current = self.monitors[row]
            for m in self.monitors:
                m['plane'] = current['plane']
                m['type'] = current['type']
            QMessageBox.information(self, "Success", "Settings applied to all monitors.")

    def export_monitor_data(self):
        """
        数据导出 (Data Export)
        """
        row = self.monitor_list.currentRow()
        if row < 0: return
        
        name = self.monitors[row]['name']
        # This logic needs to connect to the visualization panel's data
        # For now, just show a message or trigger a signal
        self.parent().status_bar.showMessage(f"Exporting {name}...")
        # Actually triggered from MainWindow in real implementation

    def load_monitor_settings(self, row):
        if row >= 0 and row < len(self.monitors):
            monitor = self.monitors[row]
            self.settings_group.setEnabled(True)
            self.sb_mon_pos.blockSignals(True)
            self.combo_mon_pos_unit.blockSignals(True)
            self.combo_mon_plane.blockSignals(True)
            self.combo_mon_type.blockSignals(True)
            
            self.sb_range1_min.blockSignals(True)
            self.sb_range1_max.blockSignals(True)
            self.sb_range2_min.blockSignals(True)
            self.sb_range2_max.blockSignals(True)
            
            # Load Unit first to set scaling correct in spinbox helper?
            # No, create_unit_spinbox helper reacts to unit change by scaling value.
            # So if I set unit first, it might scale the OLD value.
            # I should set unit (without signal) then set value.
            
            unit = monitor.get('pos_unit', 'um') # Default um if missing (legacy)
            # If legacy 'z' is present but no 'pos', assume 'z' is in um.
            pos = monitor.get('pos', monitor.get('z', 0.0))
            
            # Ensure consistency: if we load legacy (um), but want to display as 'mm', we must convert.
            # But here we just load what is stored.
            
            self.combo_mon_pos_unit.setCurrentText(unit)
            # Important: sync the "last_unit" property so future changes calculate correctly
            self.combo_mon_pos_unit.setProperty("last_unit", unit)
            
            self.sb_mon_pos.setValue(pos)
            
            self.combo_mon_plane.setCurrentIndex(monitor['plane'])
            self.combo_mon_type.setCurrentIndex(monitor['type'])
            
            # Load ranges (default to huge range if missing)
            self.sb_range1_min.setValue(monitor.get('range1_min', -1000.0))
            self.sb_range1_max.setValue(monitor.get('range1_max', 1000.0))
            self.sb_range2_min.setValue(monitor.get('range2_min', -1000.0))
            self.sb_range2_max.setValue(monitor.get('range2_max', 1000.0))
            
            self.sb_mon_pos.blockSignals(False)
            self.combo_mon_pos_unit.blockSignals(False)
            self.combo_mon_plane.blockSignals(False)
            self.combo_mon_type.blockSignals(False)
            
            self.sb_range1_min.blockSignals(False)
            self.sb_range1_max.blockSignals(False)
            self.sb_range2_min.blockSignals(False)
            self.sb_range2_max.blockSignals(False)
            
            # Load components
            comps = monitor.get('output_components', [])
            self.cb_mon_ex.blockSignals(True)
            self.cb_mon_ey.blockSignals(True)
            self.cb_mon_ez.blockSignals(True)
            
            self.cb_mon_ex.setChecked('Ex' in comps)
            self.cb_mon_ey.setChecked('Ey' in comps)
            self.cb_mon_ez.setChecked('Ez' in comps)
            
            self.cb_mon_ex.blockSignals(False)
            self.cb_mon_ey.blockSignals(False)
            self.cb_mon_ez.blockSignals(False)
            
            self.update_monitor_ui_state()
        else:
            self.settings_group.setEnabled(False)

    def update_monitor_ui_state(self):
        idx = self.combo_mon_plane.currentIndex()
        
        # Grid bounds check
        nx = self.sb_nx.value()
        ny = self.sb_ny.value()
        dx = self.sb_dx.value() # um
        dy = self.sb_dy.value() # um
        
        x_max = (nx * dx) / 2
        y_max = (ny * dy) / 2
        
        if idx == 0: # XY Plane (Normal Z)
            self.lbl_mon_pos_label.setText("设置 Z 轴位置 (Set Z Position):")
            self.lbl_mon_fixed_dim.setText("固定维度 (Fixed Dimension): Z-Axis (Propagation)")
            # self.sb_mon_pos.setSuffix(" um") # Handled by combo now
            # Z range is technically infinite, but usually positive
            # self.sb_mon_pos.setRange(-100000, 100000)
            self.sb_mon_pos.setStyleSheet("") # Clear warning
            
            self.lbl_range1.setText("X Range:")
            self.lbl_range2.setText("Y Range:")
            
        elif idx == 1: # YZ Plane (Normal X)
            self.lbl_mon_pos_label.setText("设置 X 轴位置 (Set X Position):")
            self.lbl_mon_fixed_dim.setText("固定维度 (Fixed Dimension): X-Axis")
            # self.sb_mon_pos.setSuffix(" um")
            # X range validation
            # self.sb_mon_pos.setRange(-x_max * 1.5, x_max * 1.5) # Allow slight over, warn if out
            
            self.lbl_range1.setText("Y Range:")
            self.lbl_range2.setText("Z Range:")
            
        elif idx == 2: # XZ Plane (Normal Y)
            self.lbl_mon_pos_label.setText("设置 Y 轴位置 (Set Y Position):")
            self.lbl_mon_fixed_dim.setText("固定维度 (Fixed Dimension): Y-Axis")
            # self.sb_mon_pos.setSuffix(" um")
            # Y range validation
            # self.sb_mon_pos.setRange(-y_max * 1.5, y_max * 1.5)

            self.lbl_range1.setText("X Range:")
            self.lbl_range2.setText("Z Range:")

        # Validate current value
        self.validate_monitor_pos()
        self.validate_monitor_ranges()

    def validate_monitor_ranges(self):
        # Validate Min < Max
        r1_min = self.sb_range1_min.value()
        r1_max = self.sb_range1_max.value()
        r2_min = self.sb_range2_min.value()
        r2_max = self.sb_range2_max.value()
        
        valid = True
        if r1_min >= r1_max:
            self.sb_range1_min.setStyleSheet("border: 1px solid red;")
            self.sb_range1_max.setStyleSheet("border: 1px solid red;")
            valid = False
        else:
            self.sb_range1_min.setStyleSheet("")
            self.sb_range1_max.setStyleSheet("")
            
        if r2_min >= r2_max:
            self.sb_range2_min.setStyleSheet("border: 1px solid red;")
            self.sb_range2_max.setStyleSheet("border: 1px solid red;")
            valid = False
        else:
            self.sb_range2_min.setStyleSheet("")
            self.sb_range2_max.setStyleSheet("")
            
        # Disable OK button? No OK button here, real-time update.
        # But we should maybe disable Run if invalid?
        # For now just visual cue.


    def validate_monitor_pos(self):
        idx = self.combo_mon_plane.currentIndex()
        val = self.sb_mon_pos.value()
        unit = self.combo_mon_pos_unit.currentText()
        
        # Convert to um for validation
        val_um = val * (1000 if unit == 'mm' else 1)
        
        nx = self.sb_nx.value()
        ny = self.sb_ny.value()
        dx = self.sb_dx.value() # um
        dy = self.sb_dy.value() # um
        
        x_max = (nx * dx) / 2
        y_max = (ny * dy) / 2
        
        is_valid = True
        msg = ""
        
        if idx == 1: # X
            if abs(val_um) > x_max:
                is_valid = False
                msg = f"警告: 超出仿真边界 (Warning: Out of bounds [{-x_max:.1f}, {x_max:.1f}] um)"
        elif idx == 2: # Y
            if abs(val_um) > y_max:
                is_valid = False
                msg = f"警告: 超出仿真边界 (Warning: Out of bounds [{-y_max:.1f}, {y_max:.1f}] um)"
                
        if not is_valid:
            self.sb_mon_pos.setStyleSheet("border: 1px solid red; background-color: #FFDDDD;")
            self.lbl_mon_fixed_dim.setText(f"{self.lbl_mon_fixed_dim.text()} - {msg}")
            self.lbl_mon_fixed_dim.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.sb_mon_pos.setStyleSheet("")
            # Reset label color (need to re-set text without warning)
            self.update_monitor_ui_state_text_only()

    def update_monitor_ui_state_text_only(self):
        # Helper to reset label text without full re-calc
        idx = self.combo_mon_plane.currentIndex()
        if idx == 0: self.lbl_mon_fixed_dim.setText("固定维度 (Fixed Dimension): Z-Axis (Propagation)")
        elif idx == 1: self.lbl_mon_fixed_dim.setText("固定维度 (Fixed Dimension): X-Axis")
        elif idx == 2: self.lbl_mon_fixed_dim.setText("固定维度 (Fixed Dimension): Y-Axis")
        self.lbl_mon_fixed_dim.setStyleSheet("color: gray; font-style: italic;")
        
        # Update component checkboxes visibility
        self.cb_mon_ex.setVisible(True)
        self.cb_mon_ey.setVisible(True)
        self.cb_mon_ez.setVisible(True)
        
        if idx == 0: # XY
            self.cb_mon_ez.setVisible(False)
            self.cb_mon_ez.setChecked(False)
        elif idx == 1: # YZ (Normal X) -> Ey, Ez dominant? 
            # Requirement: "YZ plane: only show Ey Ez"
            self.cb_mon_ex.setVisible(False)
            self.cb_mon_ex.setChecked(False)
        elif idx == 2: # XZ (Normal Y) -> Ex, Ez dominant?
            # Requirement: "XZ plane: only show Ex Ez"
            self.cb_mon_ey.setVisible(False)
            self.cb_mon_ey.setChecked(False)

    def update_current_monitor(self):
        row = self.monitor_list.currentRow()
        if row >= 0:
            monitor = self.monitors[row]
            new_pos = self.sb_mon_pos.value()
            new_unit = self.combo_mon_pos_unit.currentText()
            new_plane = self.combo_mon_plane.currentIndex()
            
            # Check conflict if pos or plane changed
            temp_mon = monitor.copy()
            # Normalize pos key
            temp_mon['pos'] = new_pos
            temp_mon['pos_unit'] = new_unit
            temp_mon['z'] = new_pos # Legacy key
            temp_mon['plane'] = new_plane
            
            # Conflict logic needs to know what 'z' means now.
            # Assuming conflict logic checks (pos, plane) uniqueness.
            
            if self.detect_conflict(temp_mon, exclude_index=row):
                 # Revert UI
                 self.sb_mon_pos.blockSignals(True)
                 self.sb_mon_pos.setValue(monitor.get('pos', monitor.get('z', 0)))
                 self.sb_mon_pos.blockSignals(False)
                 return
            
            # Update monitor
            monitor['pos'] = new_pos
            monitor['pos_unit'] = new_unit
            monitor['z'] = new_pos # For backward compat
            monitor['plane'] = new_plane
            monitor['type'] = self.combo_mon_type.currentIndex()
            
            monitor['range1_min'] = self.sb_range1_min.value()
            monitor['range1_max'] = self.sb_range1_max.value()
            monitor['range2_min'] = self.sb_range2_min.value()
            monitor['range2_max'] = self.sb_range2_max.value()
            
            # Components
            comps = []
            if self.cb_mon_ex.isChecked(): comps.append('Ex')
            if self.cb_mon_ey.isChecked(): comps.append('Ey')
            if self.cb_mon_ez.isChecked(): comps.append('Ez')
            monitor['output_components'] = comps
            
            self.validate_monitor_pos()
            self.validate_monitor_ranges()

    def get_project_data(self) -> dict:
        """
        Gather all parameters for project saving
        """
        data = {}
        
        # Grid
        data['grid'] = {
            'direction': 'z' if self.rb_z.isChecked() else ('y' if self.rb_y.isChecked() else 'x'),
            'nx': self.sb_nx.value(),
            'ny': self.sb_ny.value(),
            'dx': self.sb_dx.value(),
            'dy': self.sb_dy.value(),
            'wavelength': self.sb_wavelength.value()
        }
        
        # Source
        vars_list = []
        for row in range(self.table_vars.rowCount()):
            name_item = self.table_vars.item(row, 0)
            val_item = self.table_vars.item(row, 1)
            
            # Safe access
            name_text = name_item.text() if name_item else f"var{row}"
            val_text = val_item.text() if val_item else "1.0"
            
            vars_list.append({
                'name': name_text,
                'value': val_text
            })
            
        data['source'] = {
            'type_idx': self.combo_source.currentIndex(),
            'polarization_type': self.combo_pol_type.currentIndex(),
            'linear_angle': self.sb_pol_angle.value(),
            'amplitude': self.sb_amplitude.value(),
            'z_pos': self.sb_z_pos.value(),
            'normalize': self.cb_normalize.isChecked(),
            'w0': self.sb_w0.value(),
            'lg_w0': self.sb_lg_w0.value(),
            'lg_p': self.sb_lg_p.value(),
            'lg_l': self.sb_lg_l.value(),
            'bessel_w0': self.sb_bessel_w0.value(),
            'custom': {
                'coord_sys': self.combo_coord_sys.currentIndex(),
                'equation': self.txt_equation.toPlainText(),
                'variables': vars_list
            }
        }
        
        # Modulators
        for prefix in ['mod1', 'mod2']:
            pol_list = []
            if getattr(self, f"cb_{prefix}_pol_lin_x").isChecked(): pol_list.append('linear_x')
            if getattr(self, f"cb_{prefix}_pol_lcp").isChecked(): pol_list.append('lcp')
            if getattr(self, f"cb_{prefix}_pol_rcp").isChecked(): pol_list.append('rcp')
            if getattr(self, f"cb_{prefix}_pol_unpol").isChecked(): pol_list.append('unpolarized')
            
            mod_data = {
                'z': getattr(self, f"sb_{prefix}_z").value(),
                'z_unit': getattr(self, f"combo_{prefix}_z_unit").currentText(),
                'type_idx': getattr(self, f"combo_{prefix}_type").currentIndex(),
                'affected_polarizations': pol_list,
                'lens': {
                    'D': getattr(self, f"sb_{prefix}_D").value(),
                    'D_unit': getattr(self, f"combo_{prefix}_D_unit").currentText(),
                    'f': getattr(self, f"sb_{prefix}_f").value(),
                    'f_unit': getattr(self, f"combo_{prefix}_f_unit").currentText(),
                    'NA': getattr(self, f"sb_{prefix}_NA").value()
                }
            }
            # Paths
            if prefix == 'mod1':
                mod_data['phase_path'] = self.mod1_phase_path
                mod_data['amp_path'] = self.mod1_amp_path
            else:
                mod_data['phase_path'] = self.mod2_phase_path
                mod_data['angle_path'] = self.mod2_angle_path
            
            # Custom Mask Definition
            if hasattr(self, f"mask_tabs_{prefix}"):
                
                def get_geom_params(p_prefix, c_type):
                    id_p = f"{p_prefix}_{c_type}"
                    return {
                        'type': getattr(self, f"combo_shape_{id_p}").currentIndex(),
                        'ann_r_in': getattr(self, f"sb_ann_r_in_{id_p}").value(),
                        'ann_r_out': getattr(self, f"sb_ann_r_out_{id_p}").value(),
                        'ann_cx': getattr(self, f"sb_ann_cx_{id_p}").value(),
                        'ann_cy': getattr(self, f"sb_ann_cy_{id_p}").value(),
                        'ann_val': getattr(self, f"sb_ann_val_{id_p}").value(),
                        
                        'cir_r': getattr(self, f"sb_cir_r_{id_p}").value(),
                        'cir_cx': getattr(self, f"sb_cir_cx_{id_p}").value(),
                        'cir_cy': getattr(self, f"sb_cir_cy_{id_p}").value(),
                        'cir_val': getattr(self, f"sb_cir_val_{id_p}").value(),
                        
                        'rect_w': getattr(self, f"sb_rect_w_{id_p}").value(),
                        'rect_h': getattr(self, f"sb_rect_h_{id_p}").value(),
                        'rect_cx': getattr(self, f"sb_rect_cx_{id_p}").value(),
                        'rect_cy': getattr(self, f"sb_rect_cy_{id_p}").value(),
                        'rect_rot': getattr(self, f"sb_rect_rot_{id_p}").value(),
                        'rect_val': getattr(self, f"sb_rect_val_{id_p}").value(),
                        
                        'poly_val': getattr(self, f"sb_poly_val_{id_p}").value(),
                        'poly_verts': getattr(self, f"poly_editor_{id_p}").get_vertices()
                    }

                mod_data['custom_mask'] = {
                    'mode': getattr(self, f"mask_tabs_{prefix}").currentIndex(),
                    'trans_mode': getattr(self, f"combo_trans_mode_{prefix}").currentIndex(),
                    'trans_formula': getattr(self, f"fw_trans_{prefix}").get_formula(),
                    'trans_vars': getattr(self, f"fw_trans_{prefix}").custom_vars,
                    'trans_geom': get_geom_params(prefix, 'trans'),
                    
                    'phase_mode': getattr(self, f"combo_phase_mode_{prefix}").currentIndex(),
                    'phase_formula': getattr(self, f"fw_phase_{prefix}").get_formula(),
                    'phase_vars': getattr(self, f"fw_phase_{prefix}").custom_vars,
                    'phase_geom': get_geom_params(prefix, 'phase'),
                }
                
            data[prefix] = mod_data
            
        # Monitors
        data['monitors'] = self.monitors
        
        return data

    def load_project_data(self, data: dict):
        """
        Load parameters from project data
        """
        if not data: return
        
        # Block signals globally if possible, or just be careful
        # Grid
        if 'grid' in data:
            g = data['grid']
            d = g.get('direction', 'z')
            if d == 'x': self.rb_x.setChecked(True)
            elif d == 'y': self.rb_y.setChecked(True)
            else: self.rb_z.setChecked(True)
            
            self.sb_nx.setValue(g.get('nx', 512))
            self.sb_ny.setValue(g.get('ny', 512))
            self.sb_dx.setValue(g.get('dx', 1.0))
            self.sb_dy.setValue(g.get('dy', 1.0))
            self.sb_wavelength.setValue(g.get('wavelength', 0.532))
            
        # Source
        if 'source' in data:
            s = data['source']
            self.combo_source.setCurrentIndex(s.get('type_idx', 0))
            self.combo_pol_type.setCurrentIndex(s.get('polarization_type', 0))
            self.sb_pol_angle.setValue(s.get('linear_angle', 0.0))
            self.sb_amplitude.setValue(s.get('amplitude', 1.0))
            self.sb_z_pos.setValue(s.get('z_pos', 0.0))
            self.cb_normalize.setChecked(s.get('normalize', False))
            self.sb_w0.setValue(s.get('w0', 100.0))
            self.sb_lg_w0.setValue(s.get('lg_w0', 100.0))
            self.sb_lg_p.setValue(s.get('lg_p', 0))
            self.sb_lg_l.setValue(s.get('lg_l', 1))
            self.sb_bessel_w0.setValue(s.get('bessel_w0', 100.0))
            
            if 'custom' in s:
                c = s['custom']
                self.combo_coord_sys.setCurrentIndex(c.get('coord_sys', 0))
                self.txt_equation.setPlainText(c.get('equation', ''))
                
                # Table
                vars_list = c.get('variables', [])
                self.table_vars.setRowCount(0)
                for i, v in enumerate(vars_list):
                    self.table_vars.insertRow(i)
                    self.table_vars.setItem(i, 0, QTableWidgetItem(v['name']))
                    self.table_vars.setItem(i, 1, QTableWidgetItem(v['value']))
                    
        # Modulators
        for prefix in ['mod1', 'mod2']:
            if prefix in data:
                m = data[prefix]
                getattr(self, f"sb_{prefix}_z").setValue(m.get('z', 0))
                getattr(self, f"combo_{prefix}_z_unit").setCurrentText(m.get('z_unit', 'um'))
                getattr(self, f"combo_{prefix}_type").setCurrentIndex(m.get('type_idx', 0))
                
                pols = m.get('affected_polarizations', ['unpolarized'])
                getattr(self, f"cb_{prefix}_pol_lin_x").setChecked('linear_x' in pols)
                getattr(self, f"cb_{prefix}_pol_lcp").setChecked('lcp' in pols)
                getattr(self, f"cb_{prefix}_pol_rcp").setChecked('rcp' in pols)
                getattr(self, f"cb_{prefix}_pol_unpol").setChecked('unpolarized' in pols)
                
                if 'lens' in m:
                    l = m['lens']
                    getattr(self, f"sb_{prefix}_D").setValue(l.get('D', 25400))
                    getattr(self, f"combo_{prefix}_D_unit").setCurrentText(l.get('D_unit', 'um'))
                    getattr(self, f"sb_{prefix}_f").setValue(l.get('f', 100000))
                    getattr(self, f"combo_{prefix}_f_unit").setCurrentText(l.get('f_unit', 'um'))
                    getattr(self, f"sb_{prefix}_NA").setValue(l.get('NA', 0.127))
                    
                # Load files if paths exist
                # This might fail if files moved. Log warning?
                # For now, just try to set path and maybe reload if simple.
                # Actually, `load_data` takes a path.
                if prefix == 'mod1':
                    if m.get('phase_path'): self.load_data('phase1', m['phase_path'])
                    if m.get('amp_path'): self.load_data('amp1', m['amp_path'])
                else:
                    if m.get('phase_path'): self.load_data('phase2', m['phase_path'])
                    if m.get('angle_path'): self.load_data('angle2', m['angle_path'])
                
                # Load Custom Mask Definition
                if 'custom_mask' in m:
                    cm = m['custom_mask']
                    if hasattr(self, f"mask_tabs_{prefix}"):
                        
                        def load_geom(p_prefix, c_type, g_data):
                            id_p = f"{p_prefix}_{c_type}"
                            if not g_data: return
                            getattr(self, f"combo_shape_{id_p}").setCurrentIndex(g_data.get('type', 0))
                            
                            getattr(self, f"sb_ann_r_in_{id_p}").setValue(g_data.get('ann_r_in', 0.0))
                            getattr(self, f"sb_ann_r_out_{id_p}").setValue(g_data.get('ann_r_out', 100.0))
                            getattr(self, f"sb_ann_cx_{id_p}").setValue(g_data.get('ann_cx', 0.0))
                            getattr(self, f"sb_ann_cy_{id_p}").setValue(g_data.get('ann_cy', 0.0))
                            getattr(self, f"sb_ann_val_{id_p}").setValue(g_data.get('ann_val', 1.0))
                            
                            getattr(self, f"sb_cir_r_{id_p}").setValue(g_data.get('cir_r', 100.0))
                            getattr(self, f"sb_cir_cx_{id_p}").setValue(g_data.get('cir_cx', 0.0))
                            getattr(self, f"sb_cir_cy_{id_p}").setValue(g_data.get('cir_cy', 0.0))
                            getattr(self, f"sb_cir_val_{id_p}").setValue(g_data.get('cir_val', 1.0))
                            
                            getattr(self, f"sb_rect_w_{id_p}").setValue(g_data.get('rect_w', 200.0))
                            getattr(self, f"sb_rect_h_{id_p}").setValue(g_data.get('rect_h', 200.0))
                            getattr(self, f"sb_rect_cx_{id_p}").setValue(g_data.get('rect_cx', 0.0))
                            getattr(self, f"sb_rect_cy_{id_p}").setValue(g_data.get('rect_cy', 0.0))
                            getattr(self, f"sb_rect_rot_{id_p}").setValue(g_data.get('rect_rot', 0.0))
                            getattr(self, f"sb_rect_val_{id_p}").setValue(g_data.get('rect_val', 1.0))
                            
                            getattr(self, f"sb_poly_val_{id_p}").setValue(g_data.get('poly_val', 1.0))
                            getattr(self, f"poly_editor_{id_p}").set_vertices(g_data.get('poly_verts', []))
                        
                        # Legacy Check
                        if 'shape_type' in cm:
                             old_mode = cm.get('mode', 0)
                             if old_mode == 2:
                                 getattr(self, f"mask_tabs_{prefix}").setCurrentIndex(1)
                                 getattr(self, f"combo_trans_mode_{prefix}").setCurrentIndex(1)
                                 getattr(self, f"combo_phase_mode_{prefix}").setCurrentIndex(0)
                                 
                                 old_geom = {
                                     'type': cm.get('shape_type', 0),
                                     'ann_r_in': cm.get('ann_r_in', 0),
                                     'ann_r_out': cm.get('ann_r_out', 100),
                                     'ann_cx': cm.get('ann_cx', 0),
                                     'ann_cy': cm.get('ann_cy', 0),
                                     'ann_val': cm.get('ann_trans', 1.0),
                                     'cir_r': cm.get('cir_r', 100),
                                     'cir_cx': cm.get('cir_cx', 0),
                                     'cir_cy': cm.get('cir_cy', 0),
                                     'cir_val': cm.get('cir_trans', 1.0),
                                     'rect_w': cm.get('rect_w', 200),
                                     'rect_h': cm.get('rect_h', 200),
                                     'rect_cx': cm.get('rect_cx', 0),
                                     'rect_cy': cm.get('rect_cy', 0),
                                     'rect_rot': cm.get('rect_rot', 0),
                                     'rect_val': cm.get('rect_trans', 1.0),
                                 }
                                 load_geom(prefix, 'trans', old_geom)
                                 
                             elif old_mode == 1:
                                 getattr(self, f"mask_tabs_{prefix}").setCurrentIndex(1)
                                 getattr(self, f"combo_trans_mode_{prefix}").setCurrentIndex(0)
                                 getattr(self, f"combo_phase_mode_{prefix}").setCurrentIndex(0)
                                 getattr(self, f"fw_trans_{prefix}").set_formula(cm.get('trans_formula', '1.0'))
                                 getattr(self, f"fw_phase_{prefix}").set_formula(cm.get('phase_formula', '0.0'))
                                 getattr(self, f"fw_trans_{prefix}").set_variables(cm.get('trans_vars', {}))
                                 getattr(self, f"fw_phase_{prefix}").set_variables(cm.get('phase_vars', {}))
                             else:
                                 getattr(self, f"mask_tabs_{prefix}").setCurrentIndex(0)
                        else:
                            getattr(self, f"mask_tabs_{prefix}").setCurrentIndex(cm.get('mode', 0))
                            
                            getattr(self, f"combo_trans_mode_{prefix}").setCurrentIndex(cm.get('trans_mode', 0))
                            getattr(self, f"fw_trans_{prefix}").set_formula(cm.get('trans_formula', '1.0'))
                            getattr(self, f"fw_trans_{prefix}").set_variables(cm.get('trans_vars', {}))
                            if 'trans_geom' in cm: load_geom(prefix, 'trans', cm['trans_geom'])
                            
                            getattr(self, f"combo_phase_mode_{prefix}").setCurrentIndex(cm.get('phase_mode', 0))
                            getattr(self, f"fw_phase_{prefix}").set_formula(cm.get('phase_formula', '0.0'))
                            getattr(self, f"fw_phase_{prefix}").set_variables(cm.get('phase_vars', {}))
                            if 'phase_geom' in cm: load_geom(prefix, 'phase', cm['phase_geom'])
                    
        # Monitors
        if 'monitors' in data:
            self.monitors = data['monitors']
            self.monitor_list.clear()
            for m in self.monitors:
                self.monitor_list.addItem(m['name'])
            
            # Select first if exists
            if self.monitors:
                self.monitor_list.setCurrentRow(0)

    def load_data(self, target, path=None):
        """
        加载数据文件 (Load data file)
        """
        if path is None:
            filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Data Files (*.csv *.mat)")
        else:
            filename = path
            
        if not filename or not os.path.exists(filename):
            return
            
        try:
            data = None
            if filename.endswith('.csv'):
                df = pd.read_csv(filename, header=None)
                data = df.values
            elif filename.endswith('.mat'):
                mat = loadmat(filename)
                keys = [k for k in mat.keys() if not k.startswith('__')]
                if keys:
                    data = mat[keys[0]]
            
            if data is not None:
                if target == 'phase1':
                    self.mod1_phase = data
                    self.mod1_phase_path = filename
                    self.lbl_phase1_status.setText(f"Loaded: {os.path.basename(filename)}")
                elif target == 'amp1':
                    self.mod1_amp = data
                    self.mod1_amp_path = filename
                    self.lbl_amp1_status.setText(f"Loaded: {os.path.basename(filename)}")
                elif target == 'phase2':
                    self.mod2_phase = data
                    self.mod2_phase_path = filename
                    self.lbl_phase2_status.setText(f"Loaded: {os.path.basename(filename)}")
                elif target == 'angle2':
                    self.mod2_angle_trans = data
                    self.mod2_angle_path = filename
                    self.lbl_angle2_status.setText(f"Loaded: {os.path.basename(filename)}")
                    
        except Exception as e:
            if path is None:
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
            else:
                print(f"Error loading background file {filename}: {e}")

