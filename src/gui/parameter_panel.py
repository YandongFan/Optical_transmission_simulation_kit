from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QGroupBox, 
                             QRadioButton, QButtonGroup, QLabel, QDoubleSpinBox, 
                             QComboBox, QPushButton, QFormLayout, QScrollArea, QFileDialog,
                             QListWidget, QHBoxLayout, QMessageBox, QCheckBox, QDialog,
                             QSpinBox, QLineEdit, QTableWidget, QTableWidgetItem, QTextEdit,
                             QStackedWidget, QHeaderView)
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
        
        self.sb_start = QDoubleSpinBox()
        self.sb_start.setRange(0, 10000)
        self.sb_start.setValue(10.0)
        self.sb_start.setSuffix(" mm")
        
        self.sb_spacing = QDoubleSpinBox()
        self.sb_spacing.setRange(0.001, 1000)
        self.sb_spacing.setValue(1.0)
        self.sb_spacing.setSuffix(" mm")
        
        self.le_prefix = QLineEdit("Array_")
        
        layout.addRow("监视器数量 (Count):", self.sb_count)
        layout.addRow("起始位置 (Start Z):", self.sb_start)
        layout.addRow("间距 (Spacing):", self.sb_spacing)
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
        return self.sb_count.value(), self.sb_start.value(), self.sb_spacing.value(), self.le_prefix.text()

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
        
    def create_grid_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        direction_group = QGroupBox("光路传输方向 (Propagation Direction)")
        d_layout = QVBoxLayout()
        self.rb_x = QRadioButton("X 轴正方向 (+X)")
        self.rb_y = QRadioButton("Y 轴正方向 (+Y)")
        self.rb_z = QRadioButton("Z 轴正方向 (+Z)")
        self.rb_z.setChecked(True)
        
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
        self.sb_wavelength.setSuffix(" um")
        
        g_layout.addRow("Nx (Grid Points X):", self.sb_nx)
        g_layout.addRow("Ny (Grid Points Y):", self.sb_ny)
        g_layout.addRow("dx (Spacing X):", self.sb_dx)
        g_layout.addRow("dy (Spacing Y):", self.sb_dy)
        g_layout.addRow("Wavelength:", self.sb_wavelength)
        
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
        t_layout.addWidget(self.combo_source)
        type_group.setLayout(t_layout)
        layout.addWidget(type_group)
        
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
        coord_layout.addWidget(self.combo_coord_sys)
        coord_group.setLayout(coord_layout)
        pc_layout.addWidget(coord_group)
        
        # Variables Table
        pc_layout.addWidget(QLabel("变量定义 (Variables):"))
        self.table_vars = QTableWidget(0, 2)
        self.table_vars.setHorizontalHeaderLabels(["Name", "Value"])
        self.table_vars.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
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
        pc_layout.addWidget(self.txt_equation)
        
        self.source_stack.addWidget(self.page_custom)
        
        layout.addWidget(self.source_stack)
        
        # Initialize
        self.update_source_ui(0)
        
        layout.addStretch()
        return tab

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
        row = self.table_vars.rowCount()
        self.table_vars.insertRow(row)
        self.table_vars.setItem(row, 0, QTableWidgetItem(f"var{row}"))
        self.table_vars.setItem(row, 1, QTableWidgetItem("1.0"))

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

    def create_modulator_tab_generic(self, title, prefix):
        """
        Generic creator for modulator tabs
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Position
        pos_group = QGroupBox("位置 (Position)")
        pos_layout = QFormLayout()
        
        # Default 10mm = 10000um
        default_z = 10000.0 if prefix == 'mod1' else 20000.0
        w_z, sb_z, combo_z = self.create_unit_spinbox(default_z, "um")
        
        pos_layout.addRow("距离光源 (Distance):", w_z)
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Store widget reference
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
        
        # Phase
        phase_group = QGroupBox("相位调制 (Phase)")
        ph_layout = QVBoxLayout()
        btn_phase = QPushButton("加载相位分布 (Load Phase)")
        lbl_phase = QLabel("未加载 (Not Loaded)")
        btn_phase.clicked.connect(lambda: self.load_data(f'phase{prefix[-1]}'))
        ph_layout.addWidget(btn_phase)
        ph_layout.addWidget(lbl_phase)
        phase_group.setLayout(ph_layout)
        pm_layout.addWidget(phase_group)
        
        # Store refs
        setattr(self, f"btn_load_phase{prefix[-1]}", btn_phase)
        setattr(self, f"lbl_phase{prefix[-1]}_status", lbl_phase)
        
        # Amp
        amp_group = QGroupBox("透射率 (Transmission)")
        am_layout = QVBoxLayout()
        btn_amp = QPushButton("加载透射率 (Load Trans)")
        lbl_amp = QLabel("未加载 (Not Loaded)")
        btn_amp.clicked.connect(lambda: self.load_data(f'amp{prefix[-1]}'))
        am_layout.addWidget(btn_amp)
        am_layout.addWidget(lbl_amp)
        amp_group.setLayout(am_layout)
        pm_layout.addWidget(amp_group)
        
        setattr(self, f"btn_load_amp{prefix[-1]}", btn_amp)
        setattr(self, f"lbl_amp{prefix[-1]}_status", lbl_amp)
        
        stack.addWidget(page_mask)
        
        # 1: Lens Params (Shared for all lens types, logic handles visibility)
        page_lens = QWidget()
        pl_layout = QFormLayout(page_lens)
        
        # Diameter 25.4 mm = 25400 um
        w_D, sb_D, combo_D = self.create_unit_spinbox(25400.0, "um")
        
        # Focal Length 100 mm = 100000 um
        w_f, sb_f, combo_f = self.create_unit_spinbox(100000.0, "um")
        
        sb_NA = QDoubleSpinBox()
        sb_NA.setRange(0.001, 0.999)
        sb_NA.setValue(0.127) # 25.4 / (2*100) = 0.127
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
        # Also connect unit changes? create_unit_spinbox handles value update, which triggers valueChanged.
        # So update_na will be called.
        
        sb_NA.valueChanged.connect(update_f)
        
        pl_layout.addRow("直径 D (Diameter):", w_D)
        pl_layout.addRow("焦距 f (Focal Length):", w_f)
        pl_layout.addRow("数值孔径 NA:", sb_NA)
        
        stack.addWidget(page_lens) # 1: Ideal
        stack.addWidget(QWidget()) # 2: Cyl X (Reuse or new? Reuse logic, new widget or same?)
        
        setattr(self, f"sb_{prefix}_D", sb_D)
        setattr(self, f"combo_{prefix}_D_unit", combo_D)
        setattr(self, f"sb_{prefix}_f", sb_f)
        setattr(self, f"combo_{prefix}_f_unit", combo_f)
        setattr(self, f"sb_{prefix}_NA", sb_NA)
        
        combo.currentIndexChanged.connect(lambda idx: stack.setCurrentIndex(0 if idx == 0 else 1))
        
        layout.addWidget(stack)
        setattr(self, f"stack_{prefix}", stack)
        
        # Additional Angle Trans for Mod2
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
        self.monitor_list.currentRowChanged.connect(self.load_monitor_settings)
        
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
        
        self.sb_mon_z = QDoubleSpinBox()
        self.sb_mon_z.setRange(0, 10000)
        self.sb_mon_z.setSuffix(" mm")
        self.sb_mon_z.valueChanged.connect(self.update_current_monitor)
        
        self.combo_mon_plane = QComboBox()
        self.combo_mon_plane.addItems(["XY Plane", "YZ Plane", "ZX Plane"])
        self.combo_mon_plane.currentIndexChanged.connect(self.update_current_monitor)
        
        self.combo_mon_type = QComboBox()
        self.combo_mon_type.addItems(["Intensity (|E|^2)", "Complex Field (E)"])
        self.combo_mon_type.currentIndexChanged.connect(self.update_current_monitor)
        
        settings_layout.addRow("位置 (Position):", self.sb_mon_z)
        settings_layout.addRow("切面 (Plane):", self.combo_mon_plane)
        settings_layout.addRow("数据类型 (Data Type):", self.combo_mon_type)
        
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
            'z': 0.0,
            'plane': 0,
            'type': 0
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
        for i, m in enumerate(self.monitors):
            if i == exclude_index:
                continue
            if abs(m['z'] - monitor['z']) < 1e-6 and m['plane'] == monitor['plane']:
                return True
        return False

    def delete_monitor(self):
        row = self.monitor_list.currentRow()
        if row >= 0:
            self.monitor_list.takeItem(row)
            del self.monitors[row]
            if not self.monitors:
                self.settings_group.setEnabled(False)

    def add_monitor_array(self):
        dialog = MonitorArrayDialog(self)
        if dialog.exec():
            count, start, spacing, prefix = dialog.get_values()
            for i in range(count):
                z = start + i * spacing
                name = f"{prefix}{len(self.monitors) + 1}"
                monitor = {'name': name, 'z': z, 'plane': 0, 'type': 0}
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
            self.sb_mon_z.blockSignals(True)
            self.combo_mon_plane.blockSignals(True)
            self.combo_mon_type.blockSignals(True)
            
            self.sb_mon_z.setValue(monitor['z'])
            self.combo_mon_plane.setCurrentIndex(monitor['plane'])
            self.combo_mon_type.setCurrentIndex(monitor['type'])
            
            self.sb_mon_z.blockSignals(False)
            self.combo_mon_plane.blockSignals(False)
            self.combo_mon_type.blockSignals(False)
        else:
            self.settings_group.setEnabled(False)

    def update_current_monitor(self):
        row = self.monitor_list.currentRow()
        if row >= 0:
            monitor = self.monitors[row]
            new_z = self.sb_mon_z.value()
            new_plane = self.combo_mon_plane.currentIndex()
            
            # Check conflict if z or plane changed
            temp_mon = monitor.copy()
            temp_mon['z'] = new_z
            temp_mon['plane'] = new_plane
            
            if self.detect_conflict(temp_mon, exclude_index=row):
                 # Revert UI
                 self.sb_mon_z.blockSignals(True)
                 self.sb_mon_z.setValue(monitor['z'])
                 self.sb_mon_z.blockSignals(False)
                 return
                 
            monitor['z'] = new_z
            monitor['plane'] = new_plane
            monitor['type'] = self.combo_mon_type.currentIndex()

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
