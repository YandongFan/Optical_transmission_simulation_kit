from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QGroupBox, 
                             QRadioButton, QButtonGroup, QLabel, QDoubleSpinBox, 
                             QComboBox, QPushButton, QFormLayout, QScrollArea, QFileDialog)
import numpy as np
import pandas as pd
from scipy.io import loadmat

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
        
        # Data holders for modulators
        self.mod1_phase = None
        self.mod1_amp = None
        self.mod2_phase = None
        self.mod2_angle_trans = None

    def create_grid_tab(self):
        """
        创建网格与方向配置页 (Create Grid & Direction Tab)
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 1.1 Direction Group
        direction_group = QGroupBox("光路传输方向 (Propagation Direction)")
        d_layout = QVBoxLayout()
        self.rb_x = QRadioButton("X 轴正方向 (+X)")
        self.rb_y = QRadioButton("Y 轴正方向 (+Y)")
        self.rb_z = QRadioButton("Z 轴正方向 (+Z)")
        self.rb_z.setChecked(True) # Default
        
        btn_group = QButtonGroup(self)
        btn_group.addButton(self.rb_x)
        btn_group.addButton(self.rb_y)
        btn_group.addButton(self.rb_z)
        
        d_layout.addWidget(self.rb_x)
        d_layout.addWidget(self.rb_y)
        d_layout.addWidget(self.rb_z)
        direction_group.setLayout(d_layout)
        layout.addWidget(direction_group)
        
        # 1.2 Grid Group
        grid_group = QGroupBox("三维网格参数 (3D Grid Parameters)")
        g_layout = QFormLayout()
        
        # Use DoubleSpinBox for input
        self.sb_nx = QDoubleSpinBox()
        self.sb_nx.setRange(1, 10000)
        self.sb_nx.setValue(512)
        self.sb_nx.setDecimals(0)
        
        self.sb_ny = QDoubleSpinBox()
        self.sb_ny.setRange(1, 10000)
        self.sb_ny.setValue(512)
        self.sb_ny.setDecimals(0)
        
        self.sb_dx = QDoubleSpinBox()
        self.sb_dx.setRange(0.001, 1000) # um
        self.sb_dx.setValue(1.0)
        self.sb_dx.setSuffix(" um")
        
        self.sb_dy = QDoubleSpinBox()
        self.sb_dy.setRange(0.001, 1000) # um
        self.sb_dy.setValue(1.0)
        self.sb_dy.setSuffix(" um")
        
        self.sb_wavelength = QDoubleSpinBox()
        self.sb_wavelength.setRange(0.1, 100) # um
        self.sb_wavelength.setValue(0.532) # 532nm
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
        """
        创建光源配置页 (Create Source Tab)
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Source Type
        type_group = QGroupBox("光源类型 (Source Type)")
        t_layout = QVBoxLayout()
        self.combo_source = QComboBox()
        self.combo_source.addItems(["平面波 (Plane Wave)", "高斯光束 (Gaussian Beam)", 
                                    "拉盖尔-高斯 (Laguerre-Gaussian)", "贝塞尔光束 (Bessel Beam)"])
        t_layout.addWidget(self.combo_source)
        type_group.setLayout(t_layout)
        layout.addWidget(type_group)
        
        # Source Parameters (Dynamic based on type ideally, simpler here)
        param_group = QGroupBox("光源参数 (Source Parameters)")
        p_layout = QFormLayout()
        
        self.sb_amplitude = QDoubleSpinBox()
        self.sb_amplitude.setValue(1.0)
        
        self.sb_w0 = QDoubleSpinBox()
        self.sb_w0.setRange(0.1, 10000) # um
        self.sb_w0.setValue(100.0)
        self.sb_w0.setSuffix(" um")
        
        self.sb_z_pos = QDoubleSpinBox()
        self.sb_z_pos.setRange(-10000, 10000) # um
        self.sb_z_pos.setValue(0.0)
        self.sb_z_pos.setSuffix(" um")
        
        p_layout.addRow("振幅 (Amplitude):", self.sb_amplitude)
        p_layout.addRow("束腰半径 (Waist Radius w0):", self.sb_w0)
        p_layout.addRow("位置 (Z Position):", self.sb_z_pos)
        
        param_group.setLayout(p_layout)
        layout.addWidget(param_group)
        
        layout.addStretch()
        return tab

    def create_modulator1_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Position
        pos_group = QGroupBox("位置 (Position)")
        pos_layout = QFormLayout()
        self.sb_mod1_z = QDoubleSpinBox()
        self.sb_mod1_z.setRange(0, 10000) # mm
        self.sb_mod1_z.setValue(10.0)
        self.sb_mod1_z.setSuffix(" mm")
        pos_layout.addRow("距离光源 (Distance from Source):", self.sb_mod1_z)
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Phase Modulation
        phase_group = QGroupBox("相位调制 (Phase Modulation)")
        phase_layout = QVBoxLayout()
        self.btn_load_phase1 = QPushButton("加载相位分布 (Load Phase Mask) - CSV/MAT")
        self.btn_load_phase1.clicked.connect(lambda: self.load_data('phase1'))
        self.lbl_phase1_status = QLabel("未加载 (Not Loaded)")
        phase_layout.addWidget(self.btn_load_phase1)
        phase_layout.addWidget(self.lbl_phase1_status)
        phase_group.setLayout(phase_layout)
        layout.addWidget(phase_group)
        
        # Transmission Modulation
        trans_group = QGroupBox("透射率调制 (Transmission Modulation)")
        trans_layout = QVBoxLayout()
        self.btn_load_amp1 = QPushButton("加载透射率分布 (Load Transmission Mask) - CSV/MAT")
        self.btn_load_amp1.clicked.connect(lambda: self.load_data('amp1'))
        self.lbl_amp1_status = QLabel("未加载 (Not Loaded)")
        trans_layout.addWidget(self.btn_load_amp1)
        trans_layout.addWidget(self.lbl_amp1_status)
        trans_group.setLayout(trans_layout)
        layout.addWidget(trans_group)
        
        layout.addStretch()
        return tab

    def create_modulator2_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Position
        pos_group = QGroupBox("位置 (Position)")
        pos_layout = QFormLayout()
        self.sb_mod2_z = QDoubleSpinBox()
        self.sb_mod2_z.setRange(0, 10000) # mm
        self.sb_mod2_z.setValue(20.0)
        self.sb_mod2_z.setSuffix(" mm")
        pos_layout.addRow("距离光源 (Distance from Source):", self.sb_mod2_z)
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Phase Modulation
        phase_group = QGroupBox("相位调制 (Phase Modulation)")
        phase_layout = QVBoxLayout()
        self.btn_load_phase2 = QPushButton("加载相位分布 (Load Phase Mask) - CSV/MAT")
        self.btn_load_phase2.clicked.connect(lambda: self.load_data('phase2'))
        self.lbl_phase2_status = QLabel("未加载 (Not Loaded)")
        phase_layout.addWidget(self.btn_load_phase2)
        phase_layout.addWidget(self.lbl_phase2_status)
        phase_group.setLayout(phase_layout)
        layout.addWidget(phase_group)
        
        # Angle Transmission
        angle_group = QGroupBox("角度-透射率 (Angle-Transmission)")
        angle_layout = QVBoxLayout()
        self.btn_load_angle2 = QPushButton("加载角度透射率曲线 (Load Angle-T Curve) - CSV")
        self.btn_load_angle2.clicked.connect(lambda: self.load_data('angle2'))
        self.lbl_angle2_status = QLabel("未加载 (Not Loaded)")
        angle_layout.addWidget(self.btn_load_angle2)
        angle_layout.addWidget(self.lbl_angle2_status)
        angle_group.setLayout(angle_layout)
        layout.addWidget(angle_group)
        
        layout.addStretch()
        return tab

    def create_monitor_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("监视器配置 (Monitor Configuration)"))
        # Simple implementation: One monitor at end
        self.sb_monitor_z = QDoubleSpinBox()
        self.sb_monitor_z.setRange(0, 10000)
        self.sb_monitor_z.setValue(30.0)
        self.sb_monitor_z.setSuffix(" mm")
        
        layout.addWidget(QLabel("监视器位置 (Monitor Position):"))
        layout.addWidget(self.sb_monitor_z)
        
        layout.addStretch()
        return tab

    def load_data(self, target):
        """
        加载数据文件 (Load data file)
        """
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Data Files (*.csv *.mat)")
        if not filename:
            return
            
        try:
            data = None
            if filename.endswith('.csv'):
                df = pd.read_csv(filename, header=None)
                data = df.values
            elif filename.endswith('.mat'):
                mat = loadmat(filename)
                # Assume variable name 'data' or take the first variable
                keys = [k for k in mat.keys() if not k.startswith('__')]
                if keys:
                    data = mat[keys[0]]
            
            if data is not None:
                if target == 'phase1':
                    self.mod1_phase = data
                    self.lbl_phase1_status.setText(f"Loaded: {filename}")
                elif target == 'amp1':
                    self.mod1_amp = data
                    self.lbl_amp1_status.setText(f"Loaded: {filename}")
                elif target == 'phase2':
                    self.mod2_phase = data
                    self.lbl_phase2_status.setText(f"Loaded: {filename}")
                elif target == 'angle2':
                    self.mod2_angle_trans = data
                    self.lbl_angle2_status.setText(f"Loaded: {filename}")
                    
        except Exception as e:
            print(f"Error loading file: {e}")
