from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QGroupBox, 
                             QRadioButton, QButtonGroup, QLabel, QDoubleSpinBox, 
                             QComboBox, QPushButton, QFormLayout, QScrollArea, QFileDialog,
                             QListWidget, QHBoxLayout, QMessageBox, QCheckBox, QDialog,
                             QSpinBox, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os

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
        
        type_group = QGroupBox("光源类型 (Source Type)")
        t_layout = QVBoxLayout()
        self.combo_source = QComboBox()
        self.combo_source.addItems(["平面波 (Plane Wave)", "高斯光束 (Gaussian Beam)", 
                                    "拉盖尔-高斯 (Laguerre-Gaussian)", "贝塞尔光束 (Bessel Beam)"])
        t_layout.addWidget(self.combo_source)
        type_group.setLayout(t_layout)
        layout.addWidget(type_group)
        
        param_group = QGroupBox("光源参数 (Source Parameters)")
        p_layout = QFormLayout()
        
        self.sb_amplitude = QDoubleSpinBox()
        self.sb_amplitude.setValue(1.0)
        
        self.sb_w0 = QDoubleSpinBox()
        self.sb_w0.setRange(0.1, 10000)
        self.sb_w0.setValue(100.0)
        self.sb_w0.setSuffix(" um")
        
        self.sb_z_pos = QDoubleSpinBox()
        self.sb_z_pos.setRange(-10000, 10000)
        self.sb_z_pos.setValue(0.0)
        self.sb_z_pos.setSuffix(" um")
        
        p_layout.addRow("振幅 (Amplitude):", self.sb_amplitude)
        p_layout.addRow("束腰半径 (Waist Radius w0):", self.sb_w0)
        p_layout.addRow("位置 (Z Position):", self.sb_z_pos)
        
        self.cb_normalize = QCheckBox("光源电场归一化 (Normalize E-field)")
        self.cb_normalize.setToolTip("勾选后，将电场最大值归一化为1。仅影响数值比例，不影响物理功率。")
        p_layout.addRow(self.cb_normalize)
        
        param_group.setLayout(p_layout)
        layout.addWidget(param_group)
        
        layout.addStretch()
        return tab

    def create_modulator1_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        pos_group = QGroupBox("位置 (Position)")
        pos_layout = QFormLayout()
        self.sb_mod1_z = QDoubleSpinBox()
        self.sb_mod1_z.setRange(0, 10000)
        self.sb_mod1_z.setValue(10.0)
        self.sb_mod1_z.setSuffix(" mm")
        pos_layout.addRow("距离光源 (Distance from Source):", self.sb_mod1_z)
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        phase_group = QGroupBox("相位调制 (Phase Modulation)")
        phase_layout = QVBoxLayout()
        self.btn_load_phase1 = QPushButton("加载相位分布 (Load Phase Mask)")
        self.btn_load_phase1.clicked.connect(lambda: self.load_data('phase1'))
        self.lbl_phase1_status = QLabel("未加载 (Not Loaded)")
        phase_layout.addWidget(self.btn_load_phase1)
        phase_layout.addWidget(self.lbl_phase1_status)
        phase_group.setLayout(phase_layout)
        layout.addWidget(phase_group)
        
        trans_group = QGroupBox("透射率调制 (Transmission Modulation)")
        trans_layout = QVBoxLayout()
        self.btn_load_amp1 = QPushButton("加载透射率分布 (Load Transmission Mask)")
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
        
        pos_group = QGroupBox("位置 (Position)")
        pos_layout = QFormLayout()
        self.sb_mod2_z = QDoubleSpinBox()
        self.sb_mod2_z.setRange(0, 10000)
        self.sb_mod2_z.setValue(20.0)
        self.sb_mod2_z.setSuffix(" mm")
        pos_layout.addRow("距离光源 (Distance from Source):", self.sb_mod2_z)
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        phase_group = QGroupBox("相位调制 (Phase Modulation)")
        phase_layout = QVBoxLayout()
        self.btn_load_phase2 = QPushButton("加载相位分布 (Load Phase Mask)")
        self.btn_load_phase2.clicked.connect(lambda: self.load_data('phase2'))
        self.lbl_phase2_status = QLabel("未加载 (Not Loaded)")
        phase_layout.addWidget(self.btn_load_phase2)
        phase_layout.addWidget(self.lbl_phase2_status)
        phase_group.setLayout(phase_layout)
        layout.addWidget(phase_group)
        
        angle_group = QGroupBox("角度-透射率 (Angle-Transmission)")
        angle_layout = QVBoxLayout()
        self.btn_load_angle2 = QPushButton("加载角度透射率曲线 (Load Angle-T Curve)")
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
