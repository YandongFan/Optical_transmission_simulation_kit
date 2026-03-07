import sys
import numpy as np
import torch
import json
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QStatusBar, QMenuBar, QMenu, QMessageBox, 
                             QFileDialog, QProgressBar)
from PyQt6.QtGui import QAction
from .parameter_panel import ParameterPanel
from .visualization_panel import VisualizationPanel

# Import core modules
from ..core.field import Grid, OpticalField
from ..core.source import PlaneWave, GaussianBeam, LaguerreGaussianBeam, Source, CustomSource
from ..core.propagator import AngularSpectrumPropagator
from ..core.modulator import SpatialModulator, AngleModulator, IdealLens, CylindricalLens
from ..core.monitor import Monitor

class MainWindow(QMainWindow):
    """
    主窗口 (Main Window)
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("光学仿真套件 (Optical Transmission Simulation Kit)")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QHBoxLayout(central_widget)
        
        # Left Panel: Parameters
        self.parameter_panel = ParameterPanel()
        layout.addWidget(self.parameter_panel, 1) # Stretch factor 1
        
        # Right Panel: Visualization
        self.visualization_panel = VisualizationPanel()
        layout.addWidget(self.visualization_panel, 2) # Stretch factor 2
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Ready")
        
        # Menu Bar
        self.create_menu_bar()
        
        # Connect Signals
        self.parameter_panel.btn_preview.clicked.connect(self.on_preview)
        self.parameter_panel.btn_run.clicked.connect(self.on_run)
        self.parameter_panel.btn_export_data.clicked.connect(self.export_monitor_data)

        # Simulation State
        self.grid = None
        self.field = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.status_bar.showMessage(f"Ready (Device: {self.device})")
        
    def create_menu_bar(self):
        menu_bar = self.menuBar()
        
        # File Menu
        file_menu = menu_bar.addMenu("文件 (File)")
        
        save_action = QAction("保存工程 (Save Project)", self)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        load_action = QAction("加载工程 (Load Project)", self)
        load_action.triggered.connect(self.load_project)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出 (Exit)", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help Menu
        help_menu = menu_bar.addMenu("帮助 (Help)")
        
        about_action = QAction("关于 (About)", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def show_about(self):
        QMessageBox.about(self, "About", "Optical Transmission Simulation Kit\nVersion 1.2")

    def save_project(self):
        """
        保存工程 (Save Project)
        """
        filename, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "JSON Files (*.json)")
        if not filename:
            return
            
        try:
            self.status_bar.showMessage("Saving project...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            
            # Gather parameters
            nx, ny, dx, dy, wavelength = self.get_grid_params()
            source_params = self.get_source_params()
            
            data = {
                "version": "1.2",
                "grid": {
                    "nx": nx, "ny": ny, "dx": dx, "dy": dy, "wavelength": wavelength
                },
                "source": source_params,
                "modulators": {
                    "mod1": self.get_modulator_config('mod1'),
                    "mod2": self.get_modulator_config('mod2')
                },
                "monitors": self.parameter_panel.monitors
            }
            
            self.progress_bar.setValue(50)
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
                
            self.progress_bar.setValue(100)
            self.status_bar.showMessage(f"Project saved to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    def load_project(self):
        """
        加载工程 (Load Project)
        """
        filename, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "JSON Files (*.json)")
        if not filename:
            return
            
        try:
            self.status_bar.showMessage("Loading project...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # Version check (basic)
            version = data.get("version", "1.0")
            
            self.progress_bar.setValue(30)
            
            # Set Grid Params
            grid = data.get("grid", {})
            self.parameter_panel.sb_nx.setValue(grid.get("nx", 512))
            self.parameter_panel.sb_ny.setValue(grid.get("ny", 512))
            self.parameter_panel.sb_dx.setValue(grid.get("dx", 1e-6) * 1e6)
            self.parameter_panel.sb_dy.setValue(grid.get("dy", 1e-6) * 1e6)
            self.parameter_panel.sb_wavelength.setValue(grid.get("wavelength", 0.532e-6) * 1e6)
            
            # Set Source Params
            source = data.get("source", {})
            self.parameter_panel.combo_source.setCurrentIndex(source.get("type_idx", 0))
            self.parameter_panel.sb_amplitude.setValue(source.get("amplitude", 1.0))
            self.parameter_panel.sb_z_pos.setValue(source.get("z_pos", 0) * 1e6)
            self.parameter_panel.cb_normalize.setChecked(source.get("normalize", False))
            
            if "w0" in source:
                self.parameter_panel.sb_w0.setValue(source.get("w0", 100e-6) * 1e6)
                self.parameter_panel.sb_lg_w0.setValue(source.get("w0", 100e-6) * 1e6)
                self.parameter_panel.sb_bessel_w0.setValue(source.get("w0", 100e-6) * 1e6)
            
            if "p" in source: self.parameter_panel.sb_lg_p.setValue(source.get("p", 0))
            if "l" in source: self.parameter_panel.sb_lg_l.setValue(source.get("l", 1))
            
            if "equation" in source: self.parameter_panel.txt_equation.setPlainText(source.get("equation", ""))
            
            # Variables
            vars_dict = source.get("variables", {})
            self.parameter_panel.table_vars.setRowCount(0)
            for k, v in vars_dict.items():
                self.parameter_panel.add_custom_variable()
                row = self.parameter_panel.table_vars.rowCount() - 1
                self.parameter_panel.table_vars.setItem(row, 0,  QTableWidgetItem(str(k)))
                self.parameter_panel.table_vars.setItem(row, 1,  QTableWidgetItem(str(v)))

            self.progress_bar.setValue(50)
            
            # Set Modulator Params
            # Note: This is simplified. Restoring full state including paths needs more work.
            # Assuming paths are not stored in detail in get_modulator_config for now or handled separately.
            # The current save logic stores basic config.
            
            mods = data.get("modulators", {})
            for prefix in ['mod1', 'mod2']:
                m = mods.get(prefix, {})
                if m:
                    # Restore Z
                    sb_z = getattr(self.parameter_panel, f"sb_{prefix}_z")
                    combo_z = getattr(self.parameter_panel, f"combo_{prefix}_z_unit")
                    
                    if 'z_unit' in m:
                        combo_z.setCurrentText(m['z_unit'])
                        sb_z.setValue(m['z_val'])
                    else:
                        # Fallback for old files (meters -> um)
                        combo_z.setCurrentText("um")
                        sb_z.setValue(m.get('z', 0) * 1e6)
                        
                    getattr(self.parameter_panel, f"combo_{prefix}_type").setCurrentIndex(m.get('type_idx', 0))
                    
                    # Restore F and D
                    if 'f' in m:
                        sb_f = getattr(self.parameter_panel, f"sb_{prefix}_f")
                        combo_f = getattr(self.parameter_panel, f"combo_{prefix}_f_unit")
                        
                        if 'f_unit' in m:
                            combo_f.setCurrentText(m['f_unit'])
                            sb_f.setValue(m['f_val'])
                        else:
                            combo_f.setCurrentText("um")
                            sb_f.setValue(m.get('f', 0) * 1e6)
                            
                    if 'D_val' in m:
                        sb_D = getattr(self.parameter_panel, f"sb_{prefix}_D")
                        combo_D = getattr(self.parameter_panel, f"combo_{prefix}_D_unit")
                        combo_D.setCurrentText(m.get('D_unit', 'um'))
                        sb_D.setValue(m['D_val'])

            # Set Monitors
            monitors = data.get("monitors", [])
            self.parameter_panel.monitors = monitors
            self.parameter_panel.monitor_list.clear()
            for m in monitors:
                self.parameter_panel.monitor_list.addItem(m['name'])
            
            self.progress_bar.setValue(80)
            
            self.status_bar.showMessage(f"Project loaded from {filename}")
            self.on_preview() # Update preview
            self.progress_bar.setValue(100)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    def get_grid_params(self):
        """
        从界面获取网格参数 (Get grid parameters from GUI)
        """
        nx = int(self.parameter_panel.sb_nx.value())
        ny = int(self.parameter_panel.sb_ny.value())
        dx = self.parameter_panel.sb_dx.value() * 1e-6 # um to m
        dy = self.parameter_panel.sb_dy.value() * 1e-6 # um to m
        wavelength = self.parameter_panel.sb_wavelength.value() * 1e-6 # um to m
        return nx, ny, dx, dy, wavelength

    def get_source_params(self):
        """
        从界面获取光源参数 (Get source parameters from GUI)
        """
        idx = self.parameter_panel.combo_source.currentIndex()
        params = {
            'type_idx': idx,
            'amplitude': self.parameter_panel.sb_amplitude.value(),
            'z_pos': self.parameter_panel.sb_z_pos.value() * 1e-6,
            'normalize': self.parameter_panel.cb_normalize.isChecked()
        }
        
        if idx == 1: # Gaussian
            params['w0'] = self.parameter_panel.sb_w0.value() * 1e-6
        elif idx == 2: # LG
            params['w0'] = self.parameter_panel.sb_lg_w0.value() * 1e-6
            params['p'] = self.parameter_panel.sb_lg_p.value()
            params['l'] = self.parameter_panel.sb_lg_l.value()
        elif idx == 3: # Bessel
            params['w0'] = self.parameter_panel.sb_bessel_w0.value() * 1e-6
        elif idx == 4: # Custom
            params['equation'] = self.parameter_panel.txt_equation.toPlainText()
            # Parse variables
            vars_dict = {}
            table = self.parameter_panel.table_vars
            for i in range(table.rowCount()):
                name_item = table.item(i, 0)
                val_item = table.item(i, 1)
                if name_item and val_item:
                    try:
                        vars_dict[name_item.text()] = float(val_item.text())
                    except ValueError:
                        pass
            params['variables'] = vars_dict
            
        return params

    def get_value_in_meters(self, sb, combo):
        val = sb.value()
        unit = combo.currentText()
        if unit == "mm": return val * 1e-3
        elif unit == "um": return val * 1e-6
        return val

    def get_modulator_config(self, prefix):
        combo = getattr(self.parameter_panel, f"combo_{prefix}_type")
        type_idx = combo.currentIndex()
        
        sb_z = getattr(self.parameter_panel, f"sb_{prefix}_z")
        combo_z = getattr(self.parameter_panel, f"combo_{prefix}_z_unit")
        z = self.get_value_in_meters(sb_z, combo_z)
        
        config = {'type_idx': type_idx, 'z': z, 'z_val': sb_z.value(), 'z_unit': combo_z.currentText()}
        
        if type_idx != 0: # Lens
            sb_f = getattr(self.parameter_panel, f"sb_{prefix}_f")
            combo_f = getattr(self.parameter_panel, f"combo_{prefix}_f_unit")
            config['f'] = self.get_value_in_meters(sb_f, combo_f)
            config['f_val'] = sb_f.value()
            config['f_unit'] = combo_f.currentText()
            
            sb_D = getattr(self.parameter_panel, f"sb_{prefix}_D")
            combo_D = getattr(self.parameter_panel, f"combo_{prefix}_D_unit")
            config['D_val'] = sb_D.value()
            config['D_unit'] = combo_D.currentText()
        
        # Paths for custom mask are stored in parameter_panel state, not retrieved here easily 
        # unless we add them to config. For save/load they are needed.
        if type_idx == 0:
            config['phase_path'] = getattr(self.parameter_panel, f"{prefix}_phase_path")
            if prefix == 'mod1':
                config['amp_path'] = getattr(self.parameter_panel, f"{prefix}_amp_path")
            elif prefix == 'mod2':
                config['angle_path'] = getattr(self.parameter_panel, f"{prefix}_angle_path")
                
        return config

    def on_preview(self):
        """
        预览光源分布 (Preview Source Distribution)
        """
        try:
            self.status_bar.showMessage("Generating preview...")
            self.visualization_panel.clear_data()
            
            # 1. Create Grid
            nx, ny, dx, dy, wavelength = self.get_grid_params()
            self.grid = Grid(nx, ny, dx, dy, wavelength)
            
            # 2. Create Source
            params = self.get_source_params()
            idx = params['type_idx']
            amp = params['amplitude']
            z_pos = params['z_pos']
            
            source = None
            if idx == 0: # Plane Wave
                source = PlaneWave(self.grid, amplitude=amp)
            elif idx == 1: # Gaussian Beam
                source = GaussianBeam(self.grid, amplitude=amp, w0=params['w0'], z=z_pos)
            elif idx == 2: # Laguerre-Gaussian
                source = LaguerreGaussianBeam(self.grid, amplitude=amp, w0=params['w0'], 
                                              p=params['p'], l=params['l'])
            elif idx == 3: # Bessel Beam
                 source = GaussianBeam(self.grid, amplitude=amp, w0=params['w0'], z=z_pos)
            elif idx == 4: # Custom
                 source = CustomSource(self.grid, amplitude=amp, equation=params['equation'], 
                                       variables=params['variables'])
            
            if source:
                self.field = source.generate(device=self.device)
                
                # Normalize if requested
                if params['normalize']:
                    self.field.normalize()
                
                # 3. Update Plots
                intensity = self.field.get_intensity().cpu().numpy()
                phase = self.field.get_phase().cpu().numpy()
                x_um = self.grid.X * 1e6
                y_um = self.grid.Y * 1e6
                
                self.visualization_panel.add_monitor_result("Source Preview", self.field.to_numpy(), intensity, phase, x_um, y_um)
                self.status_bar.showMessage("Preview updated.")
                
        except Exception as e:
            self.status_bar.showMessage(f"Error: {str(e)}")
            print(e)

    def on_run(self):
        """
        运行完整仿真 (Run Full Simulation)
        """
        if self.field is None:
            self.on_preview()
            
        if self.field:
            try:
                self.status_bar.showMessage("Running simulation...")
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                self.visualization_panel.clear_data() # Clear previous results
                
                # Check normalization setting again (in case changed but not previewed)
                params = self.get_source_params()
                if params['normalize']:
                     self.field.normalize()

                propagator = AngularSpectrumPropagator(self.grid)
                
                # Retrieve parameters
                mod1_config = self.get_modulator_config('mod1')
                mod2_config = self.get_modulator_config('mod2')
                
                # Monitors from list
                monitors_config = self.parameter_panel.monitors
                
                # Build event list
                events = []
                events.append({'z': mod1_config['z'], 'type': 'mod1', 'config': mod1_config})
                events.append({'z': mod2_config['z'], 'type': 'mod2', 'config': mod2_config})
                
                for m in monitors_config:
                    events.append({'z': m['z'] * 1e-3, 'type': 'monitor', 'config': m})
                
                # Sort by z
                events.sort(key=lambda x: x['z'])
                
                current_z = 0.0
                current_field = self.field 
                
                total_steps = len(events)
                
                for i, event in enumerate(events):
                    z = event['z']
                    dist = z - current_z
                    
                    if dist > 0:
                        current_field = propagator.propagate(current_field, dist)
                        current_z = z
                    
                    type_ = event['type']
                    
                    if type_ == 'mod1' or type_ == 'mod2':
                        config = event['config']
                        type_idx = config['type_idx']
                        
                        if type_idx == 0: # Custom Mask
                            prefix = type_
                            # Apply Modulator 1 or 2 custom
                            if prefix == 'mod1':
                                mod = SpatialModulator(self.grid, 
                                                        amplitude_mask=self.parameter_panel.mod1_amp,
                                                        phase_mask=self.parameter_panel.mod1_phase)
                                current_field = mod.modulate(current_field)
                            elif prefix == 'mod2':
                                mod_spatial = SpatialModulator(self.grid, phase_mask=self.parameter_panel.mod2_phase)
                                current_field = mod_spatial.modulate(current_field)
                                mod_angle = AngleModulator(self.grid, angle_transmission_curve=None) 
                                current_field = mod_angle.modulate(current_field)
                                
                        elif type_idx == 1: # Ideal Lens
                            lens = IdealLens(self.grid, focal_length=config['f'])
                            current_field = lens.modulate(current_field)
                        elif type_idx == 2: # Cyl X
                            lens = CylindricalLens(self.grid, focal_length=config['f'], axis='x')
                            current_field = lens.modulate(current_field)
                        elif type_idx == 3: # Cyl Y
                            lens = CylindricalLens(self.grid, focal_length=config['f'], axis='y')
                            current_field = lens.modulate(current_field)
                        
                    elif type_ == 'monitor':
                        config = event['config']
                        monitor_name = config['name']
                        
                        # Visualize
                        intensity = current_field.get_intensity().cpu().numpy()
                        phase = current_field.get_phase().cpu().numpy()
                        x_um = self.grid.X * 1e6
                        y_um = self.grid.Y * 1e6
                        
                        self.visualization_panel.add_monitor_result(monitor_name, current_field.to_numpy(), intensity, phase, x_um, y_um)
                        
                    self.progress_bar.setValue(int((i + 1) / total_steps * 100))
                    QApplication.processEvents() # Keep UI responsive
                        
                self.status_bar.showMessage(f"Simulation complete.")
                
            except Exception as e:
                self.status_bar.showMessage(f"Error: {str(e)}")
                print(e)
            finally:
                self.progress_bar.setVisible(False)

    def export_monitor_data(self):
        row = self.parameter_panel.monitor_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Warning", "Please select a monitor to export.")
            return
        name = self.parameter_panel.monitors[row]['name']
        self.visualization_panel.export_data(name)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
