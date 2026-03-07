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
from ..core.source import PlaneWave, GaussianBeam, LaguerreGaussianBeam, Source
from ..core.propagator import AngularSpectrumPropagator
from ..core.modulator import SpatialModulator, AngleModulator
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
        QMessageBox.about(self, "About", "Optical Transmission Simulation Kit\nVersion 1.1")

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
            source_type, amp, w0, z_pos = self.get_source_params()
            
            data = {
                "version": "1.1",
                "grid": {
                    "nx": nx, "ny": ny, "dx": dx, "dy": dy, "wavelength": wavelength
                },
                "source": {
                    "type": source_type, "amplitude": amp, "w0": w0, "z_pos": z_pos
                },
                "modulators": {
                    "mod1": {
                        "z": self.parameter_panel.sb_mod1_z.value(),
                        "phase_path": self.parameter_panel.mod1_phase_path,
                        "amp_path": self.parameter_panel.mod1_amp_path
                    },
                    "mod2": {
                        "z": self.parameter_panel.sb_mod2_z.value(),
                        "phase_path": self.parameter_panel.mod2_phase_path,
                        "angle_path": self.parameter_panel.mod2_angle_path
                    }
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
            if version not in ["1.0", "1.1"]:
                 QMessageBox.warning(self, "Warning", f"Unknown project version: {version}. Loading may fail.")
            
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
            self.parameter_panel.combo_source.setCurrentIndex(source.get("type", 0))
            self.parameter_panel.sb_amplitude.setValue(source.get("amplitude", 1.0))
            self.parameter_panel.sb_w0.setValue(source.get("w0", 100e-6) * 1e6)
            self.parameter_panel.sb_z_pos.setValue(source.get("z_pos", 0) * 1e6)
            
            self.progress_bar.setValue(50)
            
            # Set Modulator Params
            mods = data.get("modulators", {})
            
            # Compatibility with v1.0 structure
            if "mod1_z" in mods:
                 # v1.0
                 self.parameter_panel.sb_mod1_z.setValue(mods.get("mod1_z", 10))
                 self.parameter_panel.sb_mod2_z.setValue(mods.get("mod2_z", 20))
                 # Monitor z was single in v1.0, ignore here as we load monitors list below if v1.1
            else:
                 # v1.1
                 mod1 = mods.get("mod1", {})
                 self.parameter_panel.sb_mod1_z.setValue(mod1.get("z", 10))
                 if mod1.get("phase_path") and os.path.exists(mod1.get("phase_path")):
                     self.parameter_panel.load_data('phase1', mod1.get("phase_path")) # Modify load_data to accept path
                 if mod1.get("amp_path") and os.path.exists(mod1.get("amp_path")):
                     self.parameter_panel.load_data('amp1', mod1.get("amp_path"))
                     
                 mod2 = mods.get("mod2", {})
                 self.parameter_panel.sb_mod2_z.setValue(mod2.get("z", 20))
                 if mod2.get("phase_path") and os.path.exists(mod2.get("phase_path")):
                     self.parameter_panel.load_data('phase2', mod2.get("phase_path"))
                 if mod2.get("angle_path") and os.path.exists(mod2.get("angle_path")):
                     self.parameter_panel.load_data('angle2', mod2.get("angle_path"))

            # Set Monitors (v1.1)
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
        source_type_idx = self.parameter_panel.combo_source.currentIndex()
        amplitude = self.parameter_panel.sb_amplitude.value()
        w0 = self.parameter_panel.sb_w0.value() * 1e-6 # um to m
        z_pos = self.parameter_panel.sb_z_pos.value() * 1e-6 # um to m
        normalize = self.parameter_panel.cb_normalize.isChecked()
        return source_type_idx, amplitude, w0, z_pos, normalize

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
            source_type_idx, amplitude, w0, z_pos, normalize = self.get_source_params()
            
            source = None
            if source_type_idx == 0: # Plane Wave
                source = PlaneWave(self.grid, amplitude=amplitude)
            elif source_type_idx == 1: # Gaussian Beam
                source = GaussianBeam(self.grid, amplitude=amplitude, w0=w0, z=z_pos)
            elif source_type_idx == 2: # Laguerre-Gaussian
                source = LaguerreGaussianBeam(self.grid, amplitude=amplitude, w0=w0, p=0, l=1) # TODO: Add p, l inputs
            elif source_type_idx == 3: # Bessel Beam
                 # TODO: Implement Bessel Beam or fallback
                 source = GaussianBeam(self.grid, amplitude=amplitude, w0=w0, z=z_pos)
            
            if source:
                self.field = source.generate(device=self.device)
                
                # Normalize if requested
                if normalize:
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
                _, _, _, _, normalize = self.get_source_params()
                if normalize:
                     self.field.normalize()

                propagator = AngularSpectrumPropagator(self.grid)
                
                # Retrieve parameters
                mod1_z = self.parameter_panel.sb_mod1_z.value() * 1e-3 # mm to m
                mod2_z = self.parameter_panel.sb_mod2_z.value() * 1e-3 # mm to m
                
                # Monitors from list
                monitors_config = self.parameter_panel.monitors
                
                # Build event list
                events = []
                events.append({'z': mod1_z, 'type': 'mod1'})
                events.append({'z': mod2_z, 'type': 'mod2'})
                
                for m in monitors_config:
                    events.append({'z': m['z'] * 1e-3, 'type': 'monitor', 'config': m})
                
                # Sort by z
                events.sort(key=lambda x: x['z'])
                
                current_z = 0.0
                current_field = self.field # This is reference, but propagator creates new. Be careful if inplace. Propagator is not inplace.
                
                total_steps = len(events)
                
                for i, event in enumerate(events):
                    z = event['z']
                    dist = z - current_z
                    
                    if dist > 0:
                        current_field = propagator.propagate(current_field, dist)
                        current_z = z
                    
                    type_ = event['type']
                    
                    if type_ == 'mod1':
                        # Apply Modulator 1
                        mod1 = SpatialModulator(self.grid, 
                                                amplitude_mask=self.parameter_panel.mod1_amp,
                                                phase_mask=self.parameter_panel.mod1_phase)
                        current_field = mod1.modulate(current_field)
                        
                    elif type_ == 'mod2':
                        # Apply Modulator 2
                        # Phase
                        mod2_spatial = SpatialModulator(self.grid, phase_mask=self.parameter_panel.mod2_phase)
                        current_field = mod2_spatial.modulate(current_field)
                        
                        # Angle
                        # TODO: Convert mod2_angle_trans data to function
                        mod2_angle = AngleModulator(self.grid, angle_transmission_curve=None) 
                        current_field = mod2_angle.modulate(current_field)
                        
                    elif type_ == 'monitor':
                        config = event['config']
                        monitor_name = config['name']
                        
                        # Record
                        # monitor = Monitor(z, monitor_name)
                        # monitor.record(current_field)
                        
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
