import sys
import numpy as np
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QStatusBar, QMenuBar, QMenu, QMessageBox)
from PyQt6.QtGui import QAction
from .parameter_panel import ParameterPanel
from .visualization_panel import VisualizationPanel

# Import core modules
from ..core.field import Grid, OpticalField
from ..core.source import PlaneWave, GaussianBeam, LaguerreGaussianBeam, Source
from ..core.propagator import AngularSpectrumPropagator
from ..core.modulator import SpatialModulator, AngleModulator
from ..core.monitor import Monitor

import json

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
        self.status_bar.showMessage("Ready")
        
        # Menu Bar
        self.create_menu_bar()
        
        # Connect Signals
        self.parameter_panel.btn_preview.clicked.connect(self.on_preview)
        self.parameter_panel.btn_run.clicked.connect(self.on_run)

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
        QMessageBox.about(self, "About", "Optical Transmission Simulation Kit\nVersion 1.0")

    def save_project(self):
        """
        保存工程 (Save Project)
        """
        filename, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "JSON Files (*.json)")
        if not filename:
            return
            
        try:
            # Gather parameters
            nx, ny, dx, dy, wavelength = self.get_grid_params()
            source_type, amp, w0, z_pos = self.get_source_params()
            
            data = {
                "grid": {
                    "nx": nx, "ny": ny, "dx": dx, "dy": dy, "wavelength": wavelength
                },
                "source": {
                    "type": source_type, "amplitude": amp, "w0": w0, "z_pos": z_pos
                },
                "modulators": {
                    "mod1_z": self.parameter_panel.sb_mod1_z.value(),
                    "mod2_z": self.parameter_panel.sb_mod2_z.value(),
                    "monitor_z": self.parameter_panel.sb_monitor_z.value()
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
                
            self.status_bar.showMessage(f"Project saved to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")

    def load_project(self):
        """
        加载工程 (Load Project)
        """
        filename, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "JSON Files (*.json)")
        if not filename:
            return
            
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
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
            
            # Set Modulator Params
            mods = data.get("modulators", {})
            self.parameter_panel.sb_mod1_z.setValue(mods.get("mod1_z", 10))
            self.parameter_panel.sb_mod2_z.setValue(mods.get("mod2_z", 20))
            self.parameter_panel.sb_monitor_z.setValue(mods.get("monitor_z", 30))
            
            self.status_bar.showMessage(f"Project loaded from {filename}")
            self.on_preview() # Update preview
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")


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
        return source_type_idx, amplitude, w0, z_pos

    def on_preview(self):
        """
        预览光源分布 (Preview Source Distribution)
        """
        try:
            self.status_bar.showMessage("Generating preview...")
            
            # 1. Create Grid
            nx, ny, dx, dy, wavelength = self.get_grid_params()
            self.grid = Grid(nx, ny, dx, dy, wavelength)
            
            # 2. Create Source
            source_type_idx, amplitude, w0, z_pos = self.get_source_params()
            
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
                
                # 3. Update Plots
                intensity = self.field.get_intensity().cpu().numpy()
                phase = self.field.get_phase().cpu().numpy()
                
                # Convert grid to um for display
                x_um = self.grid.X * 1e6
                y_um = self.grid.Y * 1e6
                
                self.visualization_panel.update_plots(self.field.to_numpy(), intensity, phase, x_um, y_um)
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
                propagator = AngularSpectrumPropagator(self.grid)
                
                # Retrieve parameters
                mod1_z = self.parameter_panel.sb_mod1_z.value() * 1e-3 # mm to m
                mod2_z = self.parameter_panel.sb_mod2_z.value() * 1e-3 # mm to m
                monitor_z = self.parameter_panel.sb_monitor_z.value() * 1e-3 # mm to m
                
                # Ensure order
                z_positions = sorted([(mod1_z, 'mod1'), (mod2_z, 'mod2'), (monitor_z, 'monitor')])
                
                current_z = 0.0
                current_field = self.field # Copy? field is object, but propagator creates new field
                
                for z, type_ in z_positions:
                    dist = z - current_z
                    if dist > 0:
                        current_field = propagator.propagate(current_field, dist)
                        current_z = z
                    
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
                        # Record
                        monitor = Monitor(z, "Monitor_End")
                        monitor.record(current_field)
                        
                        # Visualize
                        intensity = current_field.get_intensity().cpu().numpy()
                        phase = current_field.get_phase().cpu().numpy()
                        x_um = self.grid.X * 1e6
                        y_um = self.grid.Y * 1e6
                        
                        self.visualization_panel.update_plots(current_field.to_numpy(), intensity, phase, x_um, y_um)
                        
                self.status_bar.showMessage(f"Simulation complete.")
                
            except Exception as e:
                self.status_bar.showMessage(f"Error: {str(e)}")
                print(e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
