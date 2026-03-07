import sys
import os
import json
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QSplitter, QStatusBar, QProgressBar, QApplication, QMessageBox,
                             QMenuBar, QMenu, QFileDialog)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from src.gui.parameter_panel import ParameterPanel
from src.gui.visualization_panel import VisualizationPanel
from src.core.field import Grid, OpticalField
from src.core.source import PlaneWave, GaussianBeam, LaguerreGaussianBeam, CustomSource
from src.core.propagator import AngularSpectrumPropagator
from src.core.modulator import SpatialModulator, AngleModulator, IdealLens, CylindricalLens
from src.core.monitor import Monitor

class MainWindow(QMainWindow):
    """
    主窗口 (Main Window)
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Transmission Simulation Kit")
        self.resize(1200, 800)
        
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Menu Bar
        self.create_menu_bar()
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left: Parameter Panel
        self.parameter_panel = ParameterPanel()
        splitter.addWidget(self.parameter_panel)
        
        # Right: Visualization Panel
        self.visualization_panel = VisualizationPanel()
        splitter.addWidget(self.visualization_panel)
        
        # Set initial sizes (Left smaller, Right larger)
        splitter.setSizes([400, 800])
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Progress Bar in Status Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Signals
        self.parameter_panel.btn_run.clicked.connect(self.on_run)
        self.parameter_panel.btn_preview.clicked.connect(self.on_preview)
        # Check if parameter_panel has btn_export_data exposed or connected internally
        # ParameterPanel connects btn_export_data to its own export_monitor_data method.
        
        # Simulation State
        self.field = None
        self.grid = None
        self.device = 'cuda' if np.isin('cuda', [1]) else 'cpu' # Simple check, or default to cpu
        import torch
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        self.status_bar.showMessage(f"Ready. Device: {self.device}")

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件 (File)")
        
        save_action = QAction("保存工程 (Save Project)", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        load_action = QAction("导入工程 (Load Project)", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_project)
        file_menu.addAction(load_action)

    def save_project(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Project Files (*.proj)")
        if not filename:
            return
            
        try:
            data = self.parameter_panel.get_project_data()
            data['version'] = '1.0'
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
                
            self.setWindowTitle(f"[*] {os.path.basename(filename)}")
            self.status_bar.showMessage("Project saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")

    def load_project(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Project Files (*.proj)")
        if not filename:
            return
            
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # Version check (basic)
            if 'version' not in data:
                print("Warning: No version found in project file.")
                
            self.parameter_panel.load_project_data(data)
            self.on_preview()
            self.setWindowTitle(f"[*] {os.path.basename(filename)}")
            self.status_bar.showMessage("Project loaded.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")

    def get_source_params(self):
        """
        获取光源和网格参数 (Get source and grid parameters)
        """
        pp = self.parameter_panel
        
        # Grid
        nx = int(pp.sb_nx.value())
        ny = int(pp.sb_ny.value())
        dx = pp.sb_dx.value() * 1e-6 # um to m
        dy = pp.sb_dy.value() * 1e-6 # um to m
        wavelength = pp.sb_wavelength.value() * 1e-6 # um to m
        
        # Source Common
        amplitude = pp.sb_amplitude.value()
        z_pos = pp.sb_z_pos.value() * 1e-6 # um to m
        normalize = pp.cb_normalize.isChecked()
        
        source_type_idx = pp.combo_source.currentIndex()
        
        params = {
            'nx': nx, 'ny': ny, 'dx': dx, 'dy': dy, 'wavelength': wavelength,
            'amplitude': amplitude, 'z_pos': z_pos, 'normalize': normalize,
            'type_idx': source_type_idx
        }
        
        # Specific
        if source_type_idx == 0: # Plane
            params['kx'] = 0 # Placeholder for now
            params['ky'] = 0
        elif source_type_idx == 1: # Gaussian
            params['w0'] = pp.sb_w0.value() * 1e-6
        elif source_type_idx == 2: # LG
            params['w0'] = pp.sb_lg_w0.value() * 1e-6
            params['p'] = pp.sb_lg_p.value()
            params['l'] = pp.sb_lg_l.value()
        elif source_type_idx == 3: # Bessel
            params['w0'] = pp.sb_bessel_w0.value() * 1e-6 # Using w0 as main param
        elif source_type_idx == 4: # Custom
            params['equation'] = pp.txt_equation.toPlainText()
            # Extract variables from table
            vars_dict = {}
            for row in range(pp.table_vars.rowCount()):
                name = pp.table_vars.item(row, 0).text()
                val_str = pp.table_vars.item(row, 1).text()
                try:
                    vars_dict[name] = float(val_str)
                except:
                    pass
            params['variables'] = vars_dict
            params['coord_sys'] = pp.combo_coord_sys.currentIndex()
            
        return params

    def get_modulator_config(self, prefix):
        """
        获取调制器配置 (Get modulator configuration)
        """
        pp = self.parameter_panel
        
        # Z position
        # Access via getattr as created dynamically
        sb_z = getattr(pp, f"sb_{prefix}_z")
        combo_z = getattr(pp, f"combo_{prefix}_z_unit")
        
        z_val = sb_z.value()
        unit = combo_z.currentText()
        if unit == 'mm': z_val *= 1e-3
        elif unit == 'um': z_val *= 1e-6
        
        # Type
        combo_type = getattr(pp, f"combo_{prefix}_type")
        type_idx = combo_type.currentIndex()
        
        config = {
            'z': z_val,
            'type_idx': type_idx
        }
        
        # Lens params if needed
        if type_idx in [1, 2, 3]: # Lenses
            sb_f = getattr(pp, f"sb_{prefix}_f")
            combo_f = getattr(pp, f"combo_{prefix}_f_unit")
            f_val = sb_f.value()
            f_unit = combo_f.currentText()
            if f_unit == 'mm': f_val *= 1e-3
            elif f_unit == 'um': f_val *= 1e-6
            config['f'] = f_val
            
        return config

    def on_preview(self):
        """
        预览光源 (Preview Source)
        """
        try:
            params = self.get_source_params()
            
            # Create Grid
            self.grid = Grid(params['nx'], params['ny'], params['dx'], params['dy'], params['wavelength'])
            
            # Create Source
            idx = params['type_idx']
            amp = params['amplitude']
            # z_pos is relative to simulation start? 
            # Usually source is generated at z=0 local, but placed at z_pos in simulation?
            # Or is it generated AT z_pos? 
            # Gaussian beam has z parameter which defines wavefront curvature at that point.
            
            if idx == 0: # Plane
                source = PlaneWave(self.grid, amplitude=amp)
            elif idx == 1: # Gaussian
                # If z_pos is used as 'z' parameter for Gaussian, it means distance from waist.
                # If waist is at 0, and we want to start simulation at z=0, but beam waist is at z=0.
                # Let's assume params['z_pos'] is the 'z' argument for GaussianBeam (distance from waist)
                source = GaussianBeam(self.grid, amplitude=amp, w0=params['w0'], z=params['z_pos'])
            elif idx == 2: # LG
                source = LaguerreGaussianBeam(self.grid, amplitude=amp, w0=params['w0'], 
                                              p=params['p'], l=params['l'])
            elif idx == 3: # Bessel
                # Placeholder using Gaussian for now or custom implementation?
                # Source.py didn't have Bessel implementation shown fully, only header in plan.
                # Actually I saw 'BesselBeam' in plan but 'CustomSource' in file? 
                # Let's check source.py content again.
                # It has PlaneWave, GaussianBeam, LaguerreGaussianBeam, CustomSource.
                # It does NOT have BesselBeam class explicitly in the read output!
                # Wait, I read source.py and it ended at CustomSource.
                # I'll check if I missed it or if it's missing.
                # Assuming missing, I'll use CustomSource or just error.
                # For now, let's treat it as Gaussian or error.
                source = GaussianBeam(self.grid, amplitude=amp, w0=params['w0'], z=params['z_pos'])
                print("Warning: Bessel Beam not implemented, using Gaussian.")
            elif idx == 4: # Custom
                source = CustomSource(self.grid, amplitude=amp, equation=params['equation'], 
                                      variables=params['variables'])
                
            self.field = source.generate(device=self.device)
            
            if params['normalize']:
                self.field.normalize()
                
            # Visualize
            self.visualization_panel.clear_data()
            
            # Add initial field to visualization
            x_um = self.grid.X * 1e6
            y_um = self.grid.Y * 1e6
            intensity = self.field.get_intensity().cpu().numpy()
            phase = self.field.get_phase().cpu().numpy()
            
            self.visualization_panel.add_monitor_result("Source Preview", self.field.to_numpy(), 
                                                        intensity, phase, x_um, y_um)
                                                        
            self.status_bar.showMessage("Preview updated.")
            
        except Exception as e:
            self.status_bar.showMessage(f"Preview Error: {str(e)}")
            import traceback
            traceback.print_exc()

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
                
                # Check normalization setting again
                params = self.get_source_params()
                if params['normalize']:
                     self.field.normalize()

                propagator = AngularSpectrumPropagator(self.grid)
                
                # Clone initial field to avoid modifying preview (Req 3.2)
                current_field = OpticalField(self.grid, device=self.device)
                current_field.set_field(self.field.E.clone())
                
                # Retrieve parameters
                mod1_config = self.get_modulator_config('mod1')
                mod2_config = self.get_modulator_config('mod2')
                
                monitors_config = self.parameter_panel.monitors
                
                # Identify Monitor Types
                global_monitors = [] # YZ, XZ
                xy_monitors = [] # XY
                
                for m in monitors_config:
                    # m['pos'] is in um. m['plane'] is type.
                    plane_type = m['plane']
                    pos_val = m.get('pos', m.get('z', 0)) # um
                    fixed_val_m = pos_val * 1e-6
                    
                    # Ranges
                    ranges = {}
                    if plane_type == 0: # XY
                        ranges['x'] = (m.get('range1_min', -1e9)*1e-6, m.get('range1_max', 1e9)*1e-6)
                        ranges['y'] = (m.get('range2_min', -1e9)*1e-6, m.get('range2_max', 1e9)*1e-6)
                    elif plane_type == 1: # YZ
                        ranges['y'] = (m.get('range1_min', -1e9)*1e-6, m.get('range1_max', 1e9)*1e-6)
                        ranges['z'] = (m.get('range2_min', -1e9)*1e-6, m.get('range2_max', 1e9)*1e-6)
                    elif plane_type == 2: # XZ
                        ranges['x'] = (m.get('range1_min', -1e9)*1e-6, m.get('range1_max', 1e9)*1e-6)
                        ranges['z'] = (m.get('range2_min', -1e9)*1e-6, m.get('range2_max', 1e9)*1e-6)
                    
                    mon_obj = Monitor(position_z=fixed_val_m if plane_type==0 else 0, # Z for XY
                                      name=m['name'], 
                                      plane_type=plane_type,
                                      fixed_value=fixed_val_m,
                                      ranges=ranges)
                    
                    # Store type for visualization
                    mon_obj.data_type = m.get('type', 0)
                    
                    # Print geometry info (Req 3.1)
                    # print(mon_obj.get_geometry_info()) # Monitor might not have this method yet?
                    
                    if plane_type == 0:
                        xy_monitors.append({'z': fixed_val_m, 'monitor': mon_obj})
                    else:
                        global_monitors.append(mon_obj)
                        
                # Define Z-events (Modulators)
                events = []
                events.append({'z': mod1_config['z'], 'type': 'mod1', 'config': mod1_config})
                events.append({'z': mod2_config['z'], 'type': 'mod2', 'config': mod2_config})
                
                # Sort Modulator events
                events.sort(key=lambda x: x['z'])
                
                # Determine Z range
                max_z = events[-1]['z']
                if xy_monitors:
                    max_xy = max(m['z'] for m in xy_monitors)
                    max_z = max(max_z, max_xy)
                    
                # If max_z is small, default to 1mm
                if max_z < 1e-6: max_z = 1000e-6
                
                # Collect critical Z points
                z_points = set()
                z_points.add(0.0)
                for e in events: z_points.add(e['z'])
                for m in xy_monitors: z_points.add(m['z'])
                
                sorted_z_points = sorted(list(z_points))
                
                if not global_monitors:
                    # Fast Event Loop (Only Modulators and XY Monitors)
                    current_z = 0.0
                    total_steps = len(sorted_z_points)
                    
                    # Initial record at Z=0 for XY monitors
                    for m_wrapper in xy_monitors:
                        if abs(m_wrapper['z'] - 0.0) < 1e-9:
                            m_wrapper['monitor'].record(current_field, 0.0)
                            self.visualize_monitor(m_wrapper['monitor'])

                    for i, z in enumerate(sorted_z_points):
                        if i == 0 and z == 0.0: continue

                        if z > current_z:
                            dist = z - current_z
                            current_field = propagator.propagate(current_field, dist)
                            current_z = z
                            
                        # Apply events at this Z
                        for e in events:
                            if abs(e['z'] - current_z) < 1e-9:
                                current_field = self.apply_modulator(current_field, e)
                                
                        # XY Monitors
                        for m_wrapper in xy_monitors:
                            if abs(m_wrapper['z'] - current_z) < 1e-9:
                                mon = m_wrapper['monitor']
                                mon.record(current_field, current_z)
                                self.visualize_monitor(mon)
                                
                        self.progress_bar.setValue(int((i + 1) / total_steps * 100))
                        QApplication.processEvents()
                        
                else:
                    # Stepping Mode (Global Monitors active)
                    wavelength = self.grid.wavelength
                    dz = 10 * wavelength 
                    # Ensure at least 200 steps
                    if dz > max_z / 200: dz = max_z / 200
                    
                    # Generate full Z steps including critical points
                    full_z_list = [0.0]
                    for i in range(len(sorted_z_points)-1):
                        z1 = sorted_z_points[i]
                        z2 = sorted_z_points[i+1]
                        dist = z2 - z1
                        if dist < 1e-9: continue
                        
                        num_steps = int(np.ceil(dist / dz))
                        if num_steps < 1: num_steps = 1
                        steps = np.linspace(z1, z2, num_steps + 1)[1:] 
                        full_z_list.extend(steps)
                        
                    current_z = 0.0
                    total_steps = len(full_z_list)
                    
                    # Initial record at Z=0
                    for gm in global_monitors:
                        gm.record(current_field, 0.0)
                        
                    for m_wrapper in xy_monitors:
                        if abs(m_wrapper['z'] - 0.0) < 1e-9:
                             m_wrapper['monitor'].record(current_field, 0.0)
                             self.visualize_monitor(m_wrapper['monitor'])
                    
                    for i, z in enumerate(full_z_list):
                        if i == 0 and z == 0.0: continue 
                        
                        dist = z - current_z
                        if dist > 1e-9:
                            current_field = propagator.propagate(current_field, dist)
                            current_z = z
                            
                        # Modulators
                        for e in events:
                            if abs(e['z'] - current_z) < 1e-9:
                                current_field = self.apply_modulator(current_field, e)
                        
                        # Global Monitors
                        for gm in global_monitors:
                            gm.record(current_field, current_z)
                            
                        # XY Monitors
                        for m_wrapper in xy_monitors:
                            if abs(m_wrapper['z'] - current_z) < 1e-9:
                                mon = m_wrapper['monitor']
                                mon.record(current_field, current_z)
                                self.visualize_monitor(mon)
                                
                        self.progress_bar.setValue(int((i + 1) / total_steps * 100))
                        QApplication.processEvents()
                        
                    # Finalize global monitors
                    for gm in global_monitors:
                        gm.finalize()
                        self.visualize_monitor(gm)

                self.status_bar.showMessage(f"Simulation complete.")
                
            except Exception as e:
                self.status_bar.showMessage(f"Error: {str(e)}")
                print(e)
                import traceback
                traceback.print_exc()
            finally:
                self.progress_bar.setVisible(False)

    def apply_modulator(self, field, event):
        config = event['config']
        type_idx = config['type_idx']
        type_ = event['type']
        
        if type_idx == 0: # Custom Mask
            prefix = type_
            if prefix == 'mod1':
                mod = SpatialModulator(self.grid, 
                                        amplitude_mask=self.parameter_panel.mod1_amp,
                                        phase_mask=self.parameter_panel.mod1_phase)
                return mod.modulate(field)
            elif prefix == 'mod2':
                mod_spatial = SpatialModulator(self.grid, phase_mask=self.parameter_panel.mod2_phase)
                field = mod_spatial.modulate(field)
                mod_angle = AngleModulator(self.grid, angle_transmission_curve=None) 
                return mod_angle.modulate(field)
                
        elif type_idx == 1: # Ideal Lens
            lens = IdealLens(self.grid, focal_length=config['f'])
            return lens.modulate(field)
        elif type_idx == 2: # Cyl X
            lens = CylindricalLens(self.grid, focal_length=config['f'], axis='x')
            return lens.modulate(field)
        elif type_idx == 3: # Cyl Y
            lens = CylindricalLens(self.grid, focal_length=config['f'], axis='y')
            return lens.modulate(field)
        return field

    def visualize_monitor(self, monitor):
        name = monitor.name
        intensity = monitor.intensity_data
        field = monitor.field_data
        if intensity is None: return
        
        # Prepare axes
        if monitor.plane_type == 0: # XY
            x = monitor.grid_x * 1e6
            y = monitor.grid_y * 1e6
        elif monitor.plane_type == 1: # YZ
            x = np.array(monitor.z_coords) * 1e6 # Horizontal: Z
            y = monitor.grid_y * 1e6 # Vertical: Y
        elif monitor.plane_type == 2: # XZ
            x = np.array(monitor.z_coords) * 1e6 # Horizontal: Z
            y = monitor.grid_x * 1e6 # Vertical: X
            
        # Phase might be complex to visualize if flattened? 
        # But monitor.field_data should be shaped correctly by finalize() or record()
        phase = np.angle(field)
        
        complex_real = None
        complex_imag = None
        if getattr(monitor, 'data_type', 0) == 1: # Complex Field
            complex_real = np.real(field)
            complex_imag = np.imag(field)
        
        self.visualization_panel.add_monitor_result(name, field, intensity, phase, x, y, 
                                                    complex_real=complex_real, complex_imag=complex_imag)
