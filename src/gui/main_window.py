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
from src.utils.mask_generator import generate_annular_mask, generate_circular_mask, generate_rectangular_mask, generate_polygon_mask

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
        self.source_preview_data = None
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
            data['version'] = '1.4'
            
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
            version = data.get('version', '1.0')
            if version < '1.4':
                print(f"Warning: Project file version is {version}, upgrading to 1.4...")
                QMessageBox.information(self, "版本兼容性提示 (Compatibility)", 
                                      f"检测到旧版本工程文件 (v{version})。\n\n"
                                      "已自动为您补全缺失字段 (如圆环起始/结束角度)，并升级至 v1.4 标准。\n"
                                      "请在确认无误后重新保存。")
                
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
        
        # Polarization
        pol_type = pp.combo_pol_type.currentIndex()
        lin_angle = pp.sb_pol_angle.value()
        
        params = {
            'nx': nx, 'ny': ny, 'dx': dx, 'dy': dy, 'wavelength': wavelength,
            'amplitude': amplitude, 'z_pos': z_pos, 'normalize': normalize,
            'type_idx': source_type_idx,
            'polarization_type': pol_type,
            'linear_angle': lin_angle,
            'phase_offset': 0.0 # Not exposed in UI yet, default 0
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
        
        # Affected Polarizations
        pol_list = []
        if getattr(pp, f"cb_{prefix}_pol_lin_x").isChecked(): pol_list.append('linear_x')
        if getattr(pp, f"cb_{prefix}_pol_lcp").isChecked(): pol_list.append('lcp')
        if getattr(pp, f"cb_{prefix}_pol_rcp").isChecked(): pol_list.append('rcp')
        if getattr(pp, f"cb_{prefix}_pol_unpol").isChecked(): pol_list.append('unpolarized')
        
        config = {
            'z': z_val,
            'type_idx': type_idx,
            'affected_polarizations': pol_list
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
                source = PlaneWave(self.grid, amplitude=amp, 
                                   polarization_type=params['polarization_type'],
                                   linear_angle=params['linear_angle'])
            elif idx == 1: # Gaussian
                source = GaussianBeam(self.grid, amplitude=amp, w0=params['w0'], z=params['z_pos'],
                                      polarization_type=params['polarization_type'],
                                      linear_angle=params['linear_angle'])
            elif idx == 2: # LG
                source = LaguerreGaussianBeam(self.grid, amplitude=amp, w0=params['w0'], 
                                              p=params['p'], l=params['l'],
                                              polarization_type=params['polarization_type'],
                                              linear_angle=params['linear_angle'])
            elif idx == 3: # Bessel
                source = GaussianBeam(self.grid, amplitude=amp, w0=params['w0'], z=params['z_pos'],
                                      polarization_type=params['polarization_type'],
                                      linear_angle=params['linear_angle'])
                print("Warning: Bessel Beam not implemented, using Gaussian.")
            elif idx == 4: # Custom
                source = CustomSource(self.grid, amplitude=amp, equation=params['equation'], 
                                      variables=params['variables'],
                                      polarization_type=params['polarization_type'],
                                      linear_angle=params['linear_angle'])
                
            self.field = source.generate(device=self.device)
            
            if params['normalize']:
                self.field.normalize()
                
            # Visualize
            self.visualization_panel.clear_data()
            
            # Add initial field to visualization
            x_um = self.grid.X * 1e6
            y_um = self.grid.Y * 1e6
            intensity = self.field.get_intensity().cpu().numpy()
            phase = self.field.get_phase().cpu().numpy() # Phase of Ex or Ey? Default Ex
            
            # Prepare components for visualization
            components = {
                'Ex': self.field.Ex.cpu().numpy(),
                'Ey': self.field.Ey.cpu().numpy()
            }
            
            self.visualization_panel.add_monitor_result("Source Preview", self.field.to_numpy(), 
                                                        intensity, phase, x_um, y_um, components=components)
            
            # Save for later injection
            self.source_preview_data = {
                'field': self.field.to_numpy(),
                'intensity': intensity,
                'phase': phase,
                'x': x_um,
                'y': y_um,
                'components': components
            }
                                                        
            self.status_bar.showMessage("Preview updated.")
            
        except Exception as e:
            self.status_bar.showMessage(f"Preview Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_run(self):
        """
        运行完整仿真 (Run Full Simulation)
        """
        # Force read from config object (Req 2.c)
        # Instead of relying on self.field (preview cache), we regenerate source from config.
        config = self.parameter_panel.get_latest_config()
        
        try:
            self.status_bar.showMessage("Running simulation...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.visualization_panel.clear_data() # Clear previous results
            
            # DEBUG INFO
            mod1_dbg = config.get('mod1', {})
            mon_dbg = config.get('monitors', [])
            dbg_msg = f"Starting Simulation...\nMod1 Type: {mod1_dbg.get('type_idx')}\n"
            dbg_msg += f"Monitors: {len(mon_dbg)}\n"
            if mon_dbg:
                dbg_msg += f"Mon1 Components: {mon_dbg[0].get('output_components')}"
            print(dbg_msg)
            # End DEBUG
            
            # 1. Re-create Grid
            grid_cfg = config.get('grid', {})
            # get_project_data stores raw values (um), Grid needs meters
            nx = int(grid_cfg.get('nx', 512))
            ny = int(grid_cfg.get('ny', 512))
            dx = float(grid_cfg.get('dx', 1.0)) * 1e-6
            dy = float(grid_cfg.get('dy', 1.0)) * 1e-6
            wavelength = float(grid_cfg.get('wavelength', 0.532)) * 1e-6
            
            self.grid = Grid(nx, ny, dx, dy, wavelength)
            
            # 2. Re-create Source
            src_cfg = config.get('source', {})
            idx = int(src_cfg.get('type_idx', 0))
            amp = float(src_cfg.get('amplitude', 1.0))
            z_pos = float(src_cfg.get('z_pos', 0.0)) * 1e-6
            pol_type = int(src_cfg.get('polarization_type', 0))
            lin_angle = float(src_cfg.get('linear_angle', 0.0))
            
            source = None
            if idx == 0: # Plane
                source = PlaneWave(self.grid, amplitude=amp, 
                                   polarization_type=pol_type,
                                   linear_angle=lin_angle)
            elif idx == 1: # Gaussian
                w0 = float(src_cfg.get('w0', 100.0)) * 1e-6
                source = GaussianBeam(self.grid, amplitude=amp, w0=w0, z=z_pos,
                                      polarization_type=pol_type,
                                      linear_angle=lin_angle)
            elif idx == 2: # LG
                w0 = float(src_cfg.get('lg_w0', 100.0)) * 1e-6
                p = int(src_cfg.get('lg_p', 0))
                l = int(src_cfg.get('lg_l', 1))
                source = LaguerreGaussianBeam(self.grid, amplitude=amp, w0=w0, 
                                              p=p, l=l,
                                              polarization_type=pol_type,
                                              linear_angle=lin_angle)
            elif idx == 3: # Bessel
                w0 = float(src_cfg.get('bessel_w0', 100.0)) * 1e-6
                source = GaussianBeam(self.grid, amplitude=amp, w0=w0, z=z_pos,
                                      polarization_type=pol_type,
                                      linear_angle=lin_angle)
                print("Warning: Bessel Beam not implemented, using Gaussian.")
            elif idx == 4: # Custom
                custom_cfg = src_cfg.get('custom', {})
                eq = custom_cfg.get('equation', '')
                vars_list = custom_cfg.get('variables', [])
                # Convert vars list to dict
                vars_dict = {}
                for v in vars_list:
                    try:
                        vars_dict[v['name']] = float(v['value'])
                    except:
                        pass
                source = CustomSource(self.grid, amplitude=amp, equation=eq, 
                                      variables=vars_dict,
                                      polarization_type=pol_type,
                                      linear_angle=lin_angle)
            
            current_field = source.generate(device=self.device)
            
            if src_cfg.get('normalize', False):
                current_field.normalize()

            propagator = AngularSpectrumPropagator(self.grid)
            
            # Retrieve parameters (Using config directly or helper?)
            # Helper get_modulator_config reads from UI. 
            # We should update it or parse from config.
            # config contains 'mod1', 'mod2' dicts.
            
            mod1_config = config.get('mod1', {})
            mod2_config = config.get('mod2', {})
            
            # Fix units for mod config (stored as raw in project data)
            # get_modulator_config did conversion.
            def fix_mod_units(m_cfg):
                # z
                z_val = m_cfg.get('z', 0)
                z_unit = m_cfg.get('z_unit', 'um')
                if z_unit == 'mm': z_val *= 1e-3
                elif z_unit == 'um': z_val *= 1e-6
                m_cfg['z'] = z_val # Update in place/copy
                
                # lens
                if 'lens' in m_cfg:
                    l = m_cfg['lens']
                    if 'f' in l:
                        f_val = l['f']
                        f_unit = l.get('f_unit', 'um')
                        if f_unit == 'mm': f_val *= 1e-3
                        elif f_unit == 'um': f_val *= 1e-6
                        m_cfg['f'] = f_val # Flatten for apply_modulator usage
                    else:
                        m_cfg['f'] = 0.1 # Default 100mm
                else:
                    m_cfg['f'] = 0.1 # Default
                        
                return m_cfg

            mod1_config = fix_mod_units(mod1_config.copy())
            mod2_config = fix_mod_units(mod2_config.copy())
            
            monitors_config = config.get('monitors', [])
            
            # Identify Monitor Types
            global_monitors = [] # YZ, XZ
            xy_monitors = [] # XY
                
            for m in monitors_config:
                # m['pos'] is in um or mm. m['plane'] is type.
                plane_type = m['plane']
                pos_val = m.get('pos', m.get('z', 0))
                pos_unit = m.get('pos_unit', 'um')
                
                if pos_unit == 'mm':
                    fixed_val_m = pos_val * 1e-3
                else:
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
                                  ranges=ranges,
                                  output_components=m.get('output_components', []))
                
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
                max_xy = max(m['z'] for m in xy_monitors) # m['z'] here is fixed_val_m (meters)
                max_z = max(max_z, max_xy)
            
            # Check Global Monitors (YZ/XZ) for Z range coverage
            for m in monitors_config:
                plane_type = m.get('plane', 0)
                if plane_type in [1, 2]: # YZ (1) or XZ (2)
                    # For these planes, range2 is the Z-axis range
                    z_max = m.get('range2_max', -1e9) * 1e-6 # Convert um to meters
                    if z_max > max_z:
                        max_z = z_max
                
            # If max_z is small, default to 1mm
            if max_z < 1e-6: max_z = 1000e-6
            
            # Collect critical Z points
            z_points = set()
            z_points.add(0.0)
            z_points.add(max_z) # Ensure full range is covered
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

            # Inject Source Preview
            if self.source_preview_data:
                d = self.source_preview_data
                self.visualization_panel.add_monitor_result("Source Preview", 
                                                            d['field'], d['intensity'], d['phase'], d['x'], d['y'], 
                                                            components=d['components'], enabled=True)
            else:
                self.visualization_panel.add_monitor_result("Source Preview", 
                                                            None, None, None, None, None, enabled=False)

            self.status_bar.showMessage(f"Simulation complete.")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Simulation Error", f"An error occurred during simulation:\n{str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    def generate_geom_mask(self, prefix, c_type):
        """
        Generate mask from geometric params
        """
        pp = self.parameter_panel
        id_p = f"{prefix}_{c_type}"
        shape_idx = getattr(pp, f"combo_shape_{id_p}").currentIndex()
        
        X_um = self.grid.X * 1e6
        Y_um = self.grid.Y * 1e6
        
        mask = None
        
        if shape_idx == 0: # Annular
            r_in = getattr(pp, f"sb_ann_r_in_{id_p}").value()
            r_out = getattr(pp, f"sb_ann_r_out_{id_p}").value()
            cx = getattr(pp, f"sb_ann_cx_{id_p}").value()
            cy = getattr(pp, f"sb_ann_cy_{id_p}").value()
            start_angle = getattr(pp, f"sb_ann_angle_start_{id_p}").value()
            end_angle = getattr(pp, f"sb_ann_angle_end_{id_p}").value()
            val = getattr(pp, f"sb_ann_val_{id_p}").value()
            mask = generate_annular_mask(X_um, Y_um, cx, cy, r_in, r_out, val, start_angle, end_angle)
            
        elif shape_idx == 1: # Circle
            r = getattr(pp, f"sb_cir_r_{id_p}").value()
            cx = getattr(pp, f"sb_cir_cx_{id_p}").value()
            cy = getattr(pp, f"sb_cir_cy_{id_p}").value()
            val = getattr(pp, f"sb_cir_val_{id_p}").value()
            mask = generate_circular_mask(X_um, Y_um, cx, cy, r, val)
            
        elif shape_idx == 2: # Rectangle
            w = getattr(pp, f"sb_rect_w_{id_p}").value()
            h = getattr(pp, f"sb_rect_h_{id_p}").value()
            cx = getattr(pp, f"sb_rect_cx_{id_p}").value()
            cy = getattr(pp, f"sb_rect_cy_{id_p}").value()
            rot = getattr(pp, f"sb_rect_rot_{id_p}").value()
            val = getattr(pp, f"sb_rect_val_{id_p}").value()
            mask = generate_rectangular_mask(X_um, Y_um, cx, cy, w, h, rot, val)
            
        elif shape_idx == 3: # Polygon
            verts = getattr(pp, f"poly_editor_{id_p}").get_vertices()
            val = getattr(pp, f"sb_poly_val_{id_p}").value()
            mask = generate_polygon_mask(X_um, Y_um, verts, val)
            
        return mask

    def apply_modulator(self, field, event):
        config = event['config']
        type_idx = config['type_idx']
        type_ = event['type']
        
        # Get source params for polarization angle
        # Use config if possible, or fallback to UI read
        # Since apply_modulator is called during run loop, we can pass pol_angle
        # But to keep signature simple, let's just read from latest config
        config_full = self.parameter_panel.get_latest_config()
        src_cfg = config_full.get('source', {})
        pol_angle = float(src_cfg.get('linear_angle', 0.0))
        
        pols = config.get('affected_polarizations', ['unpolarized'])
        
        mod_kwargs = {
            'polarizations': pols,
            'polarization_angle': pol_angle
        }
        
        if type_idx == 0: # Custom Mask
            prefix = type_
            
            # Masks
            amp_mask = None
            phase_mask = None
            trans_formula = None
            phase_formula = None
            custom_vars = {}
            
            pp = self.parameter_panel
            if hasattr(pp, f"mask_tabs_{prefix}"):
                tabs = getattr(pp, f"mask_tabs_{prefix}")
                mode = tabs.currentIndex()
                
                if mode == 0: # File Import
                    amp_mask = getattr(pp, f"{prefix}_amp", None)
                    phase_mask = getattr(pp, f"{prefix}_phase", None)
                    
                elif mode == 1: # Param Definition
                    # Transmission
                    t_mode = getattr(pp, f"combo_trans_mode_{prefix}").currentIndex()
                    if t_mode == 0: # Formula
                        trans_formula = getattr(pp, f"fw_trans_{prefix}").get_formula()
                        custom_vars.update(getattr(pp, f"fw_trans_{prefix}").custom_vars)
                    else: # Geometric
                        amp_mask = self.generate_geom_mask(prefix, 'trans')
                        
                    # Phase
                    p_mode = getattr(pp, f"combo_phase_mode_{prefix}").currentIndex()
                    if p_mode == 0: # Formula
                        phase_formula = getattr(pp, f"fw_phase_{prefix}").get_formula()
                        custom_vars.update(getattr(pp, f"fw_phase_{prefix}").custom_vars)
                    else: # Geometric
                        phase_mask = self.generate_geom_mask(prefix, 'phase')

            if prefix == 'mod1':
                mod = SpatialModulator(self.grid, 
                                        amplitude_mask=amp_mask,
                                        phase_mask=phase_mask,
                                        transFormula=trans_formula,
                                        phaseFormula=phase_formula,
                                        customVars=custom_vars,
                                        **mod_kwargs)
                return mod.modulate(field)
            elif prefix == 'mod2':
                # Apply Spatial Modulation
                mod_spatial = SpatialModulator(self.grid, 
                                               amplitude_mask=amp_mask,
                                               phase_mask=phase_mask, 
                                               transFormula=trans_formula,
                                               phaseFormula=phase_formula,
                                               customVars=custom_vars,
                                               **mod_kwargs)
                field = mod_spatial.modulate(field)
                
                mod_angle = AngleModulator(self.grid, angle_transmission_curve=pp.mod2_angle_trans, **mod_kwargs) 
                return mod_angle.modulate(field)
                
        elif type_idx == 1: # Ideal Lens
            lens = IdealLens(self.grid, focal_length=config['f'], **mod_kwargs)
            return lens.modulate(field)
        elif type_idx == 2: # Cyl X
            lens = CylindricalLens(self.grid, focal_length=config['f'], axis='x', **mod_kwargs)
            return lens.modulate(field)
        elif type_idx == 3: # Cyl Y
            lens = CylindricalLens(self.grid, focal_length=config['f'], axis='y', **mod_kwargs)
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
            
        # Phase
        if field is not None:
            phase = np.angle(field)
        elif monitor.component_data:
            # Fallback to first available component
            k = next(iter(monitor.component_data))
            phase = np.angle(monitor.component_data[k])
        elif intensity is not None:
            phase = np.zeros_like(intensity)
        else:
            print(f"Warning: No valid data for phase calculation in monitor '{name}'")
            return
        
        # Prepare components for visualization panel
        # monitor.component_data contains {'Ex': ..., 'Ey': ..., 'Ez': ...}
        # We need to pass this dictionary to visualization panel
        
        self.visualization_panel.add_monitor_result(name, field, intensity, phase, x, y, 
                                                    components=monitor.component_data,
                                                    plane_type=monitor.plane_type)
