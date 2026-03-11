from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QSizePolicy, QComboBox, QLabel, QHBoxLayout, QMessageBox, QFileDialog)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import h5py
import pandas as pd

class VisualizationPanel(QWidget):
    """
    可视化面板 (Visualization Panel)
    """
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # Monitor Selection
        mon_layout = QHBoxLayout()
        mon_layout.addWidget(QLabel("选择监视器 (Select Monitor):"))
        self.combo_monitors = QComboBox()
        self.combo_monitors.currentIndexChanged.connect(self.on_monitor_changed)
        mon_layout.addWidget(self.combo_monitors)
        layout.addLayout(mon_layout)
        
        # Data storage
        self.monitor_data = {} # {monitor_name: {'field': ..., 'intensity': ..., 'phase': ..., 'x': ..., 'y': ...}}
        
        # Tabs for different views (e.g., Intensity, Phase, 3D)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # 1. Intensity Plot
        self.intensity_canvas = PlotCanvas(self, width=5, height=4)
        self.tabs.addTab(self.intensity_canvas, "光强分布 (Intensity)")
        
        # 2. Phase Plot
        self.phase_canvas = PlotCanvas(self, width=5, height=4)
        self.tabs.addTab(self.phase_canvas, "相位分布 (Phase)")
        
        # 3. Cross-section Plot (Optional)
        self.cross_section_canvas = PlotCanvas(self, width=5, height=4)
        self.tabs.addTab(self.cross_section_canvas, "截面分布 (Cross Section)")

    def clear_data(self):
        """
        清空数据 (Clear all data)
        """
        self.monitor_data = {}
        self.combo_monitors.clear()
        self.intensity_canvas.clear()
        self.phase_canvas.clear()
        self.cross_section_canvas.clear()

    def add_monitor_result(self, name, field_data, intensity_data, phase_data, x, y, components=None):
        """
        添加监视器结果 (Add monitor result)
        :param components: dict of {'Ex': data, 'Ey': data, 'Ez': data} (complex numpy arrays)
        """
        self.monitor_data[name] = {
            'field': field_data,
            'intensity': intensity_data,
            'phase': phase_data,
            'x': x,
            'y': y,
            'components': components if components else {}
        }
        
        # Add to combo box if not exists
        if self.combo_monitors.findText(name) == -1:
            self.combo_monitors.addItem(name)
            
        # If currently selected or first item, update view
        if self.combo_monitors.currentText() == name:
            self.on_monitor_changed(self.combo_monitors.currentIndex())
        elif self.combo_monitors.count() == 1:
            self.on_monitor_changed(0)

    def on_monitor_changed(self, index):
        """
        切换监视器 (Switch monitor view)
        """
        name = self.combo_monitors.currentText()
        if name in self.monitor_data:
            data = self.monitor_data[name]
            self.update_plots(data['field'], data['intensity'], data['phase'], data['x'], data['y'], 
                              data.get('components'))

    def update_plots(self, field_data, intensity_data, phase_data, x, y, components=None):
        """
        更新绘图 (Update plots)
        """
        # Determine extent
        if x.ndim == 2:
            extent = [x.min(), x.max(), y.min(), y.max()]
        else:
            extent = [x[0], x[-1], y[0], y[-1]]
            
        # Update Main Tabs
        self.intensity_canvas.plot_heatmap(intensity_data, extent, "Total Intensity (|E|^2)")
        self.phase_canvas.plot_heatmap(phase_data, extent, "Phase (rad)", cmap='hsv')
        
        # Cross Section
        mid_row = intensity_data.shape[0] // 2
        if x.ndim == 2:
            x_line = x[mid_row, :]
        else:
            x_line = x
        self.cross_section_canvas.plot_line(x_line, intensity_data[mid_row, :], 
                                          f"Cross Section (y={y.mean() if hasattr(y, 'mean') else y[mid_row]:.2f} um)", "x (um)", "Intensity")

        # Handle Component Tabs
        # Remove existing extra tabs (indices > 2)
        while self.tabs.count() > 3:
            self.tabs.removeTab(3)
            
        if components:
            for comp_name, comp_data in components.items():
                if comp_data is None: continue
                
                # Create canvas for 2 subplots
                comp_canvas = PlotCanvas(self, width=8, height=4)
                self.tabs.addTab(comp_canvas, f"{comp_name} 分量 (Component)")
                
                comp_int = np.abs(comp_data)**2
                comp_phase = np.angle(comp_data)
                
                comp_canvas.plot_dual_heatmap(
                    comp_int, comp_phase, extent, 
                    f"|{comp_name}|^2", f"Arg({comp_name})",
                    cmap1='viridis', cmap2='hsv'
                )

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

    def plot_heatmap(self, data, extent, title, cmap='viridis'):
        self.fig.clf() 
        ax = self.fig.add_subplot(111)
        im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        self.fig.colorbar(im, ax=ax)
        self.draw()

    def plot_dual_heatmap(self, data1, data2, extent, title1, title2, cmap1='viridis', cmap2='hsv'):
        self.fig.clf()
        
        # Subplot 1
        ax1 = self.fig.add_subplot(121)
        im1 = ax1.imshow(data1, extent=extent, origin='lower', cmap=cmap1, aspect='auto')
        ax1.set_title(title1)
        ax1.set_xlabel('x (um)')
        ax1.set_ylabel('y (um)')
        self.fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Subplot 2
        ax2 = self.fig.add_subplot(122)
        im2 = ax2.imshow(data2, extent=extent, origin='lower', cmap=cmap2, aspect='auto')
        ax2.set_title(title2)
        ax2.set_xlabel('x (um)')
        ax2.set_ylabel('y (um)')
        self.fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        self.fig.tight_layout()
        self.draw()

    def plot_line(self, x, y, title, xlabel, ylabel):
        """
        导出指定监视器的数据 (Export specific monitor data)
        """
        if monitor_name not in self.monitor_data:
            QMessageBox.warning(self, "Warning", f"No data available for monitor: {monitor_name}")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Export Data", f"{monitor_name}.h5", "HDF5 Files (*.h5);;CSV Files (*.csv)")
        if not filename:
            return
            
        data = self.monitor_data[monitor_name]
        try:
            if filename.endswith('.h5'):
                with h5py.File(filename, 'w') as f:
                    grp = f.create_group(monitor_name)
                    grp.create_dataset('field_real', data=np.real(data['field']))
                    grp.create_dataset('field_imag', data=np.imag(data['field']))
                    grp.create_dataset('intensity', data=data['intensity'])
                    grp.create_dataset('x', data=data['x'])
                    grp.create_dataset('y', data=data['y'])
            elif filename.endswith('.csv'):
                # Flatten for CSV - careful with large arrays
                x_flat = data['x'].flatten()
                y_flat = data['y'].flatten()
                int_flat = data['intensity'].flatten()
                
                # If sizes match (meshgrid)
                if len(x_flat) == len(int_flat):
                    df = pd.DataFrame({
                        'x': x_flat,
                        'y': y_flat,
                        'intensity': int_flat
                    })
                    df.to_csv(filename, index=False)
                else:
                     # Handle 1D coordinates vs 2D data
                     # Create meshgrid if needed or export differently
                     pass
            
            QMessageBox.information(self, "Success", f"Data exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

    def plot_heatmap(self, data, extent, title, cmap='viridis'):
        self.fig.clf() # Clear figure to remove old axes and colorbars
        ax = self.fig.add_subplot(111)
        
        im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        self.fig.colorbar(im, ax=ax)
        self.draw()

    def plot_dual_heatmap(self, data1, data2, extent, title1, title2, cmap1='viridis', cmap2='hsv'):
        self.fig.clf()
        
        # Subplot 1
        ax1 = self.fig.add_subplot(121)
        im1 = ax1.imshow(data1, extent=extent, origin='lower', cmap=cmap1, aspect='auto')
        ax1.set_title(title1)
        ax1.set_xlabel('x (um)')
        ax1.set_ylabel('y (um)')
        self.fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Subplot 2
        ax2 = self.fig.add_subplot(122)
        im2 = ax2.imshow(data2, extent=extent, origin='lower', cmap=cmap2, aspect='auto')
        ax2.set_title(title2)
        ax2.set_xlabel('x (um)')
        ax2.set_ylabel('y (um)')
        self.fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        self.fig.tight_layout()
        self.draw()

    def plot_line(self, x, y, title, xlabel, ylabel):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        self.draw()
        
    def clear(self):
        self.fig.clf()
        self.draw()
