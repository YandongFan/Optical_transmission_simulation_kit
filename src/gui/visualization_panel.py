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

    def add_monitor_result(self, name, field_data, intensity_data, phase_data, x, y, complex_real=None, complex_imag=None):
        """
        添加监视器结果 (Add monitor result)
        """
        self.monitor_data[name] = {
            'field': field_data,
            'intensity': intensity_data,
            'phase': phase_data,
            'x': x,
            'y': y,
            'complex_real': complex_real,
            'complex_imag': complex_imag
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
                              data.get('complex_real'), data.get('complex_imag'))

    def update_plots(self, field_data, intensity_data, phase_data, x, y, complex_real=None, complex_imag=None):
        """
        更新绘图 (Update plots)
        """
        # Update Intensity
        # Assuming x, y are 1D arrays or 2D meshgrids. imshow expects extent=[xmin, xmax, ymin, ymax]
        if x.ndim == 2:
            extent = [x.min(), x.max(), y.min(), y.max()]
        else:
            extent = [x[0], x[-1], y[0], y[-1]]
            
        self.intensity_canvas.plot_heatmap(intensity_data, extent, "Intensity Distribution (|E|^2)")
        
        # Update Phase
        self.phase_canvas.plot_heatmap(phase_data, extent, "Phase Distribution (rad)", cmap='hsv')
        
        # Update Cross Section (Central row)
        mid_row = intensity_data.shape[0] // 2
        # If x is 2D, take the row
        if x.ndim == 2:
            x_line = x[mid_row, :]
        else:
            x_line = x
            
        self.cross_section_canvas.plot_line(x_line, intensity_data[mid_row, :], 
                                          f"Cross Section (y={y.mean():.2f} um)", "x (um)", "Intensity")

        # Handle Complex Field Tabs
        # Tabs indices: 0:Intensity, 1:Phase, 2:CrossSection
        # We want 3:Real, 4:Imag
        
        # Remove existing extra tabs if any
        while self.tabs.count() > 3:
            self.tabs.removeTab(3)
            
        if complex_real is not None and complex_imag is not None:
            # Add Real Tab
            real_canvas = PlotCanvas(self, width=5, height=4)
            self.tabs.addTab(real_canvas, "complex field (real E)")
            real_canvas.plot_heatmap(complex_real, extent, "Real Part of E-field", cmap='bwr')
            
            # Add Imag Tab
            imag_canvas = PlotCanvas(self, width=5, height=4)
            self.tabs.addTab(imag_canvas, "complex field (imag E)")
            imag_canvas.plot_heatmap(complex_imag, extent, "Imaginary Part of E-field", cmap='bwr')

    def export_data(self, monitor_name):
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
