from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QSizePolicy)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class VisualizationPanel(QWidget):
    """
    可视化面板 (Visualization Panel)
    """
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
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

    def update_plots(self, field_data, intensity_data, phase_data, x, y):
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
