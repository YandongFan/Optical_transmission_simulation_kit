from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QSizePolicy, QComboBox, QLabel, QHBoxLayout, QMessageBox, QFileDialog)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import h5py
import pandas as pd
import time

class VisualizationPanel(QWidget):
    """
    可视化面板 (Visualization Panel)
    负责显示仿真结果，支持多种显示模式和数据导出。
    """
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # 监视器选择 (Monitor Selection)
        mon_layout = QHBoxLayout()
        mon_layout.addWidget(QLabel("选择监视器 (Select Monitor):"))
        self.combo_monitors = QComboBox()
        self.combo_monitors.currentIndexChanged.connect(self.on_monitor_changed)
        mon_layout.addWidget(self.combo_monitors)
        layout.addLayout(mon_layout)
        
        # 数据存储 (Data storage)
        self.monitor_data = {} # {monitor_name: {'field': ..., 'intensity': ..., 'phase': ..., 'x': ..., 'y': ...}}
        self.current_aspect_mode = 'default' # 'default', 'square', 'image'
        
        # 显示控制 (Display Control)
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("显示模式 (Display Mode):"))
        self.combo_aspect = QComboBox()
        # 对应三种模式：默认自动拉伸、物理1:1（Axis Square）、像素1:1（Axis Image）
        self.combo_aspect.addItems(["默认 (Default)", "Axis Square (Physical 1:1)", "Axis Image (Square Pixels)"])
        self.combo_aspect.currentIndexChanged.connect(self.on_aspect_changed)
        ctrl_layout.addWidget(self.combo_aspect)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)
        
        # 标签页 (Tabs for different views)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # 1. 光强分布 (Intensity Plot)
        self.intensity_canvas = PlotCanvas(self, width=5, height=4)
        self.tabs.addTab(self.intensity_canvas, "光强分布 (Intensity)")
        
        # 2. 相位分布 (Phase Plot)
        self.phase_canvas = PlotCanvas(self, width=5, height=4)
        self.tabs.addTab(self.phase_canvas, "相位分布 (Phase)")
        
        # 3. 截面分布 (Cross-section Plot)
        self.cross_section_canvas = PlotCanvas(self, width=5, height=4)
        self.tabs.addTab(self.cross_section_canvas, "截面分布 (Cross Section)")

    def on_aspect_changed(self, index):
        """
        处理显示模式切换
        """
        start_time = time.time()
        
        modes = ['default', 'square', 'image']
        if 0 <= index < len(modes):
            self.current_aspect_mode = modes[index]
            # 触发当前监视器的重绘
            self.on_monitor_changed(self.combo_monitors.currentIndex())
            
        elapsed = (time.time() - start_time) * 1000
        # 打印耗时以验证性能要求 (< 200ms)
        print(f"Display mode switch to {self.current_aspect_mode}: {elapsed:.2f} ms")

    def clear_data(self):
        """
        清空所有数据
        """
        self.monitor_data = {}
        self.combo_monitors.clear()
        self.intensity_canvas.clear()
        self.phase_canvas.clear()
        self.cross_section_canvas.clear()
        # 移除额外的分量标签页
        while self.tabs.count() > 3:
            self.tabs.removeTab(3)

    def add_monitor_result(self, name, field_data, intensity_data, phase_data, x, y, components=None):
        """
        添加监视器结果数据
        :param components: 分量字典 {'Ex': data, 'Ey': data, 'Ez': data}
        """
        self.monitor_data[name] = {
            'field': field_data,
            'intensity': intensity_data,
            'phase': phase_data,
            'x': x,
            'y': y,
            'components': components if components else {}
        }
        
        # 如果不存在则添加到下拉框
        if self.combo_monitors.findText(name) == -1:
            self.combo_monitors.addItem(name)
            
        # 如果是当前选中项或第一项，则更新视图
        if self.combo_monitors.currentText() == name:
            self.on_monitor_changed(self.combo_monitors.currentIndex())
        elif self.combo_monitors.count() == 1:
            self.on_monitor_changed(0)

    def on_monitor_changed(self, index):
        """
        切换显示的监视器数据
        """
        name = self.combo_monitors.currentText()
        if name in self.monitor_data:
            data = self.monitor_data[name]
            self.update_plots(data['field'], data['intensity'], data['phase'], data['x'], data['y'], 
                              data.get('components'))

    def update_plots(self, field_data, intensity_data, phase_data, x, y, components=None):
        """
        更新所有绘图区域
        """
        # 确定坐标范围
        if x.ndim == 2:
            extent = [x.min(), x.max(), y.min(), y.max()]
        else:
            extent = [x[0], x[-1], y[0], y[-1]]
            
        # 更新主标签页
        self.intensity_canvas.plot_heatmap(intensity_data, extent, "Total Intensity (|E|^2)", mode=self.current_aspect_mode)
        self.phase_canvas.plot_heatmap(phase_data, extent, "Phase (rad)", cmap='hsv', mode=self.current_aspect_mode)
        
        # 更新截面图
        mid_row = intensity_data.shape[0] // 2
        if x.ndim == 2:
            x_line = x[mid_row, :]
        else:
            x_line = x
        
        y_label_val = y.mean() if hasattr(y, 'mean') else y[mid_row]
        self.cross_section_canvas.plot_line(x_line, intensity_data[mid_row, :], 
                                          f"Cross Section (y={y_label_val:.2f} um)", "x (um)", "Intensity")

        # 处理分量显示 (Ex, Ey, Ez)
        # 移除现有的分量标签页 (索引 > 2)
        while self.tabs.count() > 3:
            self.tabs.removeTab(3)
            
        if components:
            for comp_name, comp_data in components.items():
                if comp_data is None: continue
                
                # 创建新的画布用于显示分量
                comp_canvas = PlotCanvas(self, width=8, height=4)
                self.tabs.addTab(comp_canvas, f"{comp_name} 分量 (Component)")
                
                comp_int = np.abs(comp_data)**2
                comp_phase = np.angle(comp_data)
                
                # 使用双热图显示分量的强度和相位
                comp_canvas.plot_dual_heatmap(
                    comp_int, comp_phase, extent, 
                    f"|{comp_name}|^2", f"Arg({comp_name})",
                    cmap1='viridis', cmap2='hsv',
                    mode=self.current_aspect_mode
                )

    def export_data(self, monitor_name):
        """
        导出指定监视器的数据到文件 (HDF5 or CSV)
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
                    if data['field'] is not None:
                        grp.create_dataset('field_real', data=np.real(data['field']))
                        grp.create_dataset('field_imag', data=np.imag(data['field']))
                    grp.create_dataset('intensity', data=data['intensity'])
                    grp.create_dataset('x', data=data['x'])
                    grp.create_dataset('y', data=data['y'])
            elif filename.endswith('.csv'):
                # 展平数据用于CSV存储
                x_flat = data['x'].flatten()
                y_flat = data['y'].flatten()
                int_flat = data['intensity'].flatten()
                
                if len(x_flat) == len(int_flat):
                    df = pd.DataFrame({
                        'x': x_flat,
                        'y': y_flat,
                        'intensity': int_flat
                    })
                    df.to_csv(filename, index=False)
                else:
                     # 坐标维度不匹配时不导出CSV
                     pass
            
            QMessageBox.information(self, "Success", f"Data exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

class PlotCanvas(FigureCanvas):
    """
    自定义 Matplotlib 画布，支持 resizeEvent 和多种绘图模式
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

    def resizeEvent(self, event):
        """
        处理窗口大小调整事件，保持布局紧凑
        """
        super().resizeEvent(event)
        self.fig.tight_layout()
        self.draw()

    def plot_heatmap(self, data, extent, title, cmap='viridis', mode='default'):
        """
        绘制热图 (Intensity/Phase)
        """
        self.fig.clf() 
        ax = self.fig.add_subplot(111)
        im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        
        if mode == 'square': # Axis Square (MATLAB style)
            # 强制坐标轴框为正方形，并确保物理 1:1 比例
            try:
                ax.set_box_aspect(1)
            except AttributeError:
                print("Warning: set_box_aspect not available (Matplotlib < 3.3)")
            ax.set_aspect('equal', adjustable='datalim')
        elif mode == 'image': # Axis Image (Square Pixels)
            # 计算宽高比以确保像素显示为正方形
            width = extent[1] - extent[0]
            height = extent[3] - extent[2]
            nx = data.shape[1]
            ny = data.shape[0]
            if ny > 0 and height > 0:
                pixel_w = width / nx
                pixel_h = height / ny
                if pixel_h > 0:
                    aspect_ratio = pixel_w / pixel_h
                    # adjustable='box': 调整坐标轴框的大小以紧贴数据，不添加额外空白
                    ax.set_aspect(aspect_ratio, adjustable='box')
                else:
                    ax.set_aspect('auto')
            else:
                ax.set_aspect('auto')
        else: # Default
            ax.set_aspect('auto')
            
        self.fig.colorbar(im, ax=ax)
        self.fig.tight_layout()
        self.draw()

    def plot_dual_heatmap(self, data1, data2, extent, title1, title2, cmap1='viridis', cmap2='hsv', mode='default'):
        """
        绘制双热图 (用于分量显示)
        """
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
        
        # 应用显示模式
        if mode == 'square':
            try:
                ax1.set_box_aspect(1)
                ax2.set_box_aspect(1)
            except AttributeError:
                pass
            ax1.set_aspect('equal', adjustable='datalim')
            ax2.set_aspect('equal', adjustable='datalim')
        elif mode == 'image':
            width = extent[1] - extent[0]
            height = extent[3] - extent[2]
            nx = data1.shape[1]
            ny = data1.shape[0]
            if ny > 0 and height > 0:
                pixel_w = width / nx
                pixel_h = height / ny
                if pixel_h > 0:
                    aspect = pixel_w / pixel_h
                    ax1.set_aspect(aspect, adjustable='box')
                    ax2.set_aspect(aspect, adjustable='box')
        
        self.fig.tight_layout()
        self.draw()

    def plot_line(self, x, y, title, xlabel, ylabel):
        """
        绘制曲线图 (截面)
        """
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
