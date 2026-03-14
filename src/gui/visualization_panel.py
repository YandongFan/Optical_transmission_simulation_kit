from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QSizePolicy, QComboBox, QLabel, QHBoxLayout, QMessageBox, QFileDialog, QListWidget, QGroupBox, QPushButton, QDialog)
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
        
        # 监视器列表 (Monitor List)
        mon_group = QGroupBox("监视器列表 (Monitor List)")
        mon_layout = QHBoxLayout(mon_group)
        
        self.list_monitors = QListWidget() # Replaces combo_monitors
        self.list_monitors.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_monitors.itemDoubleClicked.connect(self.on_monitor_double_clicked)
        self.list_monitors.itemSelectionChanged.connect(self.on_monitor_selection_changed)
        self.list_monitors.setFixedHeight(120) # Limit height
        mon_layout.addWidget(self.list_monitors)
        
        btn_layout = QVBoxLayout()
        self.btn_view = QPushButton("查看 (View)")
        self.btn_view.clicked.connect(self.view_monitor_floating)
        self.btn_compare = QPushButton("对比 (Compare)")
        self.btn_compare.clicked.connect(self.compare_monitors)
        self.btn_export = QPushButton("导出 (Export)")
        self.btn_export.clicked.connect(self.export_current_monitor)
        
        btn_layout.addWidget(self.btn_view)
        btn_layout.addWidget(self.btn_compare)
        btn_layout.addWidget(self.btn_export)
        btn_layout.addStretch()
        mon_layout.addLayout(btn_layout)
        
        layout.addWidget(mon_group)
        
        # 数据存储 (Data storage)
        self.monitor_data = {} # {monitor_name: {...}}
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
            self.on_monitor_selection_changed()
            
        elapsed = (time.time() - start_time) * 1000
        # 打印耗时以验证性能要求 (< 200ms)
        print(f"Display mode switch to {self.current_aspect_mode}: {elapsed:.2f} ms")

    def clear_data(self):
        """
        清空所有数据
        """
        self.monitor_data = {}
        self.list_monitors.clear()
        self.intensity_canvas.clear()
        self.phase_canvas.clear()
        self.cross_section_canvas.clear()
        # 移除额外的分量标签页
        while self.tabs.count() > 3:
            self.tabs.removeTab(3)

    def add_monitor_result(self, name, field_data, intensity_data, phase_data, x, y, components=None, plane_type=0, enabled=True):
        """
        添加监视器结果数据
        :param enabled: 是否启用 (用于 Source Preview 无数据时)
        """
        self.monitor_data[name] = {
            'field': field_data,
            'intensity': intensity_data,
            'phase': phase_data,
            'x': x,
            'y': y,
            'components': components if components else {},
            'plane_type': plane_type,
            'enabled': enabled
        }
        
        # Check if exists
        items = self.list_monitors.findItems(name, Qt.MatchFlag.MatchExactly)
        if not items:
            self.list_monitors.addItem(name)
            item = self.list_monitors.findItems(name, Qt.MatchFlag.MatchExactly)[0]
        else:
            item = items[0]
            
        if not enabled:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            item.setToolTip("无可用光源预览数据 (No available source preview data)")
        else:
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
            item.setToolTip("")

        # Select if first
        if self.list_monitors.count() == 1 and enabled:
            self.list_monitors.setCurrentRow(0)

    def on_monitor_selection_changed(self):
        """
        切换显示的监视器数据
        """
        items = self.list_monitors.selectedItems()
        if not items: return
        
        name = items[-1].text()
        if name in self.monitor_data:
            data = self.monitor_data[name]
            if not data.get('enabled', True): return
            
            self.update_plots(data['field'], data['intensity'], data['phase'], data['x'], data['y'], 
                              data.get('components'), data.get('plane_type', 0))

    def on_monitor_double_clicked(self, item):
        self.view_monitor_floating()

    def view_monitor_floating(self):
        items = self.list_monitors.selectedItems()
        if not items: return
        name = items[-1].text()
        
        if name in self.monitor_data:
            data = self.monitor_data[name]
            if not data.get('enabled', True): return
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Monitor View: {name}")
            dialog.resize(800, 600)
            layout = QVBoxLayout(dialog)
            
            tabs = QTabWidget()
            
            def create_tab(title, data_arr, cmap='viridis', is_line=False):
                canvas = PlotCanvas(dialog)
                if is_line:
                    mid = data_arr.shape[0] // 2
                    if data['x'].ndim == 2:
                        x_line = data['x'][mid, :]
                    else:
                        x_line = data['x']
                    canvas.plot_line(x_line, data_arr[mid, :], title, "x (um)", "Intensity")
                else:
                    self.plot_heatmap_on_canvas(canvas, data_arr, data['x'], data['y'], title, cmap=cmap, mode=self.current_aspect_mode)
                tabs.addTab(canvas, title)
                
            create_tab("Intensity", data['intensity'])
            create_tab("Phase", data['phase'], cmap='hsv')
            create_tab("Cross Section", data['intensity'], is_line=True)
            
            layout.addWidget(tabs)
            
            btn_export = QPushButton("导出 PNG/CSV (Export)")
            btn_export.clicked.connect(lambda: self.export_data(name)) 
            layout.addWidget(btn_export)
            
            dialog.show() 

    def compare_monitors(self):
        items = self.list_monitors.selectedItems()
        if len(items) != 2:
            QMessageBox.warning(self, "Selection", "请选择两个监视器进行对比 (Please select exactly 2 monitors).")
            return
            
        name1 = items[0].text()
        name2 = items[1].text()
        
        d1 = self.monitor_data.get(name1)
        d2 = self.monitor_data.get(name2)
        
        if not d1 or not d2: return
        
        if d1['intensity'].shape != d2['intensity'].shape:
             QMessageBox.warning(self, "Mismatch", "Dimensions mismatch.")
             return
             
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Comparison: {name1} vs {name2}")
        dialog.resize(1000, 600)
        layout = QVBoxLayout(dialog)
        
        tabs = QTabWidget()
        
        # Side-by-Side
        w_sbs = QWidget()
        l_sbs = QHBoxLayout(w_sbs)
        
        c1 = PlotCanvas(dialog)
        self.plot_heatmap_on_canvas(c1, d1['intensity'], d1['x'], d1['y'], f"{name1}", mode=self.current_aspect_mode)
        l_sbs.addWidget(c1)
        
        c2 = PlotCanvas(dialog)
        self.plot_heatmap_on_canvas(c2, d2['intensity'], d2['x'], d2['y'], f"{name2}", mode=self.current_aspect_mode)
        l_sbs.addWidget(c2)
        
        tabs.addTab(w_sbs, "并排 (Side-by-Side)")
        
        # Difference
        c_diff = PlotCanvas(dialog)
        diff = d1['intensity'] - d2['intensity']
        self.plot_heatmap_on_canvas(c_diff, diff, d1['x'], d1['y'], f"Diff ({name1}-{name2})", cmap='coolwarm', mode=self.current_aspect_mode)
        tabs.addTab(c_diff, "差值 (Difference)")
        
        layout.addWidget(tabs)
        dialog.show()

    def export_current_monitor(self):
        items = self.list_monitors.selectedItems()
        if items: self.export_data(items[-1].text())

    def plot_heatmap_on_canvas(self, canvas, data, x, y, title, xlabel="x (um)", ylabel="y (um)", cmap='viridis', mode='default'):
        if x.ndim == 2:
            extent = [x.min(), x.max(), y.min(), y.max()]
        else:
            extent = [x[0], x[-1], y[0], y[-1]]
        canvas.plot_heatmap(data, extent, title, xlabel, ylabel, cmap, mode)

    def update_plots(self, field_data, intensity_data, phase_data, x, y, components=None, plane_type=0):
        """
        更新所有绘图区域
        """
        # 确定坐标范围
        if x.ndim == 2:
            extent = [x.min(), x.max(), y.min(), y.max()]
        else:
            extent = [x[0], x[-1], y[0], y[-1]]
            
        # Determine labels based on plane type
        xlabel, ylabel = "x (um)", "y (um)"
        if plane_type == 1: # YZ -> Horizontal Z, Vertical Y
            xlabel, ylabel = "z (um)", "y (um)"
        elif plane_type == 2: # XZ -> Horizontal Z, Vertical X
            xlabel, ylabel = "z (um)", "x (um)"
            
        # 更新主标签页
        self.intensity_canvas.plot_heatmap(intensity_data, extent, "Total Intensity (|E|^2)", 
                                           xlabel=xlabel, ylabel=ylabel, mode=self.current_aspect_mode)
        self.phase_canvas.plot_heatmap(phase_data, extent, "Phase (rad)", cmap='hsv', 
                                       xlabel=xlabel, ylabel=ylabel, mode=self.current_aspect_mode)
        
        # 更新截面图
        mid_row = intensity_data.shape[0] // 2
        if x.ndim == 2:
            x_line = x[mid_row, :]
        else:
            x_line = x
        
        y_label_val = y.mean() if hasattr(y, 'mean') else y[mid_row]
        self.cross_section_canvas.plot_line(x_line, intensity_data[mid_row, :], 
                                          f"Cross Section ({ylabel.split(' ')[0]}={y_label_val:.2f})", xlabel, "Intensity")

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
                    xlabel=xlabel, ylabel=ylabel,
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

    def plot_heatmap(self, data, extent, title, xlabel="x (um)", ylabel="y (um)", cmap='viridis', mode='default'):
        """
        绘制热图 (Intensity/Phase)
        """
        self.fig.clf() 
        ax = self.fig.add_subplot(111)
        im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
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

    def plot_dual_heatmap(self, data1, data2, extent, title1, title2, xlabel="x (um)", ylabel="y (um)", cmap1='viridis', cmap2='hsv', mode='default'):
        """
        绘制双热图 (用于分量显示)
        """
        self.fig.clf()
        
        # Subplot 1
        ax1 = self.fig.add_subplot(121)
        im1 = ax1.imshow(data1, extent=extent, origin='lower', cmap=cmap1, aspect='auto')
        ax1.set_title(title1)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        self.fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Subplot 2
        ax2 = self.fig.add_subplot(122)
        im2 = ax2.imshow(data2, extent=extent, origin='lower', cmap=cmap2, aspect='auto')
        ax2.set_title(title2)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
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
