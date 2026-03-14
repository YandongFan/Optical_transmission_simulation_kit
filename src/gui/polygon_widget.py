from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QPushButton, QHeaderView, QMessageBox)
from PyQt6.QtCore import pyqtSignal as Signal, Qt

class PolygonEditorWidget(QWidget):
    """
    多边形顶点编辑控件 (Polygon Vertex Editor Widget)
    允许用户动态添加、删除和编辑多边形顶点坐标 (x, y)。
    """
    dataChanged = Signal() # 当顶点数据发生变化时发出信号

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 顶点表格
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["X (um)", "Y (um)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.cellChanged.connect(self.on_cell_changed)
        layout.addWidget(self.table)

        # 按钮栏
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("+ 添加顶点 (Add Vertex)")
        self.btn_add.clicked.connect(self.add_vertex)
        self.btn_remove = QPushButton("- 删除顶点 (Remove Vertex)")
        self.btn_remove.clicked.connect(self.remove_vertex)
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        layout.addLayout(btn_layout)

        # 默认添加几个点
        self.set_vertices([(0, 100), (86.6, -50), (-86.6, -50)]) # 等边三角形

    def add_vertex(self):
        """添加一个新顶点"""
        row = self.table.rowCount()
        # 限制上限
        # 用户要求: 验证新增多边形顶点上限不低于 64 点
        # 这里我们可以设置一个更高的软限制，比如 1000
        if row >= 1000:
            QMessageBox.warning(self, "Limit Reached", "Maximum vertex count reached.")
            return

        self.table.blockSignals(True)
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem("0.0"))
        self.table.setItem(row, 1, QTableWidgetItem("0.0"))
        self.table.blockSignals(False)
        self.dataChanged.emit()

    def remove_vertex(self):
        """删除当前选中的顶点或最后一行"""
        row = self.table.currentRow()
        if row == -1:
            row = self.table.rowCount() - 1
        
        if row >= 0:
            self.table.blockSignals(True)
            self.table.removeRow(row)
            self.table.blockSignals(False)
            self.dataChanged.emit()

    def on_cell_changed(self, row, column):
        """当单元格内容改变时触发"""
        item = self.table.item(row, column)
        if not item: return

        text = item.text()
        try:
            float(text)
            item.setBackground(Qt.GlobalColor.white) # 恢复背景
            self.dataChanged.emit()
        except ValueError:
            # 验证坐标输入非法时给出红色高亮错误提示
            item.setBackground(Qt.GlobalColor.red)
            # 不阻塞，但也不发出 dataChanged 信号以免导致绘图错误
            # 或者发出信号但在处理时忽略无效值? 
            # 最好是不发出信号，直到修正

    def get_vertices(self):
        """获取所有有效的顶点坐标列表 [(x1, y1), ...]"""
        vertices = []
        for row in range(self.table.rowCount()):
            item_x = self.table.item(row, 0)
            item_y = self.table.item(row, 1)
            
            if item_x and item_y:
                try:
                    x = float(item_x.text())
                    y = float(item_y.text())
                    vertices.append((x, y))
                except ValueError:
                    continue # 跳过无效点
        return vertices

    def set_vertices(self, vertices):
        """设置顶点列表"""
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for i, (x, y) in enumerate(vertices):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(str(x)))
            self.table.setItem(i, 1, QTableWidgetItem(str(y)))
        self.table.blockSignals(False)
        # self.dataChanged.emit() # 初始化时不一定需要触发
