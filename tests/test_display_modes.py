import unittest
import numpy as np
import sys
import os
from PyQt6.QtWidgets import QApplication

# Mocking matplotlib to avoid display issues in headless environment
# But we need real FigureCanvas logic for resizeEvent testing? 
# Matplotlib backend_agg is fine.

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.gui.visualization_panel import PlotCanvas

app = QApplication(sys.argv)

class TestDisplayModes(unittest.TestCase):
    def test_aspect_ratios(self):
        canvas = PlotCanvas(width=8, height=6)
        data = np.zeros((100, 200)) # 100 rows (y), 200 cols (x)
        extent = [0, 200, 0, 100] # x: 0-200, y: 0-100. Physical aspect 2:1.
        
        # 1. Default (Auto)
        canvas.plot_heatmap(data, extent, "Title", mode='default')
        ax = canvas.fig.axes[0]
        self.assertEqual(ax.get_aspect(), 'auto')
        
        # 2. Equal (Physical)
        # Should be 'equal' -> 1.0 (since x unit = y unit)
        canvas.plot_heatmap(data, extent, "Title", mode='equal')
        self.assertEqual(ax.get_aspect(), 1.0)
        
        # 3. Image (Square Pixels)
        # Pixel w = 200/200 = 1. Pixel h = 100/100 = 1.
        # Square pixels naturally. Aspect should be 1.0.
        canvas.plot_heatmap(data, extent, "Title", mode='image')
        self.assertAlmostEqual(ax.get_aspect(), 1.0)
        
        # Case: Non-square pixels
        # 100 rows, 100 cols.
        # Extent x: 0-200, y: 0-100.
        # Pixel w = 2, Pixel h = 1.
        # Aspect for square pixels = w/h = 2.0.
        data2 = np.zeros((100, 100))
        canvas.plot_heatmap(data2, extent, "Title", mode='image')
        self.assertAlmostEqual(ax.get_aspect(), 2.0)
        
        # Verify resize event doesn't crash
        canvas.resize(1200, 800)
        canvas.resize(640, 480)

if __name__ == '__main__':
    unittest.main()
