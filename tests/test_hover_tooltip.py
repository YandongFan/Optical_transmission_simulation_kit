import unittest
import numpy as np
import time
from src.gui.visualization_panel import PlotCanvas

class TestHoverInterpolation(unittest.TestCase):
    def setUp(self):
        # We don't need a real UI for the canvas to test the interpolation math
        self.canvas = PlotCanvas()

    def test_bilinear_interpolation_accuracy(self):
        """
        用 512×512 随机矩阵验证 1000 次随机悬停，断言提示框坐标与插值结果误差<1e-6
        """
        N = 512
        data = np.random.rand(N, N)
        extent = [-10.0, 10.0, -10.0, 10.0]
        
        x_min, x_max, y_min, y_max = extent
        
        # Test exact grid points first
        # Bottom left
        z_bl = self.canvas.get_interpolated_z(-10.0, -10.0, extent, data)
        self.assertAlmostEqual(z_bl, data[0, 0], places=6)
        
        # Top right
        z_tr = self.canvas.get_interpolated_z(10.0, 10.0, extent, data)
        self.assertAlmostEqual(z_tr, data[N-1, N-1], places=6)
        
        # 1000 random points
        np.random.seed(42)
        for _ in range(1000):
            # random fractional index
            px = np.random.uniform(0, N - 1)
            py = np.random.uniform(0, N - 1)
            
            # map to physical coordinate
            x = x_min + px / (N - 1) * (x_max - x_min)
            y = y_min + py / (N - 1) * (y_max - y_min)
            
            # Manual bilinear interp for reference
            x0, y0 = int(np.floor(px)), int(np.floor(py))
            x1, y1 = min(x0 + 1, N - 1), min(y0 + 1, N - 1)
            dx = px - x0
            dy = py - y0
            v00 = data[y0, x0]
            v10 = data[y0, x1]
            v01 = data[y1, x0]
            v11 = data[y1, x1]
            ref_z = (1 - dx)*(1 - dy)*v00 + dx*(1 - dy)*v10 + (1 - dx)*dy*v01 + dx*dy*v11
            
            z = self.canvas.get_interpolated_z(x, y, extent, data)
            self.assertIsNotNone(z)
            self.assertAlmostEqual(z, ref_z, places=6)
            
    def test_performance(self):
        """
        单帧数据点≥1024×1024 时，悬停到显示延迟≤30 ms
        """
        N = 1024
        data = np.random.rand(N, N)
        extent = [-10.0, 10.0, -10.0, 10.0]
        
        # We simulate the interpolation step which happens on hover
        times = []
        for _ in range(100):
            x = np.random.uniform(-10.0, 10.0)
            y = np.random.uniform(-10.0, 10.0)
            
            t0 = time.perf_counter()
            z = self.canvas.get_interpolated_z(x, y, extent, data)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            
        avg_time_ms = np.mean(times) * 1000
        max_time_ms = np.max(times) * 1000
        
        print(f"\nInterpolation perf on 1024x1024: avg={avg_time_ms:.4f}ms, max={max_time_ms:.4f}ms")
        self.assertLess(max_time_ms, 30.0)

if __name__ == '__main__':
    unittest.main()
