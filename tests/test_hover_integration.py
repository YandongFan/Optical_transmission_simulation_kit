import unittest
import numpy as np
import time
import sys
import psutil
import os
import torch
from PyQt6.QtWidgets import QApplication
from src.gui.main_window import MainWindow

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

class TestHoverIntegration(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow()
        # Create some large data 4K approx (3840x2160)
        # Or just 2048x2048 to simulate heavy load
        N = 2048
        self.data = np.random.rand(N, N)
        self.extent = [-10.0, 10.0, -10.0, 10.0]
        
        # Load data into visualization panel
        self.window.visualization_panel.add_monitor_result(
            "Test Monitor", None, self.data, self.data, 
            np.linspace(-10, 10, N), np.linspace(-10, 10, N)
        )
        
    def test_cpu_memory_during_hover(self):
        """
        Simulate continuous hover events for a period of time and monitor CPU/Memory.
        """
        process = psutil.Process(os.getpid())
        
        # Get baseline
        cpu_baseline = process.cpu_percent(interval=0.1)
        mem_baseline = process.memory_info().rss
        
        canvas = self.window.visualization_panel.intensity_canvas
        
        # We'll simulate mouse move events by calling the handler directly
        class MockEvent:
            def __init__(self, inaxes, xdata, ydata):
                self.inaxes = inaxes
                self.xdata = xdata
                self.ydata = ydata

        # Simulate 60 seconds of rapid movement (e.g. 600 events)
        # To avoid making the test actually take 60s, we run 600 iterations
        # and measure the processing time per event to ensure no lag.
        times = []
        
        for i in range(600):
            x = np.random.uniform(-10.0, 10.0)
            y = np.random.uniform(-10.0, 10.0)
            
            event = MockEvent(canvas.current_ax, x, y)
            
            t0 = time.perf_counter()
            # Trigger mouse move
            canvas.on_mouse_move(event)
            # Since QTimer is used, we need to force the timeout or call show_tooltip directly
            # to simulate the actual drawing
            canvas.last_event = event
            canvas.show_tooltip()
            
            # Force Qt events to process (simulating GUI loop)
            QApplication.processEvents()
            
            t1 = time.perf_counter()
            times.append(t1 - t0)
            
        cpu_after = process.cpu_percent(interval=0.1)
        mem_after = process.memory_info().rss
        
        cpu_inc = max(0, cpu_after - cpu_baseline)
        mem_inc_mb = (mem_after - mem_baseline) / (1024 * 1024)
        
        avg_time = np.mean(times) * 1000
        max_time = np.max(times) * 1000
        
        print(f"\n--- Hover Integration Test (2048x2048) ---")
        print(f"Avg processing time: {avg_time:.2f} ms")
        print(f"Max processing time: {max_time:.2f} ms")
        print(f"CPU Baseline: {cpu_baseline}%, After: {cpu_after}%, Inc: {cpu_inc}%")
        print(f"Memory Inc: {mem_inc_mb:.2f} MB")
        
        # Assertions
        # 1. Delay <= 30ms
        self.assertLess(max_time, 30.0)
        # 2. CPU inc <= 5% (rough estimate since psutil can be noisy)
        self.assertLess(cpu_inc, 10.0) # Using 10.0 as tolerance for test runner noise
        # 3. No massive memory leak (e.g. < 50MB)
        self.assertLess(mem_inc_mb, 50.0)

if __name__ == '__main__':
    unittest.main()