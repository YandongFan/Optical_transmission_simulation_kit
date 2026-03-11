import unittest
import numpy as np
import torch
import sys
import os

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.monitor import Monitor
from src.core.field import OpticalField, Grid

class TestMonitorOptimization(unittest.TestCase):
    def setUp(self):
        # Create a basic field
        self.grid = Grid(nx=32, ny=32, dx=1e-6, dy=1e-6, wavelength=0.532e-6)
        self.field = OpticalField(self.grid)
        # Set some data
        self.field.Ex = torch.ones((32, 32), dtype=torch.complex64)
        self.field.Ey = torch.ones((32, 32), dtype=torch.complex64) * 2
        
    def test_record_all_components(self):
        monitor = Monitor(position_z=0, output_components=['Ex', 'Ey', 'Ez'])
        monitor.record(self.field, 0.0)
        
        self.assertIn('Ex', monitor.component_data)
        self.assertIn('Ey', monitor.component_data)
        self.assertIn('Ez', monitor.component_data)
        self.assertIsNotNone(monitor.field_data) # Legacy
        
    def test_record_partial_components(self):
        monitor = Monitor(position_z=0, output_components=['Ex'])
        monitor.record(self.field, 0.0)
        
        self.assertIn('Ex', monitor.component_data)
        self.assertNotIn('Ey', monitor.component_data)
        self.assertNotIn('Ez', monitor.component_data)
        self.assertIsNotNone(monitor.field_data)
        
        # Intensity should still be calculated
        self.assertIsNotNone(monitor.intensity_data)
        
    def test_record_no_components(self):
        monitor = Monitor(position_z=0, output_components=[])
        monitor.record(self.field, 0.0)
        
        self.assertNotIn('Ex', monitor.component_data)
        self.assertNotIn('Ey', monitor.component_data)
        self.assertNotIn('Ez', monitor.component_data)
        self.assertIsNone(monitor.field_data)
        
        # Intensity should still be calculated
        self.assertIsNotNone(monitor.intensity_data)

    def test_record_ez_only(self):
        monitor = Monitor(position_z=0, output_components=['Ez'])
        monitor.record(self.field, 0.0)
        
        self.assertNotIn('Ex', monitor.component_data)
        self.assertNotIn('Ey', monitor.component_data)
        self.assertIn('Ez', monitor.component_data)
        self.assertIsNone(monitor.field_data) # field_data is usually main pol (Ex)
        self.assertIsNotNone(monitor.intensity_data)

if __name__ == '__main__':
    unittest.main()
