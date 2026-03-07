
import unittest
import numpy as np
import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.field import Grid, OpticalField
from src.core.monitor import Monitor
from src.core.propagator import AngularSpectrumPropagator
from src.core.source import GaussianBeam

class TestMonitorLogic(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(nx=64, ny=64, dx=1e-6, dy=1e-6, wavelength=0.5e-6)
        self.device = 'cpu'
        
        # Create a source
        self.source = GaussianBeam(self.grid, w0=10e-6, z=0)
        self.field = self.source.generate(device=self.device)
        self.propagator = AngularSpectrumPropagator(self.grid)

    def test_yz_monitor(self):
        # Case A: YZ Plane, X=0.5um
        fixed_x = 0.5e-6
        monitor = Monitor(position_z=0, name="YZ_Test", plane_type=1, fixed_value=fixed_x)
        
        # Geometry info check
        info = monitor.get_geometry_info()
        self.assertIn("Type=YZ", info)
        self.assertIn("X=0.500 um", info)
        self.assertIn("Normal=(1,0,0)", info)
        
        # Simulate propagation (manual stepping)
        current_field = self.field
        z_steps = [0, 10e-6, 20e-6]
        
        for z in z_steps:
            if z > 0:
                current_field = self.propagator.propagate(current_field, 10e-6)
            monitor.record(current_field, z)
            
        monitor.finalize()
        
        # Check data shape
        # YZ data shape: (Ny, Nz)
        expected_shape = (self.grid.ny, len(z_steps))
        self.assertEqual(monitor.intensity_data.shape, expected_shape)
        
        # Check X position correctness
        self.assertAlmostEqual(monitor.fixed_value, fixed_x)

    def test_xz_monitor(self):
        # Case B: XZ Plane, Y=-0.2um
        fixed_y = -0.2e-6
        monitor = Monitor(position_z=0, name="XZ_Test", plane_type=2, fixed_value=fixed_y)
        
        info = monitor.get_geometry_info()
        self.assertIn("Type=XZ", info)
        self.assertIn("Y=-0.200 um", info) # -0.200
        
        # Simulate
        current_field = self.field
        z_steps = [0, 5e-6]
        for z in z_steps:
            if z > 0:
                current_field = self.propagator.propagate(current_field, 5e-6)
            monitor.record(current_field, z)
        monitor.finalize()
        
        expected_shape = (self.grid.nx, len(z_steps))
        self.assertEqual(monitor.intensity_data.shape, expected_shape)

    def test_dual_monitors_diff(self):
        # Case C: YZ and XZ monitors, check difference
        mon1 = Monitor(position_z=0, name="YZ", plane_type=1, fixed_value=5e-6)
        mon2 = Monitor(position_z=0, name="XZ", plane_type=2, fixed_value=0.0)
        
        current_field = self.field
        z_steps = [0, 10e-6]
        for z in z_steps:
            if z > 0:
                current_field = self.propagator.propagate(current_field, 10e-6)
            mon1.record(current_field, z)
            mon2.record(current_field, z)
            
        mon1.finalize()
        mon2.finalize()
        
        peak1 = np.max(mon1.intensity_data)
        peak2 = np.max(mon2.intensity_data)
        
        self.assertLess(peak1, peak2)
        diff = abs(peak1 - peak2) / max(peak1, peak2)
        self.assertGreater(diff, 0.01)

    def test_position_sensitivity(self):
        # Case D: Sensitivity check
        # Two YZ monitors at slightly different X
        x1 = 0.0
        x2 = 2e-6 # 2um offset
        
        mon1 = Monitor(position_z=0, name="YZ1", plane_type=1, fixed_value=x1)
        mon2 = Monitor(position_z=0, name="YZ2", plane_type=1, fixed_value=x2)
        
        current_field = self.field
        z_steps = [0, 10e-6]
        for z in z_steps:
            if z > 0:
                current_field = self.propagator.propagate(current_field, 10e-6)
            mon1.record(current_field, z)
            mon2.record(current_field, z)
            
        mon1.finalize()
        mon2.finalize()
        
        # Gaussian beam intensity drops with r
        peak1 = np.max(mon1.intensity_data)
        peak2 = np.max(mon2.intensity_data)
        
        self.assertGreater(peak1, peak2)
        
        # Ensure arrays are not identical
        self.assertFalse(np.allclose(mon1.intensity_data, mon2.intensity_data))

if __name__ == '__main__':
    unittest.main()
