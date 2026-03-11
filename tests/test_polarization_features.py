
import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.field import OpticalField, Grid
from src.core.source import PlaneWave
from src.core.modulator import SpatialModulator

class TestPolarization(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(nx=10, ny=10, dx=1e-6, dy=1e-6, wavelength=0.532e-6)
        
    def test_linear_polarization(self):
        # 0 degrees (X)
        src = PlaneWave(self.grid, polarization_type=0, linear_angle=0)
        field = src.generate()
        # Ex should be 1, Ey should be 0
        self.assertTrue(torch.allclose(torch.abs(field.Ex), torch.tensor(1.0)))
        self.assertTrue(torch.allclose(torch.abs(field.Ey), torch.tensor(0.0), atol=1e-6))
        
        # 90 degrees (Y)
        src = PlaneWave(self.grid, polarization_type=0, linear_angle=90)
        field = src.generate()
        self.assertTrue(torch.allclose(torch.abs(field.Ex), torch.tensor(0.0), atol=1e-6))
        self.assertTrue(torch.allclose(torch.abs(field.Ey), torch.tensor(1.0)))
        
    def test_circular_polarization(self):
        # LCP: Ex=i/sqrt(2), Ey=1/sqrt(2)
        src = PlaneWave(self.grid, polarization_type=1)
        field = src.generate()
        Ex = field.Ex[5,5].item()
        Ey = field.Ey[5,5].item()
        
        # Check ratio Ex/Ey = i (if Ex=i/sqrt2, Ey=1/sqrt2)
        # My impl: Jx = 1j/sqrt(2), Jy = 1/sqrt(2) -> Ex/Ey = j
        self.assertAlmostEqual(Ex / Ey, 1j, places=5)
        
    def test_modulator_sensitivity(self):
        # Source: Linear 45 deg source: Ex=0.707, Ey=0.707
        src = PlaneWave(self.grid, polarization_type=0, linear_angle=45)
        field = src.generate()
        
        # Modulator: Blocks X (T=0 for X), Passes Y (T=1 for Y)
        # T=0 (amplitude mask 0), affected=['linear_x'] (angle 0)
        # If 'linear_x' affected, T applies to X. T=0 -> X blocked.
        # Y is unaffected -> Identity -> Y passes.
        
        mod = SpatialModulator(self.grid, amplitude_mask=np.zeros((10,10)), 
                               polarizations=['linear_x'], polarization_angle=0)
        out_field = mod.modulate(field)
        
        # Ex should be 0 (modulated by 0)
        # Ey should be 0.707 (unaffected)
        
        self.assertTrue(torch.allclose(torch.abs(out_field.Ex), torch.tensor(0.0), atol=1e-6))
        self.assertTrue(torch.allclose(torch.abs(out_field.Ey), torch.tensor(0.707106), atol=1e-4))
        
    def test_modulator_lcp_block(self):
        # Source: LCP
        src = PlaneWave(self.grid, polarization_type=1)
        field = src.generate()
        
        # Modulator: Blocks LCP (T=0 for LCP)
        mod = SpatialModulator(self.grid, amplitude_mask=np.zeros((10,10)), 
                               polarizations=['lcp'])
        out_field = mod.modulate(field)
        
        # Should be zero output
        self.assertTrue(torch.allclose(torch.abs(out_field.Ex), torch.tensor(0.0), atol=1e-6))
        self.assertTrue(torch.allclose(torch.abs(out_field.Ey), torch.tensor(0.0), atol=1e-6))
        
        # Source: RCP
        src_rcp = PlaneWave(self.grid, polarization_type=2)
        field_rcp = src_rcp.generate()
        
        # Modulator: Blocks LCP
        # RCP should pass unaffected
        out_rcp = mod.modulate(field_rcp)
        
        # Input RCP intensity ~ 1
        # Output RCP intensity ~ 1
        self.assertTrue(torch.allclose(out_rcp.get_intensity(), torch.tensor(1.0), atol=1e-4))

if __name__ == '__main__':
    unittest.main()
