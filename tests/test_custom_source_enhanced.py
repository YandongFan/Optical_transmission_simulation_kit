
import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.field import Grid
from src.core.source import CustomSource

class TestCustomSourceEnhanced(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(nx=128, ny=128, dx=1e-6, dy=1e-6, wavelength=532e-9)
        
    def test_cylindrical_coordinates(self):
        # Test using r and phi
        # equation: r * exp(1j * phi)
        # Should correspond to x + iy
        
        source = CustomSource(self.grid, equation="r * exp(1j * phi)")
        field = source.generate()
        E = field.E.cpu().numpy()
        
        # Manually calculate
        x = np.linspace(-self.grid.nx/2 * self.grid.dx, self.grid.nx/2 * self.grid.dx, self.grid.nx)
        y = np.linspace(-self.grid.ny/2 * self.grid.dy, self.grid.ny/2 * self.grid.dy, self.grid.ny)
        X, Y = np.meshgrid(x, y)
        Expected = (X + 1j*Y)
        
        # Normalize to compare shape/phase (amplitude depends on source logic which is 1.0 * result)
        # Here amplitude is 1.0.
        
        np.testing.assert_allclose(E, Expected, atol=1e-10)
        
    def test_bessel_function(self):
        # Test if besselj is available
        source = CustomSource(self.grid, equation="besselj(0, r/1e-5)")
        field = source.generate()
        self.assertIsNotNone(field)
        
    def test_complex_math(self):
        # Test sqrt(-1)
        source = CustomSource(self.grid, equation="sqrt(-1)")
        field = source.generate()
        E = field.E.cpu().numpy()
        self.assertTrue(np.allclose(E, 1j))
        
    def test_error_handling(self):
        # Test invalid syntax
        source = CustomSource(self.grid, equation="invalid_syntax(((")
        with self.assertRaises(ValueError):
            source.generate()

if __name__ == '__main__':
    unittest.main()
