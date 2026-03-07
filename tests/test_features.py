import sys
import os
import numpy as np
import torch
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.field import Grid, OpticalField
from src.core.source import CustomSource
from src.core.modulator import IdealLens, CylindricalLens

def test_custom_source():
    """
    测试自定义光源 (Test Custom Source)
    """
    nx, ny = 129, 129 # Odd number to include origin
    dx, dy = 10e-6, 10e-6
    wavelength = 0.532e-6
    grid = Grid(nx, ny, dx, dy, wavelength)
    
    # Equation: Gaussian using r
    w0 = 100e-6
    equation = f"exp(-r**2 / {w0}**2)"
    
    source = CustomSource(grid, amplitude=1.0, equation=equation)
    field = source.generate(device='cpu')
    
    # Check peak
    intensity = field.get_intensity().numpy()
    # Should be exactly 1.0 at center
    assert np.isclose(np.max(intensity), 1.0, atol=1e-3)
    
    # Check width
    # At r=w0, intensity should be exp(-2) ~ 0.135
    center_idx = nx // 2
    # x array: linspace(-L/2, L/2, N)
    x = grid.X[center_idx, :]
    
    # Find index closest to w0
    idx_w0 = np.argmin(np.abs(x - w0))
    val_w0 = intensity[center_idx, idx_w0]
    
    expected = np.exp(-2) 
    
    assert np.isclose(val_w0, expected, atol=0.05)
    print("Custom source Gaussian test passed.")
    
    # Test variables
    equation_var = "A * exp(-r**2 / width**2)"
    vars_dict = {"A": 2.0, "width": w0}
    source_var = CustomSource(grid, amplitude=1.0, equation=equation_var, variables=vars_dict)
    field_var = source_var.generate(device='cpu')
    
    assert np.isclose(np.max(np.abs(field_var.to_numpy())), 2.0, atol=1e-3)
    print("Custom source variables test passed.")

def test_lens_phase():
    """
    测试透镜相位 (Test Lens Phase)
    """
    nx, ny = 129, 129 # Odd
    dx, dy = 10e-6, 10e-6
    wavelength = 0.532e-6
    grid = Grid(nx, ny, dx, dy, wavelength)
    
    f = 0.1 # 100 mm
    lens = IdealLens(grid, focal_length=f)
    
    # Create uniform field
    field = OpticalField(grid, device='cpu')
    field.set_field(np.ones((ny, nx), dtype=np.complex128))
    
    field_out = lens.modulate(field)
    phase = field_out.get_phase().numpy()
    
    # Theoretical phase: -k/(2f) * r^2
    # Center (r=0) should be 0
    k = 2 * np.pi / wavelength
    r = 100e-6 # 100 um
    
    # Theoretical phase difference
    dphi_theo = -k / (2 * f) * r**2
    
    # Numerical
    center_idx = nx // 2
    idx_r = np.argmin(np.abs(grid.X[center_idx, :] - r))
    
    phi_center = phase[center_idx, center_idx]
    phi_r = phase[center_idx, idx_r]
    
    dphi_num = phi_r - phi_center
    
    # Wrap to [-pi, pi]
    dphi_num = (dphi_num + np.pi) % (2 * np.pi) - np.pi
    dphi_theo = (dphi_theo + np.pi) % (2 * np.pi) - np.pi
    
    assert np.isclose(dphi_num, dphi_theo, atol=0.1) 
    print("Lens phase test passed.")

if __name__ == "__main__":
    test_custom_source()
    test_lens_phase()
