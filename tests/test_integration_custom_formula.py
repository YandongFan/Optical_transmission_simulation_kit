
import pytest
import numpy as np
import torch
from src.core.field import Grid, OpticalField
from src.core.modulator import SpatialModulator

def test_custom_formula_mod1():
    # Setup
    nx, ny = 128, 128
    dx, dy = 1.0, 1.0 # um
    wavelength = 0.7 # um (700nm)
    
    grid = Grid(nx, ny, dx * 1e-6, dy * 1e-6, wavelength * 1e-6)
    field = OpticalField(grid, device='cpu')
    field.Ex = torch.ones((ny, nx), dtype=torch.complex64)
    field.Ey = torch.zeros((ny, nx), dtype=torch.complex64)
    
    # Formula (using microns)
    # We want Trans ~ 1.0. lambda is 0.7.
    # Formula: sin(x/10) * (lambda / 0.7)
    trans_formula = "np.sin(x/10) * (lambda / 0.7)" 
    phase_formula = "x**2 + y**2"
    
    # Expected calculations (using microns)
    X_um = grid.X * 1e6
    Y_um = grid.Y * 1e6
    lam_um = wavelength
    
    expected_trans = np.sin(X_um/10) * (lam_um/0.7)
    expected_trans = np.clip(expected_trans, 0, 1)
    
    expected_phase = X_um**2 + Y_um**2
    
    mod = SpatialModulator(grid, 
                           transFormula=trans_formula, 
                           phaseFormula=phase_formula)
    
    out_field = mod.modulate(field)
    
    # Verify
    res_complex = out_field.Ex.numpy()
    expected_complex = expected_trans * np.exp(1j * expected_phase)
    
    assert np.allclose(res_complex, expected_complex, atol=1e-4)

if __name__ == "__main__":
    test_custom_formula_mod1()
