import sys
import os
import numpy as np
import torch
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.field import Grid, OpticalField
from src.core.source import PlaneWave, GaussianBeam
from src.core.propagator import AngularSpectrumPropagator

def test_plane_wave_propagation():
    """
    测试平面波传播 (Test Plane Wave Propagation)
    验证光强是否保持恒定 (Verify intensity remains constant)
    """
    nx, ny = 256, 256
    dx, dy = 1e-6, 1e-6
    wavelength = 0.532e-6
    
    grid = Grid(nx, ny, dx, dy, wavelength)
    source = PlaneWave(grid, amplitude=1.0)
    
    # Use CPU for testing to ensure compatibility
    field = source.generate(device='cpu')
    
    propagator = AngularSpectrumPropagator(grid)
    distance = 100e-6 # 100 um
    
    new_field = propagator.propagate(field, distance)
    
    intensity = new_field.get_intensity().cpu().numpy()
    
    # Check if intensity is uniform and close to 1.0
    # Allow some numerical error
    assert np.allclose(intensity, 1.0, atol=1e-5)
    print("Plane wave propagation test passed.")

def test_gaussian_beam_energy_conservation():
    """
    测试高斯光束能量守恒 (Test Gaussian Beam Energy Conservation)
    """
    nx, ny = 512, 512
    dx, dy = 1e-6, 1e-6
    wavelength = 0.532e-6
    
    grid = Grid(nx, ny, dx, dy, wavelength)
    source = GaussianBeam(grid, amplitude=1.0, w0=10e-6)
    
    field = source.generate(device='cpu')
    initial_energy = torch.sum(field.get_intensity()).item()
    
    propagator = AngularSpectrumPropagator(grid)
    distance = 50e-6
    
    new_field = propagator.propagate(field, distance)
    final_energy = torch.sum(new_field.get_intensity()).item()
    
    # Energy should be conserved
    assert abs(initial_energy - final_energy) / initial_energy < 1e-3
    print("Gaussian beam energy conservation test passed.")

if __name__ == "__main__":
    test_plane_wave_propagation()
    test_gaussian_beam_energy_conservation()
