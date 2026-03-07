import sys
import os
import numpy as np
import torch
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.field import Grid, OpticalField
from src.core.source import PlaneWave, GaussianBeam, LaguerreGaussianBeam
from src.core.propagator import AngularSpectrumPropagator

def test_lg_beam_waist_decoupling():
    """
    测试拉盖尔-高斯光束束腰与网格解耦 (Test LG beam waist decoupling from grid)
    固定 w0, 改变网格大小，验证光斑直径不变
    """
    w0_list = [0.5e-3, 1.0e-3, 2.0e-3]
    grid_sizes = [64, 128, 256, 512, 1024] # Power of 2
    
    # Use a large enough physical domain to avoid clipping
    L = 10e-3 # 10 mm
    wavelength = 0.532e-6
    
    for w0 in w0_list:
        diameters = []
        for N in grid_sizes:
            dx = L / N
            dy = L / N
            grid = Grid(N, N, dx, dy, wavelength)
            
            # Create LG beam (p=0, l=1)
            source = LaguerreGaussianBeam(grid, amplitude=1.0, w0=w0, p=0, l=1)
            field = source.generate(device='cpu')
            intensity = field.get_intensity().numpy()
            
            # Calculate beam diameter (e.g., 4-sigma or based on second moment)
            # For LG01 (donut), let's use the distance between peak intensities
            # Or simply integral width. Let's use 2nd moment width D4sigma
            
            x = grid.X
            y = grid.Y
            I_sum = np.sum(intensity)
            x_mean = np.sum(x * intensity) / I_sum
            y_mean = np.sum(y * intensity) / I_sum
            
            x2_mean = np.sum((x - x_mean)**2 * intensity) / I_sum
            y2_mean = np.sum((y - y_mean)**2 * intensity) / I_sum
            
            # D4sigma = 4 * sqrt(sigma^2)
            d_x = 4 * np.sqrt(x2_mean)
            d_y = 4 * np.sqrt(y2_mean)
            diameter = (d_x + d_y) / 2
            
            diameters.append(diameter)
            
        # Verify variation is small
        diameters = np.array(diameters)
        mean_d = np.mean(diameters)
        max_diff = np.max(np.abs(diameters - mean_d))
        relative_error = max_diff / mean_d
        
        print(f"w0={w0*1e3}mm: Mean D={mean_d*1e3:.4f}mm, Max Diff={max_diff*1e3:.4f}mm, Rel Err={relative_error:.5f}")
        
        assert relative_error < 0.01, f"Beam diameter varies too much with grid size for w0={w0}"

def test_normalization():
    """
    测试光源归一化功能 (Test source normalization)
    """
    nx, ny = 256, 256
    dx, dy = 10e-6, 10e-6 # 10um
    wavelength = 0.532e-6
    grid = Grid(nx, ny, dx, dy, wavelength)
    
    # 1. Without Normalization
    amp = 10.0
    source = LaguerreGaussianBeam(grid, amplitude=amp, w0=0.5e-3, p=0, l=1)
    field = source.generate(device='cpu')
    max_val_orig = torch.max(torch.abs(field.E)).item()
    
    # LG01 peak is not exactly amplitude A.
    # But it should be proportional to A.
    # Let's just check it's not 1.0 (unless by coincidence)
    assert abs(max_val_orig - 1.0) > 1e-3, "Original field should not be normalized by default"
    
    # 2. With Normalization
    field.normalize()
    max_val_norm = torch.max(torch.abs(field.E)).item()
    
    assert abs(max_val_norm - 1.0) < 1e-6, "Normalized field max should be 1.0"
    
    # 3. Verify Propagation Consistency (Peak remains related if lossless?)
    # Actually, diffraction changes peak intensity.
    # We just check the input to propagator is normalized.
    # Let's check if the shape is preserved.
    
    # Check ratio at a specific point (e.g., peak location)
    # Find peak index
    idx = torch.argmax(torch.abs(field.E))
    # Normalized field at peak is 1.0 (complex phase exists)
    # Original field at peak
    
    # Re-generate original to compare shape
    field_orig = source.generate(device='cpu')
    ratio_map = torch.abs(field_orig.E) / (torch.abs(field.E) + 1e-10)
    
    # Mask out zeros to avoid division by zero issues
    mask = torch.abs(field.E) > 1e-3
    ratios = ratio_map[mask]
    
    # Ratios should be constant (equal to max_val_orig)
    std_ratio = torch.std(ratios).item()
    assert std_ratio < 1e-5, "Normalization should scale field uniformly"

if __name__ == "__main__":
    test_lg_beam_waist_decoupling()
    test_normalization()
