
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.field import Grid, OpticalField
from src.core.source import PlaneWave
from src.core.modulator import SpatialModulator, IdealLens
from src.core.monitor import Monitor
from src.core.propagator import AngularSpectrumPropagator

def run_simulation(pol_type, pol_angle, mod_pols, monitor_plane):
    """
    Run a simulation with specific parameters
    """
    grid = Grid(nx=256, ny=256, dx=1e-6, dy=1e-6, wavelength=0.532e-6)
    
    # Source
    src = PlaneWave(grid, polarization_type=pol_type, linear_angle=pol_angle)
    field = src.generate()
    
    # Modulator: Lens that affects specific polarization
    # Focal length 10mm
    mod = IdealLens(grid, focal_length=10e-3, polarizations=mod_pols)
    field = mod.modulate(field)
    
    # Propagate to focus (z=10mm)
    prop = AngularSpectrumPropagator(grid)
    field = prop.propagate(field, 10e-3)
    
    # Monitor
    # Record at focus
    mon = Monitor(position_z=10e-3, plane_type=monitor_plane, fixed_value=0, output_components=['Ex', 'Ey'])
    mon.record(field, 10e-3) # For XY
    # For YZ/XZ we need range, but here we just record one slice if XY
    
    return mon

def main():
    pdf_filename = "polarization_test_report.pdf"
    print(f"Generating report: {pdf_filename}")
    
    with PdfPages(pdf_filename) as pdf:
        # Test Cases
        pol_cases = [
            (0, 0, "Linear X"),
            (0, 90, "Linear Y"),
            (1, 0, "LCP"),
            (2, 0, "RCP")
        ]
        
        mod_cases = [
            (['unpolarized'], "Lens (All)"),
            (['linear_x'], "Lens (X-only)")
        ]
        
        for pol_type, angle, pol_name in pol_cases:
            for mod_pols, mod_name in mod_cases:
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                fig.suptitle(f"Source: {pol_name}, Modulator: {mod_name}")
                
                # Run Simulation
                mon = run_simulation(pol_type, angle, mod_pols, 0) # XY plane
                
                # Plot Ex
                Ex = mon.component_data['Ex']
                I_x = np.abs(Ex)**2
                axes[0, 0].imshow(I_x, cmap='viridis')
                axes[0, 0].set_title("|Ex|^2")
                axes[0, 1].imshow(np.angle(Ex), cmap='hsv')
                axes[0, 1].set_title("Arg(Ex)")
                
                # Plot Ey
                Ey = mon.component_data['Ey']
                I_y = np.abs(Ey)**2
                axes[1, 0].imshow(I_y, cmap='viridis')
                axes[1, 0].set_title("|Ey|^2")
                axes[1, 1].imshow(np.angle(Ey), cmap='hsv')
                axes[1, 1].set_title("Arg(Ey)")
                
                # Plot Total Intensity
                I_total = mon.intensity_data
                axes[0, 2].imshow(I_total, cmap='inferno')
                axes[0, 2].set_title("Total Intensity")
                
                axes[1, 2].axis('off')
                
                pdf.savefig(fig)
                plt.close(fig)
                print(f"Processed {pol_name} + {mod_name}")

if __name__ == "__main__":
    try:
        main()
        print("Integration test completed successfully.")
    except Exception as e:
        print(f"Integration test failed: {e}")
