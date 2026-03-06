import torch
import numpy as np
from .field import OpticalField

class Propagator:
    """
    传播器基类 (Propagator Base Class)
    """
    def __init__(self, grid):
        self.grid = grid

    def propagate(self, field: OpticalField, distance: float) -> OpticalField:
        raise NotImplementedError

class AngularSpectrumPropagator(Propagator):
    """
    角谱传播法 (Angular Spectrum Method)
    """
    def __init__(self, grid):
        super().__init__(grid)
        self._precompute_transfer_function_params()

    def _precompute_transfer_function_params(self):
        """
        预计算传递函数所需的参数 (Precompute parameters for transfer function)
        """
        # FX, FY are already in grid
        pass

    def propagate(self, field: OpticalField, distance: float) -> OpticalField:
        """
        传播光场 (Propagate optical field)
        :param field: 输入光场 (Input optical field)
        :param distance: 传播距离 z (Propagation distance) [m]
        :return: 传播后的光场 (Propagated optical field)
        """
        E = field.E
        device = field.device
        
        # Move grid frequencies to device
        FX = torch.from_numpy(self.grid.FX).to(device)
        FY = torch.from_numpy(self.grid.FY).to(device)
        
        # Calculate wave vector components
        k = self.grid.k
        KX = 2 * np.pi * FX
        KY = 2 * np.pi * FY
        
        # Calculate KZ^2 = k^2 - KX^2 - KY^2
        KZ_sq = k**2 - KX**2 - KY**2
        
        # Calculate KZ with evanescent wave handling
        # Cast to complex for sqrt of negative numbers
        KZ = torch.sqrt(KZ_sq.to(torch.complex64))
        
        # Transfer function H = exp(i * kz * z)
        # Note: torch.exp handles complex exponent correctly
        H = torch.exp(1j * KZ * distance)
        
        # 1. FFT
        E_fft = torch.fft.fft2(E)
        
        # 2. Multiply by Transfer Function
        E_new_fft = E_fft * H
        
        # 3. IFFT
        E_new = torch.fft.ifft2(E_new_fft)
        
        # Create new field object
        new_field = OpticalField(self.grid, device=device)
        new_field.set_field(E_new)
        
        return new_field
