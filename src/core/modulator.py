import torch
import numpy as np
from .field import OpticalField

class Modulator:
    """
    调制器基类 (Modulator Base Class)
    """
    def __init__(self, grid):
        self.grid = grid
        
    def modulate(self, field: OpticalField) -> OpticalField:
        raise NotImplementedError

class SpatialModulator(Modulator):
    """
    空间调制器 (Spatial Modulator)
    支持相位和振幅调制 (Supports phase and amplitude modulation)
    """
    def __init__(self, grid, amplitude_mask=None, phase_mask=None):
        super().__init__(grid)
        self.amplitude_mask = amplitude_mask
        self.phase_mask = phase_mask
        
    def set_amplitude(self, mask):
        """
        设置振幅掩膜 (Set amplitude mask)
        :param mask: numpy array (Ny, Nx) range [0, 1]
        """
        self.amplitude_mask = mask
        
    def set_phase(self, mask):
        """
        设置相位掩膜 (Set phase mask)
        :param mask: numpy array (Ny, Nx) range [0, 2pi]
        """
        self.phase_mask = mask

    def modulate(self, field: OpticalField) -> OpticalField:
        """
        应用调制 (Apply modulation)
        """
        E = field.E
        device = field.device
        
        # Prepare modulation tensor
        T = torch.ones_like(E)
        
        if self.amplitude_mask is not None:
            amp = torch.from_numpy(self.amplitude_mask).to(device)
            T = T * amp
            
        if self.phase_mask is not None:
            phi = torch.from_numpy(self.phase_mask).to(device)
            T = T * torch.exp(1j * phi)
            
        # Apply modulation
        E_new = E * T
        
        new_field = OpticalField(self.grid, device=device)
        new_field.set_field(E_new)
        return new_field

class AngleModulator(Modulator):
    """
    角度调制器 (Angle Modulator)
    用于模拟角度选择性透射 (Simulate angle-selective transmission)
    """
    def __init__(self, grid, angle_transmission_curve=None):
        """
        :param angle_transmission_curve: function T(theta) -> transmission [0, 1]
        """
        super().__init__(grid)
        self.angle_transmission_curve = angle_transmission_curve
        
    def modulate(self, field: OpticalField) -> OpticalField:
        if self.angle_transmission_curve is None:
            return field
            
        E = field.E
        device = field.device
        
        # Convert to frequency domain to access angles
        E_fft = torch.fft.fft2(E)
        
        # Calculate angles for each frequency component
        # kx = k * sin(theta_x)
        # ky = k * sin(theta_y)
        # sin(theta) = sqrt(kx^2 + ky^2) / k
        
        FX = torch.from_numpy(self.grid.FX).to(device)
        FY = torch.from_numpy(self.grid.FY).to(device)
        k = self.grid.k
        KX = 2 * np.pi * FX
        KY = 2 * np.pi * FY
        
        K_transverse = torch.sqrt(KX**2 + KY**2)
        sin_theta = K_transverse / k
        
        # Clip sin_theta to [0, 1] to avoid domain errors (evanescent waves have sin_theta > 1)
        sin_theta = torch.clamp(sin_theta, 0, 1)
        theta = torch.asin(sin_theta) # Radians
        
        # Apply transmission curve T(theta)
        # We assume angle_transmission_curve takes tensor in radians and returns tensor
        # For now, let's assume it's a simple lookup or function. 
        # Since the user might provide a curve (data points), we might need interpolation.
        # Here we assume it's a callable for simplicity, or we implement interpolation.
        
        if callable(self.angle_transmission_curve):
            T_angle = self.angle_transmission_curve(theta)
        else:
            # Fallback or interpolation implementation needed if it's data
            T_angle = torch.ones_like(theta)
            
        # Apply to FFT
        E_new_fft = E_fft * T_angle
        
        # IFFT back
        E_new = torch.fft.ifft2(E_new_fft)
        
        new_field = OpticalField(self.grid, device=device)
        new_field.set_field(E_new)
        return new_field
