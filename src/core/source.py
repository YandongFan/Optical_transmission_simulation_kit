import numpy as np
import torch
from .field import Grid, OpticalField
from scipy.special import genlaguerre

class Source:
    """
    光源基类 (Base class for light sources)
    """
    def __init__(self, grid: Grid, amplitude: float = 1.0, phase: float = 0.0):
        self.grid = grid
        self.amplitude = amplitude
        self.phase = phase
    
    def generate(self) -> OpticalField:
        raise NotImplementedError

class PlaneWave(Source):
    """
    平面波 (Plane Wave)
    """
    def __init__(self, grid: Grid, amplitude: float = 1.0, kx: float = 0.0, ky: float = 0.0):
        super().__init__(grid, amplitude)
        self.kx = kx
        self.ky = ky
        
    def generate(self, device='cpu') -> OpticalField:
        # Generate meshgrid for spatial coordinates
        x = np.linspace(-self.grid.nx/2 * self.grid.dx, self.grid.nx/2 * self.grid.dx, self.grid.nx)
        y = np.linspace(-self.grid.ny/2 * self.grid.dy, self.grid.ny/2 * self.grid.dy, self.grid.ny)
        X, Y = np.meshgrid(x, y)
        
        # Calculate field E(x, y) = A * exp(i(kx*x + ky*y))
        E = self.amplitude * np.exp(1j * (self.kx * X + self.ky * Y))
        
        field = OpticalField(self.grid, device=device)
        field.set_field(E)
        return field

class GaussianBeam(Source):
    """
    高斯光束 (Gaussian Beam)
    """
    def __init__(self, grid: Grid, amplitude: float = 1.0, w0: float = 1.0e-3, z: float = 0.0):
        """
        :param w0: 腰斑半径 (Waist radius)
        :param z: 当前位置相对于腰斑的距离 (Distance from waist)
        """
        super().__init__(grid, amplitude)
        self.w0 = w0
        self.z = z
        
    def generate(self, device='cpu') -> OpticalField:
        k = 2 * np.pi / self.grid.wavelength
        z_R = np.pi * self.w0**2 / self.grid.wavelength # Rayleigh range
        
        if self.z == 0:
            w_z = self.w0
            R_z = np.inf
            psi_z = 0
        else:
            w_z = self.w0 * np.sqrt(1 + (self.z / z_R)**2)
            R_z = self.z * (1 + (z_R / self.z)**2)
            psi_z = np.arctan(self.z / z_R)
            
        x = np.linspace(-self.grid.nx/2 * self.grid.dx, self.grid.nx/2 * self.grid.dx, self.grid.nx)
        y = np.linspace(-self.grid.ny/2 * self.grid.dy, self.grid.ny/2 * self.grid.dy, self.grid.ny)
        X, Y = np.meshgrid(x, y)
        r2 = X**2 + Y**2
        
        # Gaussian beam formula
        # E(r, z) = A * (w0/w(z)) * exp(-r^2/w(z)^2) * exp(-i(kz + k*r^2/(2R(z)) - psi(z)))
        # Here we ignore the exp(-ikz) term as it's a global phase factor
        
        if R_z == np.inf:
            phase_curvature = 0
        else:
            phase_curvature = k * r2 / (2 * R_z)
            
        E = self.amplitude * (self.w0 / w_z) * np.exp(-r2 / w_z**2) * \
            np.exp(-1j * (phase_curvature - psi_z))
            
        field = OpticalField(self.grid, device=device)
        field.set_field(E)
        return field

class LaguerreGaussianBeam(Source):
    """
    拉盖尔-高斯光束 (Laguerre-Gaussian Beam)
    """
    def __init__(self, grid: Grid, amplitude: float = 1.0, w0: float = 1.0e-3, p: int = 0, l: int = 0):
        super().__init__(grid, amplitude)
        self.w0 = w0
        self.p = p # Radial index
        self.l = l # Azimuthal index (topological charge)
        
    def generate(self, device='cpu') -> OpticalField:
        x = np.linspace(-self.grid.nx/2 * self.grid.dx, self.grid.nx/2 * self.grid.dx, self.grid.nx)
        y = np.linspace(-self.grid.ny/2 * self.grid.dy, self.grid.ny/2 * self.grid.dy, self.grid.ny)
        X, Y = np.meshgrid(x, y)
        
        r = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X)
        
        # Simplified for z=0 (at waist)
        term1 = (r * np.sqrt(2) / self.w0) ** np.abs(self.l)
        term2 = np.exp(-r**2 / self.w0**2)
        term3 = genlaguerre(self.p, np.abs(self.l))(2 * r**2 / self.w0**2)
        term4 = np.exp(1j * self.l * phi)
        
        E = self.amplitude * term1 * term2 * term3 * term4
        
        field = OpticalField(self.grid, device=device)
        field.set_field(E)
        return field
