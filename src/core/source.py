import numpy as np
import torch
from .field import Grid, OpticalField
from scipy.special import genlaguerre, jv

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
        x = np.linspace(-self.grid.nx/2 * self.grid.dx, self.grid.nx/2 * self.grid.dx, self.grid.nx)
        y = np.linspace(-self.grid.ny/2 * self.grid.dy, self.grid.ny/2 * self.grid.dy, self.grid.ny)
        X, Y = np.meshgrid(x, y)
        
        E = self.amplitude * np.exp(1j * (self.kx * X + self.ky * Y))
        
        field = OpticalField(self.grid, device=device)
        field.set_field(E)
        return field

class GaussianBeam(Source):
    """
    高斯光束 (Gaussian Beam)
    """
    def __init__(self, grid: Grid, amplitude: float = 1.0, w0: float = 1.0e-3, z: float = 0.0):
        super().__init__(grid, amplitude)
        self.w0 = w0
        self.z = z
        
    def generate(self, device='cpu') -> OpticalField:
        k = 2 * np.pi / self.grid.wavelength
        z_R = np.pi * self.w0**2 / self.grid.wavelength 
        
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
        self.p = p 
        self.l = l 
        
    def generate(self, device='cpu') -> OpticalField:
        x = np.linspace(-self.grid.nx/2 * self.grid.dx, self.grid.nx/2 * self.grid.dx, self.grid.nx)
        y = np.linspace(-self.grid.ny/2 * self.grid.dy, self.grid.ny/2 * self.grid.dy, self.grid.ny)
        X, Y = np.meshgrid(x, y)
        
        r = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X)
        
        term1 = (r * np.sqrt(2) / self.w0) ** np.abs(self.l)
        term2 = np.exp(-r**2 / self.w0**2)
        term3 = genlaguerre(self.p, np.abs(self.l))(2 * r**2 / self.w0**2)
        term4 = np.exp(1j * self.l * phi)
        
        E = self.amplitude * term1 * term2 * term3 * term4
        
        field = OpticalField(self.grid, device=device)
        field.set_field(E)
        return field

class CustomSource(Source):
    """
    自定义光源 (Custom Source)
    """
    def __init__(self, grid: Grid, amplitude: float = 1.0, equation: str = "1", variables: dict = None):
        super().__init__(grid, amplitude)
        self.equation = equation
        self.variables = variables if variables else {}

    def generate(self, device='cpu') -> OpticalField:
        # Generate coordinates
        x = np.linspace(-self.grid.nx/2 * self.grid.dx, self.grid.nx/2 * self.grid.dx, self.grid.nx)
        y = np.linspace(-self.grid.ny/2 * self.grid.dy, self.grid.ny/2 * self.grid.dy, self.grid.ny)
        X, Y = np.meshgrid(x, y)
        
        # Cylindrical coordinates
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)
        
        # Safe evaluation context
        context = {
            "np": np,
            "sqrt": np.lib.scimath.sqrt, # Support complex sqrt
            "exp": np.exp,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "pi": np.pi,
            "abs": np.abs,
            "arctan": np.arctan,
            "arctan2": np.arctan2,
            "power": np.power,
            "real": np.real,
            "imag": np.imag,
            "conj": np.conj,
            "angle": np.angle,
            "besselj": jv, # Bessel function of first kind
            "x": X,
            "y": Y,
            "z": 0, # Assume source is at z=0 local
            "r": R,
            "phi": PHI,
            "1j": 1j,
            "i": 1j,
            "j": 1j
        }
        
        # Add custom variables
        if self.variables:
            context.update(self.variables)
            
        try:
            # Evaluate equation
            # Use empty dict for locals/globals to sandbox slightly, but we pass context as locals
            # Note: __builtins__ must be handled carefully.
            
            with np.errstate(divide='ignore', invalid='ignore'):
                E_val = eval(self.equation, {"__builtins__": {}}, context)
            
            # Handle NaN/Inf from division by zero
            if isinstance(E_val, np.ndarray):
                E_val = np.nan_to_num(E_val, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure it's complex or float array
            if np.isscalar(E_val):
                E_val = np.full_like(X, E_val, dtype=np.complex128)
            
            E = self.amplitude * E_val
            
        except Exception as e:
            # Propagate error with detail
            # Check for common user errors
            msg = str(e)
            if "NoneType" in msg and "subscriptable" in msg:
                msg += " (Hint: Check if you are using a function that requires a list or array but got None, or if __builtins__ access is required by some library)"
            raise ValueError(f"Custom source evaluation failed: {msg}")
            
        field = OpticalField(self.grid, device=device)
        field.set_field(E)
        return field
