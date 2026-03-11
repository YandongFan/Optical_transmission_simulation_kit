import numpy as np
import torch
from .field import Grid, OpticalField
from scipy.special import genlaguerre, jv

class Source:
    """
    光源基类 (Base class for light sources)
    """
    def __init__(self, grid: Grid, amplitude: float = 1.0, phase: float = 0.0,
                 polarization_type: int = 0, linear_angle: float = 0.0, phase_offset: float = 0.0):
        """
        :param polarization_type: 0=Linear, 1=LCP, 2=RCP, 3=Unpolarized
        :param linear_angle: Angle in degrees for linear polarization
        :param phase_offset: Additional phase offset (not typically used for standard types but available)
        """
        self.grid = grid
        self.amplitude = amplitude
        self.phase = phase
        self.polarization_type = polarization_type
        self.linear_angle = linear_angle
        self.phase_offset = phase_offset
    
    def _apply_polarization(self, E_scalar):
        """
        Apply polarization state to scalar field to produce (Ex, Ey)
        :param E_scalar: complex scalar field (numpy array)
        :return: (Ex, Ey) tuple of complex numpy arrays
        """
        # Calculate Jones Vector components
        Jx = 0.0
        Jy = 0.0
        
        if self.polarization_type == 0: # Linear
            theta_rad = np.deg2rad(self.linear_angle)
            Jx = np.cos(theta_rad)
            Jy = np.sin(theta_rad)
            
        elif self.polarization_type == 1: # LCP (Ex leads Ey by 90 deg -> Ey = -i Ex? No, typically LCP: Ex=1, Ey=i)
            # User def: "LCP: Ex leads Ey 90 deg" => phase(Ex) - phase(Ey) = pi/2
            # Let Ey = 1, Ex = exp(i*pi/2) = i
            # Normalized: Ex = i/sqrt(2), Ey = 1/sqrt(2)
            # Standard optics (born & wolf): LCP -> Ex=1, Ey=i (Ey leads Ex by 90?? No, usually left/right depends on observer)
            # User explicitly defined: "LCP: Ex leads Ey 90 deg"
            # So: Ex = i, Ey = 1 (ignoring normalization for a moment)
            # Normalized: 
            Jx = 1j / np.sqrt(2)
            Jy = 1.0 / np.sqrt(2)
            
        elif self.polarization_type == 2: # RCP (Ey leads Ex 90 deg)
            # Ey = i, Ex = 1
            Jx = 1.0 / np.sqrt(2)
            Jy = 1j / np.sqrt(2)
            
        elif self.polarization_type == 3: # Unpolarized (Random Monte Carlo)
            # Generate random linear polarization state for this instance
            # Or random elliptical?
            # "Random Polarization" usually means random angle linear or random point on Poincare sphere.
            # Let's use random point on Poincare sphere for generality, or just random angle linear?
            # "Monte Carlo" implies we sample the space of polarizations.
            # Simplest: Random linear angle + Random phase diff?
            # Let's pick a random unitary Jones vector.
            # alpha = random [0, pi/2], delta = random [0, 2pi]
            # Jx = cos(alpha), Jy = sin(alpha) * exp(i*delta)
            
            # For simplicity and coverage:
            alpha = np.random.uniform(0, np.pi/2)
            delta = np.random.uniform(0, 2*np.pi)
            Jx = np.cos(alpha)
            Jy = np.sin(alpha) * np.exp(1j * delta)
            
        # Apply
        Ex = E_scalar * Jx
        Ey = E_scalar * Jy
        return Ex, Ey

    def generate(self) -> OpticalField:
        raise NotImplementedError

class PlaneWave(Source):
    """
    平面波 (Plane Wave)
    """
    def __init__(self, grid: Grid, amplitude: float = 1.0, kx: float = 0.0, ky: float = 0.0, **kwargs):
        super().__init__(grid, amplitude, **kwargs)
        self.kx = kx
        self.ky = ky
        
    def generate(self, device='cpu') -> OpticalField:
        x = np.linspace(-self.grid.nx/2 * self.grid.dx, self.grid.nx/2 * self.grid.dx, self.grid.nx)
        y = np.linspace(-self.grid.ny/2 * self.grid.dy, self.grid.ny/2 * self.grid.dy, self.grid.ny)
        X, Y = np.meshgrid(x, y)
        
        E_scalar = self.amplitude * np.exp(1j * (self.kx * X + self.ky * Y))
        Ex, Ey = self._apply_polarization(E_scalar)
        
        field = OpticalField(self.grid, device=device)
        field.Ex = torch.from_numpy(Ex).to(device)
        field.Ey = torch.from_numpy(Ey).to(device)
        return field

class GaussianBeam(Source):
    """
    高斯光束 (Gaussian Beam)
    """
    def __init__(self, grid: Grid, amplitude: float = 1.0, w0: float = 1.0e-3, z: float = 0.0, **kwargs):
        super().__init__(grid, amplitude, **kwargs)
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
            
        E_scalar = self.amplitude * (self.w0 / w_z) * np.exp(-r2 / w_z**2) * \
            np.exp(-1j * (phase_curvature - psi_z))
            
        Ex, Ey = self._apply_polarization(E_scalar)
        
        field = OpticalField(self.grid, device=device)
        field.Ex = torch.from_numpy(Ex).to(device)
        field.Ey = torch.from_numpy(Ey).to(device)
        return field

class LaguerreGaussianBeam(Source):
    """
    拉盖尔-高斯光束 (Laguerre-Gaussian Beam)
    """
    def __init__(self, grid: Grid, amplitude: float = 1.0, w0: float = 1.0e-3, p: int = 0, l: int = 0, **kwargs):
        super().__init__(grid, amplitude, **kwargs)
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
        
        E_scalar = self.amplitude * term1 * term2 * term3 * term4
        
        Ex, Ey = self._apply_polarization(E_scalar)
        
        field = OpticalField(self.grid, device=device)
        field.Ex = torch.from_numpy(Ex).to(device)
        field.Ey = torch.from_numpy(Ey).to(device)
        return field

class CustomSource(Source):
    """
    自定义光源 (Custom Source)
    """
    def __init__(self, grid: Grid, amplitude: float = 1.0, equation: str = "1", variables: dict = None, **kwargs):
        super().__init__(grid, amplitude, **kwargs)
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
            
            E_scalar = self.amplitude * E_val
            
        except Exception as e:
            # Propagate error with detail
            # Check for common user errors
            msg = str(e)
            if "NoneType" in msg and "subscriptable" in msg:
                msg += " (Hint: Check if you are using a function that requires a list or array but got None, or if __builtins__ access is required by some library)"
            raise ValueError(f"Custom source evaluation failed: {msg}")
            
        Ex, Ey = self._apply_polarization(E_scalar)
        
        field = OpticalField(self.grid, device=device)
        field.Ex = torch.from_numpy(Ex).to(device)
        field.Ey = torch.from_numpy(Ey).to(device)
        return field
