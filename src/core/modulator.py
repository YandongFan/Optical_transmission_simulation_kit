import torch
import numpy as np
import re
from .field import OpticalField

def evaluate_formula(formula, custom_vars, x, y, wavelength):
    """
    Evaluates a mathematical formula string using numpy.
    Supports '^' for power operation (replaced with '**').
    """
    if not formula:
        return None
        
    # Preprocess formula: Replace '^' with '**' for power operation
    # This is common in scientific notation and avoids bitwise XOR errors
    formula = formula.replace('^', '**')
    
    # 'lambda' is a reserved keyword in Python, so we must replace it
    # We use regex word boundaries to only replace 'lambda' as a whole word
    formula_mod = re.sub(r'\blambda\b', '_lambda', formula)
        
    # Context
    context = {
        'np': np,
        'x': x,
        'y': y,
        '_lambda': wavelength,
        'pi': np.pi,
        'sqrt': np.sqrt,
        'exp': np.exp,
        'sin': np.sin,
        'cos': np.cos,
        'abs': np.abs,
        'theta': np.arctan2(y, x),
        'phi': np.arctan2(y, x),
        'r': np.sqrt(x**2 + y**2)
    }
    if custom_vars:
        context.update(custom_vars)
        
    try:
        # Compile first to catch syntax errors cleanly
        code = compile(formula_mod, "<string>", "eval")
        res = eval(code, {"__builtins__": {}}, context)
        
        # Ensure result is numpy array
        if np.isscalar(res):
            res = np.full_like(x, res)
        elif not isinstance(res, np.ndarray):
            res = np.array(res)
            
        # Handle NaN and Inf
        if np.any(np.isnan(res)) or np.any(np.isinf(res)):
            res = np.nan_to_num(res, nan=0.0, posinf=1e10, neginf=-1e10)
            
        return res
    except SyntaxError as e:
        # Re-raise with line info for GUI catching
        raise Exception(f"Syntax Error at line {e.lineno}, offset {e.offset}: {e.msg}")
    except Exception as e:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        # Find the frame corresponding to the user string
        line_no = 1
        for frame in tb:
            if frame.filename == "<string>":
                line_no = frame.lineno
                break
        raise Exception(f"Runtime Error at line {line_no}: {str(e)}")

class Modulator:
    """
    调制器基类 (Modulator Base Class)
    """
    def __init__(self, grid, polarizations=None, polarization_angle=0.0):
        """
        :param polarizations: List of strings ['linear_x', 'lcp', 'rcp', 'unpolarized']
        :param polarization_angle: Angle for linear_x in degrees
        """
        self.grid = grid
        self.polarizations = polarizations if polarizations else ['unpolarized']
        self.polarization_angle = polarization_angle
        
    def _get_projection_matrix(self):
        """
        Calculate the total projection matrix based on selected polarizations
        Returns tensor shape (2, 2)
        """
        P_total = torch.zeros((2, 2), dtype=torch.complex64, device=self.grid.device if hasattr(self.grid, 'device') else 'cpu')
        
        # Helper for scalar to tensor
        def to_tensor(val):
            return torch.tensor(val, dtype=torch.complex64, device=P_total.device)

        if 'unpolarized' in self.polarizations:
            return torch.eye(2, dtype=torch.complex64, device=P_total.device)
            
        for p in self.polarizations:
            if p == 'linear_x':
                theta = np.deg2rad(self.polarization_angle)
                c = np.cos(theta)
                s = np.sin(theta)
                # P = [[c^2, cs], [cs, s^2]]
                P_lin = torch.tensor([[c**2, c*s], [c*s, s**2]], dtype=torch.complex64, device=P_total.device)
                P_total += P_lin
                
            elif p == 'lcp':
                # User def: Ex leads Ey 90 -> Ex=i, Ey=1
                # v = [i, 1] / sqrt(2)
                # P = v * v.H = 0.5 * [[1, i], [-i, 1]]
                P_lcp = torch.tensor([[0.5, 0.5j], [-0.5j, 0.5]], dtype=torch.complex64, device=P_total.device)
                P_total += P_lcp
                
            elif p == 'rcp':
                # Orthogonal to LCP: Ex=1, Ey=i -> v=[1, i]/sqrt(2)
                # P = 0.5 * [[1, -i], [i, 1]]
                P_rcp = torch.tensor([[0.5, -0.5j], [0.5j, 0.5]], dtype=torch.complex64, device=P_total.device)
                P_total += P_rcp
                
        return P_total

    def _apply_scalar_modulation(self, field: OpticalField, T_scalar: torch.Tensor) -> OpticalField:
        """
        Apply scalar modulation T to the field respecting polarization settings.
        E_out = (I + (T - 1) * P_total) * E_in
        """
        device = field.device
        
        # Ensure T_scalar is on device
        if T_scalar.device != device:
            T_scalar = T_scalar.to(device)
            
        P = self._get_projection_matrix()
        if P.device != device:
            P = P.to(device)
            
        # P is (2, 2). T_scalar is (Ny, Nx).
        # We need to broadcast.
        # E vector is (2, Ny, Nx) -> stack(Ex, Ey)
        
        Ex = field.Ex
        Ey = field.Ey
        
        # Calculate P * E
        # P00*Ex + P01*Ey
        # P10*Ex + P11*Ey
        
        Px_Ex = P[0, 0] * Ex + P[0, 1] * Ey
        Py_Ey = P[1, 0] * Ex + P[1, 1] * Ey
        
        # Modulation difference term: (T - 1)
        # Note: T_scalar might be scalar (1.0) or tensor
        delta_T = T_scalar - 1.0
        
        # E_out = E_in + delta_T * (P * E_in)
        Ex_out = Ex + delta_T * Px_Ex
        Ey_out = Ey + delta_T * Py_Ey
        
        new_field = OpticalField(self.grid, device=device)
        new_field.Ex = Ex_out
        new_field.Ey = Ey_out
        return new_field

    def modulate(self, field: OpticalField) -> OpticalField:
        raise NotImplementedError

class SpatialModulator(Modulator):
    """
    空间调制器 (Spatial Modulator)
    支持相位和振幅调制 (Supports phase and amplitude modulation)
    """
    def __init__(self, grid, amplitude_mask=None, phase_mask=None, 
                 transFormula=None, phaseFormula=None, customVars=None, **kwargs):
        super().__init__(grid, **kwargs)
        self.amplitude_mask = amplitude_mask
        self.phase_mask = phase_mask
        self.transFormula = transFormula
        self.phaseFormula = phaseFormula
        self.customVars = customVars if customVars else {}
        
    def setCustomFormula(self, type_, formula):
        if type_ == 'trans':
            self.transFormula = formula
        elif type_ == 'phase':
            self.phaseFormula = formula
        return True
        
    def getCustomValue(self, type_, x, y, lambda_, **kwargs):
        formula = self.transFormula if type_ == 'trans' else self.phaseFormula
        return evaluate_formula(formula, self.customVars, x, y, lambda_)

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
        device = field.device
        
        # Prepare modulation tensor
        T = torch.ones((self.grid.ny, self.grid.nx), dtype=torch.complex64, device=device)
        
        # Use SI units (meters) for formula evaluation to ensure dimensional consistency
        X_m = self.grid.X
        Y_m = self.grid.Y
        lam_m = self.grid.wavelength
        
        if self.amplitude_mask is None and self.transFormula:
            res = evaluate_formula(self.transFormula, self.customVars, 
                                   X_m, Y_m, lam_m)
            if res is not None:
                amp = torch.from_numpy(np.clip(res, 0, 1)).to(device)
                T = T * amp

        if self.phase_mask is None and self.phaseFormula:
            res = evaluate_formula(self.phaseFormula, self.customVars,
                                   X_m, Y_m, lam_m)
            if res is not None:
                phi = torch.from_numpy(res).to(device)
                T = T * torch.exp(1j * phi)
        
        if self.amplitude_mask is not None:
            amp = torch.from_numpy(self.amplitude_mask).to(device)
            T = T * amp
            
        if self.phase_mask is not None:
            phi = torch.from_numpy(self.phase_mask).to(device)
            T = T * torch.exp(1j * phi)
            
        return self._apply_scalar_modulation(field, T)

class AngleModulator(Modulator):
    """
    角度调制器 (Angle Modulator)
    用于模拟角度选择性透射 (Simulate angle-selective transmission)
    """
    def __init__(self, grid, angle_transmission_curve=None, **kwargs):
        """
        :param angle_transmission_curve: function T(theta) -> transmission [0, 1]
        """
        super().__init__(grid, **kwargs)
        self.angle_transmission_curve = angle_transmission_curve
        
    def modulate(self, field: OpticalField) -> OpticalField:
        if self.angle_transmission_curve is None:
            return field
            
        # Angle modulator operates in k-space.
        # Polarization logic:
        # We apply T(theta) only to affected polarizations.
        # This means we transform E to k-space, apply T(theta) to P*E_k, then transform back.
        # Linearity allows: FFT( E + (T-1)PE ) = FFT(E) + FFT((T-1)PE)
        # BUT T depends on k (theta), so it's a multiplication in k-space.
        # So we should do: E_k = FFT(E).
        # E_k_out = E_k + (T(k) - 1) * P * E_k
        # Then IFFT.
        
        # Note: P is constant in spatial/k-space (assuming polarization device is uniform).
        
        Ex = field.Ex
        Ey = field.Ey
        device = field.device
        
        Ex_fft = torch.fft.fft2(Ex)
        Ey_fft = torch.fft.fft2(Ey)
        
        # Calculate T(k)
        FX = torch.from_numpy(self.grid.FX).to(device)
        FY = torch.from_numpy(self.grid.FY).to(device)
        k = self.grid.k
        KX = 2 * np.pi * FX
        KY = 2 * np.pi * FY
        
        K_transverse = torch.sqrt(KX**2 + KY**2)
        sin_theta = K_transverse / k
        sin_theta = torch.clamp(sin_theta, 0, 1)
        theta = torch.asin(sin_theta)
        
        if callable(self.angle_transmission_curve):
            T_angle = self.angle_transmission_curve(theta)
        else:
            T_angle = torch.ones_like(theta)
            
        # Apply logic: E_k_out = E_k + (T_angle - 1) * (P * E_k)
        
        P = self._get_projection_matrix()
        if P.device != device: P = P.to(device)
        
        Px_Ex_k = P[0, 0] * Ex_fft + P[0, 1] * Ey_fft
        Py_Ey_k = P[1, 0] * Ex_fft + P[1, 1] * Ey_fft
        
        delta_T = T_angle - 1.0
        
        Ex_fft_out = Ex_fft + delta_T * Px_Ex_k
        Ey_fft_out = Ey_fft + delta_T * Py_Ey_k
        
        Ex_out = torch.fft.ifft2(Ex_fft_out)
        Ey_out = torch.fft.ifft2(Ey_fft_out)
        
        new_field = OpticalField(self.grid, device=device)
        new_field.Ex = Ex_out
        new_field.Ey = Ey_out
        return new_field

class IdealLens(Modulator):
    """
    理想薄透镜 (Ideal Thin Lens)
    """
    def __init__(self, grid, focal_length: float, **kwargs):
        super().__init__(grid, **kwargs)
        self.f = focal_length

    def modulate(self, field: OpticalField) -> OpticalField:
        device = field.device
        k = self.grid.k
        
        # Grid coordinates
        X = torch.from_numpy(self.grid.X).to(device)
        Y = torch.from_numpy(self.grid.Y).to(device)
        
        # Phase transformation: phi(x,y) = -k/(2f) * (x^2 + y^2)
        phase = -k / (2 * self.f) * (X**2 + Y**2)
        
        # Apply phase
        T = torch.exp(1j * phase)
        return self._apply_scalar_modulation(field, T)

class CylindricalLens(Modulator):
    """
    理想柱透镜 (Ideal Cylindrical Lens)
    """
    def __init__(self, grid, focal_length: float, axis: str = 'x', **kwargs):
        """
        :param axis: 'x' (focuses in x direction, constant in y) or 'y'
        """
        super().__init__(grid, **kwargs)
        self.f = focal_length
        self.axis = axis.lower()

    def modulate(self, field: OpticalField) -> OpticalField:
        device = field.device
        k = self.grid.k
        
        if self.axis == 'x':
            coord = torch.from_numpy(self.grid.X).to(device)
        else:
            coord = torch.from_numpy(self.grid.Y).to(device)
            
        phase = -k / (2 * self.f) * (coord**2)
        
        T = torch.exp(1j * phase)
        return self._apply_scalar_modulation(field, T)
