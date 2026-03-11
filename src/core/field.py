import numpy as np
import torch

class Grid:
    """
    定义仿真网格参数 (Defines simulation grid parameters)
    """
    def __init__(self, nx: int, ny: int, dx: float, dy: float, wavelength: float):
        """
        初始化网格 (Initialize grid)
        :param nx: x方向网格数 (Number of grid points in x)
        :param ny: y方向网格数 (Number of grid points in y)
        :param dx: x方向网格间距 (Grid spacing in x) [m]
        :param dy: y方向网格间距 (Grid spacing in y) [m]
        :param wavelength: 波长 (Wavelength) [m]
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
        
        # 生成坐标网格 (Generate coordinate grid)
        x = np.linspace(-nx/2 * dx, nx/2 * dx, nx)
        y = np.linspace(-ny/2 * dy, ny/2 * dy, ny)
        self.X, self.Y = np.meshgrid(x, y)

        # 频率坐标 (Frequency coordinates for FFT)
        self.dfx = 1.0 / (self.nx * self.dx)
        self.dfy = 1.0 / (self.ny * self.dy)
        
        # 预计算频率网格 (Pre-calculate frequency grid)
        fx = np.fft.fftfreq(self.nx, d=self.dx)
        fy = np.fft.fftfreq(self.ny, d=self.dy)
        self.FX, self.FY = np.meshgrid(fx, fy)

class OpticalField:
    """
    定义光场 (Defines optical field)
    """
    def __init__(self, grid: Grid, device: str = 'cpu'):
        """
        初始化光场 (Initialize optical field)
        :param grid: 网格对象 (Grid object)
        :param device: 计算设备 'cpu' 或 'cuda' (Computation device)
        """
        self.grid = grid
        self.device = device
        
        # 复振幅分布 (Complex amplitude distribution) E(x, y)
        # 初始化为零场 (Initialize as zero field)
        # 默认使用 complex64 以节省显存 (Default to complex64 to save VRAM)
        # Vector field support: Ex, Ey
        self.Ex = torch.zeros((grid.ny, grid.nx), dtype=torch.complex64, device=device)
        self.Ey = torch.zeros((grid.ny, grid.nx), dtype=torch.complex64, device=device)

    @property
    def E(self):
        """
        Backward compatibility: return Ex (Scalar approximation usually assumes Linear X)
        """
        return self.Ex

    @E.setter
    def E(self, value):
        """
        Backward compatibility: set Ex
        """
        self.Ex = value
    
    def set_field(self, field_data, component='Ex'):
        """
        设置光场分布 (Set field distribution)
        :param field_data: numpy array or torch tensor
        :param component: 'Ex' or 'Ey'
        """
        data = None
        if isinstance(field_data, np.ndarray):
            data = torch.from_numpy(field_data).to(self.device)
        elif isinstance(field_data, torch.Tensor):
            data = field_data.to(self.device)
        else:
            raise ValueError("Unsupported data type")
            
        if component == 'Ex':
            self.Ex = data
        elif component == 'Ey':
            self.Ey = data
        else:
            raise ValueError(f"Unknown component: {component}")

    def normalize(self):
        """
        归一化电场，使最大幅值为1 (Normalize field so max amplitude is 1)
        Based on total intensity |Ex|^2 + |Ey|^2
        """
        intensity = self.get_intensity()
        max_val = torch.sqrt(torch.max(intensity))
        if max_val > 0:
            self.Ex = self.Ex / max_val
            self.Ey = self.Ey / max_val

    def get_intensity(self):
        """
        获取光强分布 (Get intensity distribution) |Ex|^2 + |Ey|^2
        :return: torch tensor (on device)
        """
        return torch.abs(self.Ex)**2 + torch.abs(self.Ey)**2
    
    def get_phase(self, component='Ex'):
        """
        获取相位分布 (Get phase distribution)
        :param component: 'Ex' or 'Ey'
        :return: torch tensor (on device)
        """
        if component == 'Ex':
            return torch.angle(self.Ex)
        elif component == 'Ey':
            return torch.angle(self.Ey)
        return torch.angle(self.Ex) # Default

    def to_numpy(self, component='Ex'):
        """
        转换为 numpy 数组 (Convert to numpy array)
        :param component: 'Ex' or 'Ey'
        :return: complex numpy array
        """
        if component == 'Ex':
            return self.Ex.cpu().numpy()
        elif component == 'Ey':
            return self.Ey.cpu().numpy()
        return self.Ex.cpu().numpy()
