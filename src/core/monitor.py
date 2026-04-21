import torch
import numpy as np
import h5py
import pandas as pd
from scipy.io import savemat
from .field import OpticalField

class Monitor:
    """
    光场监视器 (Optical Field Monitor)
    """
    def __init__(self, position_z: float, name: str = "monitor", plane_type: int = 0, fixed_value: float = 0.0, ranges: dict = None, output_components: list = None):
        """
        :param position_z: Position along propagation axis (for XY plane).
        :param plane_type: 0=XY (Normal Z), 1=YZ (Normal X), 2=XZ (Normal Y)
        :param fixed_value: The value of the fixed dimension (X for YZ, Y for XZ).
        :param ranges: Dictionary defining min/max for axes, e.g., {'x': (-10, 10), 'y': (-10, 10), 'z': (0, 100)}
        :param output_components: List of components to record ['Ex', 'Ey', 'Ez']
        """
        self.position_z = position_z 
        self.name = name
        self.plane_type = plane_type
        self.fixed_value = fixed_value
        self.ranges = ranges if ranges else {}
        self.output_components = output_components if output_components else []
        
        self.field_data = None # Legacy main field (Ex or E_total depending on usage)
        self.component_data = {} # {'Ex': data, 'Ey': data, 'Ez': data}
        
        self.intensity_data = None # Intensity data (numpy array)
        self.grid_x = None
        self.grid_y = None
        
        # Buffers for volumetric/slice recording
        self.z_coords = []
        self.slice_buffers = {} # {'Ex': [], 'Ey': [], 'Ez': []}
        
    def get_geometry_info(self) -> str:
        """
        获取几何参数信息 (Get geometric parameters)
        """
        info = ""
        if self.plane_type == 0:
            info = f"Monitor '{self.name}': Type=XY (Normal Z), Z={self.position_z * 1e6:.3f} um, Normal=(0,0,1)"
        elif self.plane_type == 1:
            info = f"Monitor '{self.name}': Type=YZ (Normal X), X={self.fixed_value * 1e6:.3f} um, Normal=(1,0,0)"
        elif self.plane_type == 2:
            info = f"Monitor '{self.name}': Type=XZ (Normal Y), Y={self.fixed_value * 1e6:.3f} um, Normal=(0,1,0)"
        else:
            info = "Unknown"
            
        if self.ranges:
            info += f", Ranges={self.ranges}"
        return info

    def _is_in_range(self, val, axis):
        if axis not in self.ranges:
            return True
        min_v, max_v = self.ranges[axis]
        return min_v <= val <= max_v

    def _get_slice_indices(self, axis_values, axis_name):
        """
        Get indices for slicing based on range
        """
        if axis_name not in self.ranges:
            return slice(None), axis_values
            
        min_v, max_v = self.ranges[axis_name]
        mask = (axis_values >= min_v) & (axis_values <= max_v)
        if not np.any(mask):
            mid = 0.5 * (min_v + max_v)
            idx = int(np.abs(axis_values - mid).argmin())
            return slice(idx, idx + 1), axis_values[idx:idx + 1]
            
        indices = np.where(mask)[0]
        start, end = indices[0], indices[-1] + 1
        return slice(start, end), axis_values[start:end]

    def _calculate_Ez(self, field: OpticalField):
        """
        Calculate Ez component from Ex and Ey using divergence condition in k-space
        Ez = -(kx*Ex + ky*Ey) / kz
        """
        device = field.device
        Ex = field.Ex
        Ey = field.Ey
        
        Ex_fft = torch.fft.fft2(Ex)
        Ey_fft = torch.fft.fft2(Ey)
        
        FX = torch.from_numpy(field.grid.FX).to(device)
        FY = torch.from_numpy(field.grid.FY).to(device)
        k = field.grid.k
        KX = 2 * np.pi * FX
        KY = 2 * np.pi * FY
        
        KZ_sq = k**2 - KX**2 - KY**2
        # Avoid evanescent issues for Ez calc or just use same logic as Propagator
        KZ = torch.sqrt(KZ_sq.to(torch.complex64))
        
        # Avoid division by zero
        KZ_safe = torch.where(torch.abs(KZ) < 1e-6, torch.ones_like(KZ)*1e-6, KZ)
        
        Ez_fft = -(KX * Ex_fft + KY * Ey_fft) / KZ_safe
        Ez = torch.fft.ifft2(Ez_fft)
        return Ez

    def record(self, field: OpticalField, current_z: float):
        """
        记录当前光场数据 (Record current optical field data)
        :param field: Current optical field object
        :param current_z: Current Z position of the field
        """
        # 1. Identify required components
        # We always need Ex/Ey for Intensity calculation (temporarily), 
        # plus any explicitly requested components for storage.
        required_for_storage = set(self.output_components)
        required_for_calc = {'Ex', 'Ey'} 
        
        # 2. Prepare fields on device (only what's needed)
        fields_gpu = {}
        
        # Always fetch Ex/Ey for intensity calculation
        fields_gpu['Ex'] = field.Ex
        fields_gpu['Ey'] = field.Ey
            
        # Ez requires calculation
        if 'Ez' in required_for_storage:
            fields_gpu['Ez'] = self._calculate_Ez(field)
            
        if self.plane_type == 0: # XY Plane
            # Move to CPU numpy (only what we prepared)
            full_fields = {k: v.cpu().numpy() for k, v in fields_gpu.items()}
            
            x_axis = field.grid.X[0, :]
            y_axis = field.grid.Y[:, 0]
            
            sl_x, self.grid_x = self._get_slice_indices(x_axis, 'x')
            sl_y, self.grid_y = self._get_slice_indices(y_axis, 'y')
            
            # Slice and store ONLY requested components
            self.component_data = {}
            for k, v in full_fields.items():
                if k in self.output_components:
                    self.component_data[k] = v[sl_y, sl_x]
            
            # Calculate Intensity using full_fields (which contains Ex/Ey even if not stored)
            # Handle case where Ex/Ey might be missing if we optimized too much (but we ensured they are in fields_gpu)
            ex_data = full_fields.get('Ex')
            ey_data = full_fields.get('Ey')
            
            if ex_data is not None and ey_data is not None:
                # Slice for intensity
                ex_slice = ex_data[sl_y, sl_x]
                ey_slice = ey_data[sl_y, sl_x]
                self.intensity_data = np.abs(ex_slice)**2 + np.abs(ey_slice)**2
                
                # Legacy field_data support: prefer Ex, else Ey, else None
                # If Ex is not stored in component_data, we shouldn't keep it in field_data to save space?
                # But visualization might need it.
                # If user didn't request Ex, field_data should probably be None or empty, 
                # but existing code might crash. 
                # Let's set field_data only if Ex is requested, or maybe Intensity is enough for default view.
                if 'Ex' in self.output_components:
                    self.field_data = self.component_data['Ex']
                else:
                    self.field_data = None 
            
        elif self.plane_type in [1, 2]: # YZ or XZ
            if not self._is_in_range(current_z, 'z'):
                return

            # Determine fixed index and slice dimension
            if self.plane_type == 1: # Fixed X
                axis_vals = np.linspace(-field.grid.nx/2 * field.grid.dx, field.grid.nx/2 * field.grid.dx, field.grid.nx)
                fixed_idx = (np.abs(axis_vals - self.fixed_value)).argmin()
                slice_dim = 1 # Column
                
                y_axis = np.linspace(-field.grid.ny/2 * field.grid.dy, field.grid.ny/2 * field.grid.dy, field.grid.ny)
                sl_trans, self.grid_y = self._get_slice_indices(y_axis, 'y')
                
            else: # Fixed Y
                axis_vals = np.linspace(-field.grid.ny/2 * field.grid.dy, field.grid.ny/2 * field.grid.dy, field.grid.ny)
                fixed_idx = (np.abs(axis_vals - self.fixed_value)).argmin()
                slice_dim = 0 # Row
                
                x_axis = np.linspace(-field.grid.nx/2 * field.grid.dx, field.grid.nx/2 * field.grid.dx, field.grid.nx)
                sl_trans, self.grid_x = self._get_slice_indices(x_axis, 'x')

            # Extract slices
            # We track Ex/Ey for intensity calculation at finalize, plus requested ones
            comps_to_track = set(self.output_components) | {'Ex', 'Ey'}
            
            for k in comps_to_track:
                if k not in fields_gpu: continue
                
                if slice_dim == 1:
                    data = fields_gpu[k][:, fixed_idx].cpu().numpy()
                else:
                    data = fields_gpu[k][fixed_idx, :].cpu().numpy()
                    
                final_slice = data[sl_trans]
                
                if k not in self.slice_buffers:
                    self.slice_buffers[k] = []
                self.slice_buffers[k].append(final_slice)
                
            self.z_coords.append(current_z)

    def finalize(self):
        """
        Process accumulated buffer into final data arrays
        """
        if self.plane_type in [1, 2]:
            if not self.slice_buffers: return
            
            # Stack all tracked buffers
            temp_data = {}
            for k, buf in self.slice_buffers.items():
                if not buf: continue
                # Stack (N_transverse, Nz) -> (N_transverse, Nz)
                temp_data[k] = np.array(buf).T
                
            # Intensity Calculation
            Ex = temp_data.get('Ex')
            Ey = temp_data.get('Ey')
            if Ex is not None and Ey is not None:
                self.intensity_data = np.abs(Ex)**2 + np.abs(Ey)**2
            
            # Store ONLY requested components in self.component_data
            self.component_data = {}
            for k, v in temp_data.items():
                if k in self.output_components:
                    self.component_data[k] = v
            
            # Legacy field_data
            if 'Ex' in self.output_components:
                self.field_data = self.component_data['Ex']
            else:
                self.field_data = None
            
            self.slice_buffers = {}

    def save_hdf5(self, filename: str):
        """
        保存为 HDF5 格式
        """
        if self.intensity_data is None: return
            
        with h5py.File(filename, 'w') as f:
            grp = f.create_group(self.name)
            grp.create_dataset('intensity', data=self.intensity_data)
            
            # Save components
            for k, v in self.component_data.items():
                if k in self.output_components:
                    sub = grp.create_group(k)
                    sub.create_dataset('real', data=np.real(v))
                    sub.create_dataset('imag', data=np.imag(v))
            
            grp.attrs['plane_type'] = self.plane_type
            
            if self.plane_type == 0:
                grp.create_dataset('x', data=self.grid_x)
                grp.create_dataset('y', data=self.grid_y)
                grp.attrs['z_position'] = self.position_z
            elif self.plane_type == 1: # YZ
                grp.create_dataset('y', data=self.grid_y)
                grp.create_dataset('z', data=np.array(self.z_coords))
                grp.attrs['x_position'] = self.fixed_value
            elif self.plane_type == 2: # XZ
                grp.create_dataset('x', data=self.grid_x)
                grp.create_dataset('z', data=np.array(self.z_coords))
                grp.attrs['y_position'] = self.fixed_value
            
    def save_mat(self, filename: str):
        """
        保存为 MAT 格式
        """
        if self.intensity_data is None: return
            
        data = {
            'intensity': self.intensity_data,
            'plane_type': self.plane_type
        }
        
        for k, v in self.component_data.items():
            if k in self.output_components:
                data[k] = v
        
        if self.plane_type == 0:
            data['x'] = self.grid_x
            data['y'] = self.grid_y
            data['z_position'] = self.position_z
        elif self.plane_type == 1:
            data['y'] = self.grid_y
            data['z'] = np.array(self.z_coords)
            data['x_position'] = self.fixed_value
        elif self.plane_type == 2:
            data['x'] = self.grid_x
            data['z'] = np.array(self.z_coords)
            data['y_position'] = self.fixed_value
            
        savemat(filename, data)
        
    def save_csv(self, filename: str):
        """
        保存为 CSV 格式
        """
        if self.intensity_data is None: return
            
        data_dict = {
            'intensity': self.intensity_data.flatten()
        }
        
        for k, v in self.component_data.items():
            if k in self.output_components:
                data_dict[f'{k}_real'] = np.real(v).flatten()
                data_dict[f'{k}_imag'] = np.imag(v).flatten()
        
        df = pd.DataFrame(data_dict)
        df.to_csv(filename, index=False)
