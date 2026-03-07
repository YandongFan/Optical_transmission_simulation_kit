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
    def __init__(self, position_z: float, name: str = "monitor", plane_type: int = 0, fixed_value: float = 0.0, ranges: dict = None):
        """
        :param position_z: Position along propagation axis (for XY plane).
        :param plane_type: 0=XY (Normal Z), 1=YZ (Normal X), 2=XZ (Normal Y)
        :param fixed_value: The value of the fixed dimension (X for YZ, Y for XZ).
        :param ranges: Dictionary defining min/max for axes, e.g., {'x': (-10, 10), 'y': (-10, 10), 'z': (0, 100)}
        """
        self.position_z = position_z 
        self.name = name
        self.plane_type = plane_type
        self.fixed_value = fixed_value
        self.ranges = ranges if ranges else {}
        
        self.field_data = None # Complex field data (numpy array)
        self.intensity_data = None # Intensity data (numpy array)
        self.grid_x = None
        self.grid_y = None
        
        # Buffers for volumetric/slice recording
        self.z_coords = []
        self.slice_buffer = [] 
        
    def get_geometry_info(self) -> str:
        """
        获取几何参数信息 (Get geometric parameters)
        Requirements: Print origin, normal, span.
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
        # Convert um range to meters? Assuming ranges passed in meters to match internal logic?
        # User input is um. Monitor usually stores SI. Let's assume ranges are SI (meters).
        return min_v <= val <= max_v

    def _get_slice_indices(self, axis_values, axis_name):
        """
        Get indices for slicing based on range
        """
        if axis_name not in self.ranges:
            return slice(None), axis_values
            
        min_v, max_v = self.ranges[axis_name]
        # Assuming axis_values is sorted
        # Using numpy searchsorted or boolean mask
        mask = (axis_values >= min_v) & (axis_values <= max_v)
        if not np.any(mask):
            return slice(0, 0), np.array([])
            
        # Find first and last True
        # Optimization: finding indices
        indices = np.where(mask)[0]
        start, end = indices[0], indices[-1] + 1
        return slice(start, end), axis_values[start:end]

    def record(self, field: OpticalField, current_z: float):
        """
        记录当前光场数据 (Record current optical field data)
        :param field: Current optical field object
        :param current_z: Current Z position of the field
        """
        if self.plane_type == 0: # XY Plane
            # XY Plane usually records at specific Z.
            # If ranges are set for X and Y, we slice.
            
            # Get full field first (on CPU)
            full_field = field.to_numpy() # (Ny, Nx)
            
            # Get axes
            x_axis = field.grid.X[0, :] # 1D array
            y_axis = field.grid.Y[:, 0] # 1D array
            
            # Slice X
            sl_x, self.grid_x = self._get_slice_indices(x_axis, 'x')
            # Slice Y
            sl_y, self.grid_y = self._get_slice_indices(y_axis, 'y')
            
            # Apply slices
            # full_field is (Ny, Nx) -> (Y, X)
            self.field_data = full_field[sl_y, sl_x]
            self.intensity_data = np.abs(self.field_data)**2
            
        elif self.plane_type == 1: # YZ Plane (Fixed X)
            # Check Z range first
            if not self._is_in_range(current_z, 'z'):
                return

            # Slice field at X = fixed_value
            # We need the full x_axis to find the index of fixed_value
            # But we don't slice X range here (since X is fixed).
            # We slice Y range.
            
            x_axis = np.linspace(-field.grid.nx/2 * field.grid.dx, field.grid.nx/2 * field.grid.dx, field.grid.nx)
            idx = (np.abs(x_axis - self.fixed_value)).argmin()
            
            # Field E is (ny, nx). We want column idx.
            # field.E is on device.
            slice_data = field.E[:, idx].cpu().numpy() # Shape (ny,)
            
            # Slice Y
            y_axis = np.linspace(-field.grid.ny/2 * field.grid.dy, field.grid.ny/2 * field.grid.dy, field.grid.ny)
            sl_y, sliced_y_axis = self._get_slice_indices(y_axis, 'y')
            
            final_slice = slice_data[sl_y]
            
            self.slice_buffer.append(final_slice)
            self.z_coords.append(current_z)
            
            # Update grid_y only once (or overwrite, it should be same)
            self.grid_y = sliced_y_axis
            
        elif self.plane_type == 2: # XZ Plane (Fixed Y)
            # Check Z range
            if not self._is_in_range(current_z, 'z'):
                return

            # Slice field at Y = fixed_value
            y_axis = np.linspace(-field.grid.ny/2 * field.grid.dy, field.grid.ny/2 * field.grid.dy, field.grid.ny)
            idx = (np.abs(y_axis - self.fixed_value)).argmin()
            
            # Field E is (ny, nx). We want row idx.
            slice_data = field.E[idx, :].cpu().numpy() # Shape (nx,)
            
            # Slice X
            x_axis = np.linspace(-field.grid.nx/2 * field.grid.dx, field.grid.nx/2 * field.grid.dx, field.grid.nx)
            sl_x, sliced_x_axis = self._get_slice_indices(x_axis, 'x')
            
            final_slice = slice_data[sl_x]
            
            self.slice_buffer.append(final_slice)
            self.z_coords.append(current_z)
            
            self.grid_x = sliced_x_axis

    def finalize(self):
        """
        Process accumulated buffer into final data arrays
        """
        if self.plane_type in [1, 2] and self.slice_buffer:
            # Stack slices
            # slice_buffer is list of 1D arrays (N_transverse,)
            # Stacked: (Nz, N_transverse)
            # We usually want to visualize as (N_transverse, Nz) where Z is horizontal.
            
            data_stack = np.array(self.slice_buffer).T # (N_transverse, Nz)
            self.field_data = data_stack
            self.intensity_data = np.abs(self.field_data)**2
            
            # Setup coordinate grids for plotting/saving
            # self.grid_x/y already set.
            # For YZ plane: axes are Y and Z.
            # For XZ plane: axes are X and Z.
            
    def save_hdf5(self, filename: str):
        """
        保存为 HDF5 格式 (Save as HDF5)
        """
        if self.field_data is None:
            return
            
        with h5py.File(filename, 'w') as f:
            grp = f.create_group(self.name)
            grp.create_dataset('field_real', data=np.real(self.field_data))
            grp.create_dataset('field_imag', data=np.imag(self.field_data))
            grp.create_dataset('intensity', data=self.intensity_data)
            
            grp.attrs['plane_type'] = self.plane_type
            
            if self.plane_type == 0:
                grp.create_dataset('x', data=self.grid_x)
                grp.create_dataset('y', data=self.grid_y)
                grp.attrs['z_position'] = self.position_z
            elif self.plane_type == 1: # YZ
                grp.create_dataset('y', data=self.grid_y) # Vertical axis
                grp.create_dataset('z', data=np.array(self.z_coords)) # Horizontal axis
                grp.attrs['x_position'] = self.fixed_value
            elif self.plane_type == 2: # XZ
                grp.create_dataset('x', data=self.grid_x) # Vertical axis (or Horizontal?) usually X is horizontal
                grp.create_dataset('z', data=np.array(self.z_coords))
                grp.attrs['y_position'] = self.fixed_value
            
    def save_mat(self, filename: str):
        """
        保存为 MAT 格式 (Save as MAT)
        """
        if self.field_data is None:
            return
            
        data = {
            'field': self.field_data,
            'intensity': self.intensity_data,
            'plane_type': self.plane_type
        }
        
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
        保存为 CSV 格式 (Save as CSV)
        """
        if self.field_data is None:
            return
            
        # Flatten data for CSV
        # This is tricky for different shapes.
        # Just dumping flattened arrays.
        
        data_dict = {
            'field_real': np.real(self.field_data).flatten(),
            'field_imag': np.imag(self.field_data).flatten(),
            'intensity': self.intensity_data.flatten()
        }
        
        # Add coords? Too big for CSV usually.
        df = pd.DataFrame(data_dict)
        df.to_csv(filename, index=False)
