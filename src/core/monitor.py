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
    def __init__(self, position_z: float, name: str = "monitor", plane_type: int = 0, fixed_value: float = 0.0):
        """
        :param position_z: Position along propagation axis (for XY plane). Ignored/used as range end for YZ/XZ?
                           Actually for YZ/XZ, position_z might be the Z location of the monitor if it was a point?
                           But in this architecture, monitors are placed at 'events'.
                           For YZ/XZ, the 'z' in the event list usually denotes where to START or END recording?
                           Or maybe we just record EVERYTHING?
                           User request says "Monitor Plane Type... YZ Plane... Set X Position".
                           So the monitor is defined by (Type=YZ, X=fixed_value).
                           It doesn't specify Z range. Implicitly it covers the simulation Z range.
        :param plane_type: 0=XY (Normal Z), 1=YZ (Normal X), 2=XZ (Normal Y)
        :param fixed_value: The value of the fixed dimension (X for YZ, Y for XZ).
        """
        self.position_z = position_z # Keep this for compatibility, but for YZ/XZ it might be irrelevant or mean "center Z"?
        self.name = name
        self.plane_type = plane_type
        self.fixed_value = fixed_value
        
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
        if self.plane_type == 0:
            return f"Monitor '{self.name}': Type=XY (Normal Z), Z={self.position_z * 1e6:.3f} um, Normal=(0,0,1)"
        elif self.plane_type == 1:
            return f"Monitor '{self.name}': Type=YZ (Normal X), X={self.fixed_value * 1e6:.3f} um, Normal=(1,0,0)"
        elif self.plane_type == 2:
            return f"Monitor '{self.name}': Type=XZ (Normal Y), Y={self.fixed_value * 1e6:.3f} um, Normal=(0,1,0)"
        return "Unknown"

    def record(self, field: OpticalField, current_z: float):
        """
        记录当前光场数据 (Record current optical field data)
        :param field: Current optical field object
        :param current_z: Current Z position of the field
        """
        if self.plane_type == 0: # XY Plane
            # For XY plane, we typically record at a specific Z.
            # If this method is called, we assume it's the right Z.
            # But in the stepping loop, we might call it many times.
            # We should only record if current_z is close to self.position_z.
            # However, the event-based loop usually calls this ONLY when we are at the event Z.
            # So we can just overwrite.
            self.field_data = field.to_numpy()
            self.intensity_data = np.abs(self.field_data)**2
            self.grid_x = field.grid.X
            self.grid_y = field.grid.Y
            
        elif self.plane_type == 1: # YZ Plane (Fixed X)
            # Slice field at X = fixed_value
            # Grid X is (ny, nx), varying along columns. X[0, :] gives x-axis.
            # We assume grid is uniform.
            x_axis = np.linspace(-field.grid.nx/2 * field.grid.dx, field.grid.nx/2 * field.grid.dx, field.grid.nx)
            # Find closest index
            # Convert fixed_value (um usually passed from UI, but here we expect meters? 
            # Wait, Monitor stores SI units usually. Main window handles unit conversion.)
            # Assuming fixed_value is in meters.
            idx = (np.abs(x_axis - self.fixed_value)).argmin()
            
            # Field E is (ny, nx). We want column idx.
            # field.E is on device.
            slice_data = field.E[:, idx].cpu().numpy() # Shape (ny,)
            
            self.slice_buffer.append(slice_data)
            self.z_coords.append(current_z)
            
            if self.grid_y is None:
                # For YZ plane, the "horizontal" axis in the plot will be Z, "vertical" will be Y.
                # Or vice versa. Let's store Y axis.
                y_axis = np.linspace(-field.grid.ny/2 * field.grid.dy, field.grid.ny/2 * field.grid.dy, field.grid.ny)
                self.grid_y = y_axis

        elif self.plane_type == 2: # XZ Plane (Fixed Y)
            # Slice field at Y = fixed_value
            y_axis = np.linspace(-field.grid.ny/2 * field.grid.dy, field.grid.ny/2 * field.grid.dy, field.grid.ny)
            idx = (np.abs(y_axis - self.fixed_value)).argmin()
            
            # Field E is (ny, nx). We want row idx.
            slice_data = field.E[idx, :].cpu().numpy() # Shape (nx,)
            
            self.slice_buffer.append(slice_data)
            self.z_coords.append(current_z)
            
            if self.grid_x is None:
                x_axis = np.linspace(-field.grid.nx/2 * field.grid.dx, field.grid.nx/2 * field.grid.dx, field.grid.nx)
                self.grid_x = x_axis

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
