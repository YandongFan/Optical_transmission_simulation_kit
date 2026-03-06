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
    def __init__(self, position_z: float, name: str = "monitor"):
        self.position_z = position_z
        self.name = name
        self.field_data = None # Complex field data (numpy array)
        self.intensity_data = None # Intensity data (numpy array)
        self.grid_x = None
        self.grid_y = None
        
    def record(self, field: OpticalField):
        """
        记录当前光场数据 (Record current optical field data)
        """
        # Convert to numpy and store
        self.field_data = field.to_numpy()
        self.intensity_data = np.abs(self.field_data)**2
        self.grid_x = field.grid.X
        self.grid_y = field.grid.Y
        
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
            grp.create_dataset('x', data=self.grid_x)
            grp.create_dataset('y', data=self.grid_y)
            grp.attrs['z_position'] = self.position_z
            
    def save_mat(self, filename: str):
        """
        保存为 MAT 格式 (Save as MAT)
        """
        if self.field_data is None:
            return
            
        data = {
            'field': self.field_data,
            'intensity': self.intensity_data,
            'x': self.grid_x,
            'y': self.grid_y,
            'z_position': self.position_z
        }
        savemat(filename, data)
        
    def save_csv(self, filename: str):
        """
        保存为 CSV 格式 (Save as CSV)
        注意：CSV 对于大数组效率较低 (Note: CSV is inefficient for large arrays)
        """
        if self.field_data is None:
            return
            
        # Flatten data for CSV
        df = pd.DataFrame({
            'x': self.grid_x.flatten(),
            'y': self.grid_y.flatten(),
            'field_real': np.real(self.field_data).flatten(),
            'field_imag': np.imag(self.field_data).flatten(),
            'intensity': self.intensity_data.flatten()
        })
        df.to_csv(filename, index=False)
