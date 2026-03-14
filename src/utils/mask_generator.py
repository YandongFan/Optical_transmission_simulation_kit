
from matplotlib.path import Path
import numpy as np

def generate_polygon_mask(x_grid, y_grid, vertices, transmission=1.0):
    """
    Generates a polygon mask.
    :param x_grid: 2D meshgrid X (um)
    :param y_grid: 2D meshgrid Y (um)
    :param vertices: List of (x, y) tuples defining the polygon vertices
    :param transmission: Transmission value (0-1) inside the polygon
    :return: Mask array (0 or transmission)
    """
    if not vertices or len(vertices) < 3:
        return np.zeros_like(x_grid)
        
    path = Path(vertices)
    
    # Flatten grid for point containment check
    points = np.column_stack((x_grid.flatten(), y_grid.flatten()))
    
    # Check containment
    mask_flat = path.contains_points(points)
    mask = mask_flat.reshape(x_grid.shape).astype(float)
    
    mask[mask > 0] = transmission
    return mask

def generate_annular_mask(x_grid, y_grid, center_x, center_y, r_in, r_out, transmission=1.0):
    """
    Generates an annular mask.
    :param x_grid: 2D meshgrid X (um)
    :param y_grid: 2D meshgrid Y (um)
    :param center_x: Center X (um)
    :param center_y: Center Y (um)
    :param r_in: Inner Radius (um)
    :param r_out: Outer Radius (um)
    :param transmission: Transmission value (0-1) inside the annulus
    :return: Mask array (0 or transmission)
    """
    # Calculate distance from center
    r_sq = (x_grid - center_x)**2 + (y_grid - center_y)**2
    r = np.sqrt(r_sq)
    
    # Create mask
    mask = np.zeros_like(r)
    
    # Apply annulus logic: r_in <= r <= r_out
    # We use exact comparison.
    mask[(r >= r_in) & (r <= r_out)] = transmission
    
    return mask

def generate_circular_mask(x_grid, y_grid, center_x, center_y, radius, transmission=1.0):
    """
    Generates a circular mask.
    """
    r_sq = (x_grid - center_x)**2 + (y_grid - center_y)**2
    r = np.sqrt(r_sq)
    
    mask = np.zeros_like(r)
    mask[r <= radius] = transmission
    return mask

def generate_rectangular_mask(x_grid, y_grid, center_x, center_y, width, height, rotation=0.0, transmission=1.0):
    """
    Generates a rectangular mask with rotation.
    :param rotation: Rotation in degrees
    """
    # Shift to center
    X = x_grid - center_x
    Y = y_grid - center_y
    
    # Rotate coordinates
    theta = np.deg2rad(rotation)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    
    mask = np.zeros_like(x_grid)
    mask[(np.abs(X_rot) <= width/2) & (np.abs(Y_rot) <= height/2)] = transmission
    return mask
