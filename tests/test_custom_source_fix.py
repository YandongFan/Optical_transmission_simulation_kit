import sys
import os
import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.field import Grid, OpticalField
from src.core.source import CustomSource

def test_complex_equation():
    """
    测试复杂方程 (Test complex equation: sqrt(-1))
    """
    nx, ny = 64, 64
    grid = Grid(nx, ny, 1e-6, 1e-6, 0.532e-6)
    
    # Equation with sqrt(-1) -> 1j
    equation = "sqrt(-1) * 10"
    source = CustomSource(grid, amplitude=1.0, equation=equation)
    field = source.generate(device='cpu')
    
    # Should be 10j
    expected = 10j
    result = field.to_numpy()
    assert np.allclose(result, expected), "sqrt(-1) failed"

def test_division_by_zero():
    """
    测试除零 (Test division by zero: arctan(y/x))
    """
    nx, ny = 65, 65 # Include 0
    grid = Grid(nx, ny, 1e-6, 1e-6, 0.532e-6)
    
    # arctan(y/x)
    equation = "arctan(y/x)"
    source = CustomSource(grid, amplitude=1.0, equation=equation)
    field = source.generate(device='cpu')
    
    # Check center (x=0, y=0)
    # numpy divides by zero gives inf/nan, handled by nan_to_num -> 0
    center_idx = nx // 2
    val = field.to_numpy()[center_idx, center_idx]
    assert val == 0.0, "Division by zero handling failed"

def test_variables():
    """
    测试变量传递 (Test variables)
    """
    nx, ny = 64, 64
    grid = Grid(nx, ny, 1e-6, 1e-6, 0.532e-6)
    
    equation = "m * x"
    vars_dict = {"m": 5.0}
    source = CustomSource(grid, amplitude=1.0, equation=equation, variables=vars_dict)
    field = source.generate(device='cpu')
    
    x_val = grid.X[0, 0]
    expected = 5.0 * x_val
    result = field.to_numpy()[0, 0]
    assert np.isclose(result, expected), "Variable 'm' not passed correctly"

def test_user_equation_preview_crash():
    """
    测试用户报告的崩溃方程 (Test user reported crash equation)
    exp(sqrt(-1)*m*arctan(y/x))
    """
    nx, ny = 65, 65
    grid = Grid(nx, ny, 1e-6, 1e-6, 0.532e-6)
    
    equation = "exp(sqrt(-1)*m*arctan(y/x))"
    vars_dict = {"m": 10.0}
    
    source = CustomSource(grid, amplitude=1.0, equation=equation, variables=vars_dict)
    
    # This should not raise exception now
    try:
        field = source.generate(device='cpu')
        assert field is not None
        print("User equation passed without crash.")
    except Exception as e:
        pytest.fail(f"User equation crashed: {e}")

if __name__ == "__main__":
    test_complex_equation()
    test_division_by_zero()
    test_variables()
    test_user_equation_preview_crash()
