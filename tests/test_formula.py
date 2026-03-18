
import pytest
import numpy as np
from src.core.modulator import evaluate_formula

def test_evaluate_formula_basic():
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    wavelength = 0.5
    
    # Constant
    res = evaluate_formula("0.5", {}, x, y, wavelength)
    assert np.allclose(res, 0.5)
    
    # Variable x
    res = evaluate_formula("x * 2", {}, x, y, wavelength)
    assert np.allclose(res, x * 2)
    
    # Variable y
    res = evaluate_formula("y + 1", {}, x, y, wavelength)
    assert np.allclose(res, y + 1)
    
    # Lambda
    res = evaluate_formula("lambda * 2", {}, x, y, wavelength)
    assert np.allclose(res, 1.0)

def test_evaluate_formula_numpy():
    x = np.array([0, np.pi/2, np.pi])
    y = np.zeros_like(x)
    wavelength = 1.0
    
    res = evaluate_formula("np.sin(x)", {}, x, y, wavelength)
    expected = np.sin(x)
    assert np.allclose(res, expected)

def test_evaluate_formula_custom_vars():
    x = np.ones(5)
    y = np.ones(5)
    wavelength = 1.0
    vars = {'k': 2.0, 'offset': 0.5}
    
    res = evaluate_formula("x * k + offset", vars, x, y, wavelength)
    assert np.allclose(res, 2.5)

def test_evaluate_formula_errors():
    x = np.zeros(5)
    y = np.zeros(5)
    
    # Syntax error
    with pytest.raises(Exception, match="Syntax Error"):
        evaluate_formula("x * ", {}, x, y, 1.0)
    
    # Unknown variable
    with pytest.raises(Exception, match="Runtime Error"):
        evaluate_formula("unknown_var", {}, x, y, 1.0)

def test_evaluate_formula_coordinates():
    # Test r, theta, phi
    x = np.array([1.0, 0.0, -1.0])
    y = np.array([0.0, 1.0, 0.0])
    # r should be 1
    res_r = evaluate_formula("r", {}, x, y, 1.0)
    assert np.allclose(res_r, 1.0)
    
    # theta (atan2(y,x))
    # (1,0) -> 0
    # (0,1) -> pi/2
    # (-1,0) -> pi
    res_theta = evaluate_formula("theta", {}, x, y, 1.0)
    expected = np.array([0, np.pi/2, np.pi])
    assert np.allclose(res_theta, expected)

def test_evaluate_formula_annular_phase():
    # Phase formula: -2*pi/lambda*(sqrt((x-x0)^2+(y-y0)^2+f^2)-f)
    x = np.array([20e-6, 0.0, -20e-6])
    y = np.array([0.0, 20e-6, 0.0])
    wavelength = 0.532e-6
    custom_vars = {'f': 50e-6, 'x0': 0.0, 'y0': 0.0}
    formula = "-2*pi/lambda*(sqrt((x-x0)^2+(y-y0)^2+f^2)-f)"
    
    res = evaluate_formula(formula, custom_vars, x, y, wavelength)
    
    # Expected analytical calculation
    # For point (20e-6, 0): r = 20e-6
    # sqrt(r^2 + f^2) - f = sqrt((20e-6)^2 + (50e-6)^2) - 50e-6 = 10e-6 * (sqrt(2^2 + 5^2) - 5)
    # sqrt(29) - 5 = 5.3851648 - 5 = 0.3851648
    # value = 3.851648e-6
    # phase = -2*pi / (0.532e-6) * 3.851648e-6 = -2*pi * 7.2399... = -45.4899...
    r_sq = x**2 + y**2
    expected = -2 * np.pi / wavelength * (np.sqrt(r_sq + custom_vars['f']**2) - custom_vars['f'])
    
    assert np.allclose(res, expected, atol=1e-6)
