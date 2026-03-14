
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
    res = evaluate_formula("x * ", {}, x, y, 1.0)
    assert res is None
    
    # Unknown variable
    res = evaluate_formula("unknown_var", {}, x, y, 1.0)
    assert res is None

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
