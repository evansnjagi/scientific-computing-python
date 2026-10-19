"""
    Scintific Computing with Python Mastery
    Unit 3: Scipy
        Newtons Optimization Test script
"""
# Import
import numpy as np

from newton_optimization import (
    f_double_prime_xx,
    f_double_prime_yy,
    f_double_prime_xy,
    hessian,
    newtons
)

# Tests
# 1. Test f_xx = 2, using bowl function
def test_hessian_xx_bowl():
    # Bowl function
    def bowl(x: float, y: float) -> float:
        return x**2 + y**2

    # Actual test
    assert np.isclose(f_double_prime_xx(bowl, 1.0, 1.0), 2.0)

# 2. Test the second derivative, fyy = 2, using the bowl funciton
def test_hessian_yy_bowl():
    # The bowl function
    def bowl(x: float, y: float) -> float:
        return x**2 + y **2

    # Test
    assert np.isclose(f_double_prime_yy(bowl, 1.0, 1.0), 2.0)

# 3. Test the second derivative of f with respect to x, then another derivative with respect to y.
def test_hessian_xy():
    # Bowl function
    def bowl(x: float, y: float) -> float:
        return x**2 + y**2
    # Mixed function
    def mixed(x: float, y: float) -> float:
        return x*y

    # The test, using bowl function, expected f_xy = 0
    assert np.isclose(f_double_prime_xy(bowl, 1.0, 1.0), 0.0)

    # Test using mixed function, expected f_xy = 1
    assert np.isclose(f_double_prime_xy(mixed, 1.0, 1.0), 1.0)


# Testing the hessian matrix with a quadratic function
def test_hessian_quadratic():
    # Quadratic function
    def quadratic(x: float, y: float) -> float:
        return x**2 + x*y + 2*y**2

    expected = np.array([[2, 1], [1, 4]])

    # |f| at (1, 1) = 4, h = 1e-5, eps = 2.2e-16
    # f_xx and f_yy noise is eps * |f|/h**2 => 8.8e-8
    # # f_xy noise is eps * |f| / (4 * h**2) => 2.2e-8
    # Truncation error = 0, the function is quadratic
    # atol = worst noise * 10 => 8.9e-9 * 10 = 9e-7, rounded to the power of ten = 1e-6.
    
    assert np.allclose(hessian(quadratic, 1, 1), expected, atol=1e-6)

# Newtons optimization test
def test_newtons_optimization():
    # Quadratic function
    def f(x: float, y: float) -> float:
        return x**2 + x*y + 2*y**2

    # Expected
    expected = np.array([0.0, 0.0])
    # The test
    assert np.allclose(newtons(f, 1, 1), expected)