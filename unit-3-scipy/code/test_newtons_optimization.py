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
    f_double_prime_yx
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
    assert np.isclose(f_double_prime_xy(bowl, 1.0, 1.0), 0.0, atol = 1e-4)

    # Test using mixed function, expected f_xy = 1
    assert np.isclose(f_double_prime_xy(mixed, 1.0, 1.0), 1.0, atol = 1e-4)

# 4. Test the second derivative of f with respect to y, then another derivative with respect to x, i.e. df/dydx also, f_yx.

def test_hessian_yx():
    # Bowl function
    def bowl(x: float, y: float) -> float:
        return x**2 + y**2

    # Mixed function
    def mixed_function(x: float, y: float) -> float:
        return x * y

    # Test using the bowl function, expected f_yx = 0
    assert np.isclose(f_double_prime_yx(bowl, 1.0, 1.0), 0.0, atol = 1e-4)

    # Test using the mixed funcition, expected f_yx = 1.
    assert np.isclose(f_double_prime_yx(mixed_function, 1.0, 1.0), 1.0, atol = 1e-4)

