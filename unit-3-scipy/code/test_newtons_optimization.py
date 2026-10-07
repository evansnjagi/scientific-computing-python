"""
    Scintific Computing with Python Mastery
    Unit 3: Scipy
        Newtons Optimization Test script
"""
# Import
import numpy as np

from newton_optimization import (
    f_double_prime_xx,
    f_double_prime_yy
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