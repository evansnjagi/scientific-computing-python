"""
    Scientific Computing with Python Mastery
        Testing gradient algorithms
"""
# Import
import numpy as np
import pytest

from optimize import (
    partial_derivative_x,
    partial_derivative_y,
    gradient,
    gradient_descent,
    f_prime, 
    f_double_prime,
    newtons_optimization
)
# Tests
# Testing functions
# 1. Bowl function
def bowl(x: float, y: float) -> float:
    return x**2 + y**2

# 2. Rosenbrock function
def rosenbrock(x: float, y: float) -> float:
    return (1 - x)**2 + 100*(y - x**2)**2

# 3. A single variable function, f(x) = x^4 - 3x^2 + 2
def single_variable_function(x: float) -> float:
    return x**4 - 3*x**3 + 2


# 1. Test partial derivative of f with respect to x, at x = 1
def test_partial_derivative_x():
    assert np.isclose(partial_derivative_x(bowl, 1, 1), 2)

# 2. Test partial derivative of f with repsect to y, at y = 1.
def test_partial_derivative_y():
    assert np.isclose(partial_derivative_y(bowl, 1, 1), 2)

# 3. Test the gradient function
def test_gradient():
    assert type(gradient(bowl, 1, 1)) == tuple
    assert np.allclose(gradient(bowl, 1, 1), (2, 2))

# 4. Test gradient descent, using bowl function
def test_gradient_descent_one_step():
    assert type(gradient_descent(bowl, 5, 8, steps=1)) == tuple
    assert np.allclose(gradient_descent(bowl, 5, 8, steps = 1), (4, 6.4))
    assert np.allclose(gradient_descent(bowl, 5, 8), (0, 0), atol = 1e-5)

# 5. Test gradient descent using a harder function, Rosenbrock function.
def test_gradient_descent_using_rosenbrock_function():
    """
        Plain gradient descent optimization struggles to converge on a function like Rosenberg due to the function highly varying curvature. See on (unit 3, scipy, notes). This test confirms it moves to atleast closer to it's minimum point (1,1), rather than expecting a full convergence from a distant start.
    """
    assert np.allclose(gradient_descent(rosenbrock, 1.5, 1.5, lr = 0.001, steps = 10000), (1, 1), atol = 1e-2)

# 6. Test first derivative of the function f(x) = x^4 - 3x^3 + 2, at x  = 1
def test_f_prime():
    assert np.isclose(f_prime(single_variable_function, 1), -5, atol = 1e-4)

# 7. Test second derivative of f(x) = x^4 = 3^3 + 2, at x = 1
def test_f_double_prime():
    assert np.isclose(f_double_prime(single_variable_function, 1), -6, atol = 1e-4)

# 8. Test Newton's optimization, starting from 3
def test_newtons_optimization():
    assert np.isclose(newtons_optimization(single_variable_function, x_start=3), 2.25, atol = 1e-4)

# 9. Test what Newton's optimization raises on zero curvature
def test_newtons_optimization_raises_zero_curvature():
    def flat_function(x: float) -> float:
        return 5.0 # Constant zero curvature everywhere
    with pytest.raises(ValueError):
        newtons_optimization(flat_function, x_start= 3)
