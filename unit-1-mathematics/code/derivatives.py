"""
Scientific Computing with Python
    Unit-1-Mathematics
    ------------------
    Derivatives
"""
# Import 
import numpy as np

# Numerical derivative functions
def numerical_derivative(f, x: float, h: float = 1e-5) -> float:
    """
        Approximate derivative of a function f with respect to x using a small step size h.

        Parameters
        ---------
        f: callable
            Function f to differentiate

        x: float
            The value of x, to evaluate the differentiated function
        h: float = 1e5
            Small step size, default 1e5

        Returns
        -------
        float
            Approximated differentiation with respect to x.
    """
    return (f(x + h) - f(x)) / h

def partial_derivative_x(f, x: float, y: float, h: float = 1e-5) -> float:
    """
        Approximate the partial derivative of a two-variable function, with respect to x, holding the value of y fixed.

        Parameters
        ----------
        f: callable
            The function of two variables, f(x, y) to differentiate.
        x: float 
            The value of x at which to evaluate the partial derivative.
        y: float
            The y-value, held fixed during differentiation.
        h: float
            A small step size, by default 1e-5.

    Returns
    -------
    float
        Approximate partial derivate of `f` with respect to x at (x, y).
    """
    return (f(x + h, y) - f(x, y)) / h

def partial_derivative_y(f, x: float, y: float, h: float = 1e-5) -> float:
    """
        Approximate partial derivative of f with respect to y, keeping x constant.

        Parameters
        ---------
        f: callable
            A function to partially differentiate with respect to x. The function has two entires, x-values and y-values: f(x, y).
        y: float
            The value of y at which evaluation is done.
        x: float
            The x-value, fixed.
        h: float
            Small step size, fixed at 1e-5.
        
        Returns
        -------
        float
            An approximated numerical partial derivative with respect to x, y kept constant.
    """
    return (f(x, y + h) - f(x, y)) / h

def square(x: float) -> float:
    return x ** 2

print(f"{numerical_derivative(square, 3, 1e-15)}")

# Chain example
def chain_example(x: float) -> float:
    return (3 * x + 1) ** 2

print(f"{numerical_derivative(chain_example, 2)}")
print(f"{np.isclose(numerical_derivative(chain_example, 2), 42)}")

# Partial derivatives
def f(x: float, y: float) -> float:
    return x ** 3 + 2 * x * y ** 2 - y

print(f(2, 3))
print(partial_derivative_x(f, 2, 3))
print(partial_derivative_y(f, 2, 3))