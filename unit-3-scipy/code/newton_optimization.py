"""
    Scientific Computing with Python Mastery
    Lesson 3: Scipy
        Newtons Optimization Script     
"""
# Import 
import numpy as np

# Second derivation function
def f_double_prime_xx(f, x: float, y: float, h: float = 1e-5) -> float:
    """
        Second derivative of a functon f with respect to x, f_xx.

        Parameters
        ----------
        f: callable
            The function f to compute the second derivative from.
        x: float
            The value of x.
        y: float
            The value of y.
        h: float, optional
            Step size, h, default = 1e-5

        Returns
        -------
        float
            The second derivative of a function f, with respect to x. 
    """ 
    return (f(x+h, y) - 2*f(x, y) + f(x - h, y)) / h**2

# Second derivative of f, with respect to y, computing f_yy
def f_double_prime_yy(f, x: float, y: float, h: float = 1e-5) -> float:
    """
        Second derivative of a functon f with respect to y, f_yy.

        Parameters
        ----------
        f: callable
            The function f to compute the second derivative from.
        x: float
            The value of x.
        y: float
            The value of y.
        h: float, optional
            Step size, h, default = 1e-5

        Returns
        -------
        float
            The second derivative of a function f, with respect to y. 
    """ 
    return (f(x, y + h) - 2*f(x, y) + f(x, y - h)) / h**2

# Derivative of f with respect to x, then another derivative with respect to y, i.e. f_xy
def f_double_prime_xy(f, x: float, y: float, h: float = 1e-5) -> float:
    """
        The derivative of f with respect to x, then another derivative with respect to y, df/dxdy -> f_xy.

        Parameters
        ----------
        f: callable
            The function to compute the derivative from.
        x: float
            The value of x.
        y: float
            The value of y.
        h: float, optional
            The small step size, default = 1e-5.
        
        Returns
        -------
        float
            The computed derivative of f, df/dxdy
    """
    return (f(x + h, y + h) - f(x + h, y-h) - f(x-h, y + h) + f(x -h, y - h)) / (4 * h**2)

# Derivative of f with respect to y, then another derivative with respect to x.
def f_double_prime_yx(f, x: float, y: float, h: float = 1e-5) -> float:
    """
        Partial derivative of f with respect to y, then another derivative with respect to x, df/dydx

        Parameters
        ----------
        f: callable
            The function to compute partial derivative on.
        x: float
            The value of x.
        y: float
            The value of y.
        h: float, optional
            Small step size, default = 1e-5

        Returns
        -------
        float
            The derivative of f, df/dydx also f_yx.
    """
    return (f(x + h, y + h) - f(x +  h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h**2)