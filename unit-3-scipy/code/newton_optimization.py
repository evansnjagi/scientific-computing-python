"""
    Scientific Computing with Python Mastery
    Lesson 3: Scipy
        Newtons Optimization Script, Multivariable Computation  
"""
# Import 
import numpy as np

from optimize import gradient 

# Second derivation function
def f_double_prime_xx(f, x: float, y: float, h: float = 1e-4) -> float:
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
            Step size, h, default = 1e-4

        Returns
        -------
        float
            The second derivative of a function f, with respect to x. 
    """ 
    return (f(x+h, y) - 2*f(x, y) + f(x - h, y)) / h**2

# Second derivative of f, with respect to y, computing f_yy
def f_double_prime_yy(f, x: float, y: float, h: float = 1e-4) -> float:
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
            Step size, h, default = 1e-4

        Returns
        -------
        float
            The second derivative of a function f, with respect to y. 
    """ 
    return (f(x, y + h) - 2*f(x, y) + f(x, y - h)) / h**2

# Derivative of f with respect to x, then another derivative with respect to y, i.e. f_xy
def f_double_prime_xy(f, x: float, y: float, h: float = 1e-4) -> float:
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
            The small step size, default = 1e-4.
        
        Returns
        -------
        float
            The computed derivative of f, df/dxdy
    """
    return (f(x + h, y + h) - f(x + h, y-h) - f(x-h, y + h) + f(x -h, y - h)) / (4 * h**2)

# Hessian matrix
def hessian(f, x: float, y: float, h: float = 1e-4) -> np.ndarray:
    """
        The Hessian matrix:
            f_xx f_xy
            f_xy f_yy
        Parameters
        ----------
        f: callable
            The function to compute the hessian matrix from.
        x: float
            The float value of x.
        y: float
            The value of y.
        h: float, optional
            Small step size, default = 1e-4

        Returns
        -------
        np.ndarray
            A 2x2, Hessian matrix, H. 
    """
    # Matrix entries
    f_xx = f_double_prime_xx(f, x, y, h)
    f_yy = f_double_prime_yy(f, x, y, h)
    f_xy = f_double_prime_xy(f, x, y, h)

    # Return
    return np.array([[f_xx, f_xy], [f_xy, f_yy]])

# Compute the newtons optimization
def newtons(f, x: float, y: float, h: float = 1e-4) -> np.ndarray:
    """
        Newton's multivariable optimization, i.e. f(x,y).

        Parameters
        ----------
        f: callable
            The function to compute the newton's optimization.
        x: float
            The value of x to evaluate the algorithm
        y: float
            The value of y, to evaluate the algorithm
        h: float, optional
            Small step size used during differentiation, default = 1e-4
        Returns
        -------
        np.ndarray
            A vector of the optimal solution, i.e (x_value, y_value)
    """
    # Get the gradient
    f_tilder = gradient(f, x, y, h)

    # Inverse of Hessian matrix
    h_inverse = np.linalg.inv(hessian(f, x, y, h))

    # Return newtons optimization
    return (x, y) - np.dot(h_inverse, f_tilder)