"""
    Scientific Computing with Python Mastery
        Unit 3: SciPY
        Optimization algorithms
"""
# Import 
import numpy as np


# Gradient descent optimization algorithm
# 1. Compute partial derivation of f with respect to x
def partial_derivative_x(f, x: float, y: float, h: float = 1e-5) -> float:
    """
        Partial derivative of a composite function, f(x, y) with respect to x.

        Parameters
        ----------
        f: callable
            The function to perform partial derivative.
        x: float
            The x value.
        y: float
            The y value.
        h: float, optional
            The step size, close to zero, default = 1e-5.

        Returns
        ------
        float
            Partial derivative of f with respect to x.
    """
    return (f(x+h, y) - f(x, y)) / h

# 2. Partial derivative of f with respect to y
def partial_derivative_y(f, x: float, y: float, h: float = 1e-5) -> float:
    """
        Computes the partial derivative of the composite function, f(x, y) with respect to y.

        Parameters
        ----------
        f: callable
            The composite function, f(x, y)
        x: float
            The x value for the function
        y: float
            The y value.
        h: float, optional
            The step size, default = 1e-5

        Returns
        -------
        float
            The partial derivative of the composite function, f(x, y) with respect to y.
    """
    return (f(x, y+ h) - f(x, y)) / h

# 3. Compute gradient of the function
def gradient(f, x: float, y: float, h: float = 1e-5) -> tuple[float, float]:
    """
        Compute gradient \tilder f(x, y)

        Parameters
        ----------
        f: callable
            The composite function to compute the gradient.
        x: float
            The value of x.
        y: float
            The value of y.
        h: float, optional
            The step size, default = 1e-5

        Returns
        -------
        tuple[float, float]
            A tuple with the gradient of the composite function, exact value of x and y after derivative.
    """
    # Compute partial derivatives
    df_dx = partial_derivative_x(f, x, y, h)
    df_dy = partial_derivative_y(f, x, y, h)

    # Return
    return (df_dx, df_dy)

# 4. Compute gradient descent, optimization
def gradient_descent(f, x_start: int, y_start: int, h: float = 1e-5, steps: int = 100, lr: float = 0.1) -> tuple[float, float]:
    """
        Compute the gradient descent of a composite funciton, f(x, y)

        Parameters
        ----------
        f: callable
            The composite function, f(x, y)
        x_start: int
            The starting value of x.
        y_start: int
            The starting value of y.
        h: float, optional
            The step size, default = 1e-5.
        steps: int, optional
            The number of iteration to be taken, default = 100
        lr: float, optional
            The learning rate, default = 0.1

        Returns
        -------
        tuple[float, float]
            The minimum values of x and y.
    """
    x = x_start
    y = y_start

    # Loop
    for _ in range(steps):
        df_dx, df_dy = gradient(f, x, y, h)
        x = x - lr * df_dx
        y = y - lr * df_dy

    # Return 
    return (x, y)

# The first derivative, f_prime
def f_prime(f, x: float, h: float = 1e-5) -> float:
    """
        The first derivative of f(x), using numerical method.

        Parameters
        ----------
        f: callable
            The function to perform derivative on.
        x: float
            The point at which to evaluate the derivative.
        h: float, optional
            The step size, default = 1e-5.
        Returns
        ------
        float
            Derivative of the function f'(x) at the point x.
    """
    return (f(x + h) - f(x)) / h

# Second derivative of the function f(x)
def f_double_prime(f, x: float, h: float = 1e-5) -> float:
    """
        The second derivative, using numerical method.

        Parameters
        ---------
        f: callable
            The function to perform the second derivative on.
        x: float
            The evaluating value of x.
        h: float, optional
            The step size, default = 1e-5

        Returns
        -------
        float
            The second derivative for the function f(x).
    """
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2

# Newtons optimization method
def newtons_optimization(f, x_start: float, h: float = 1e-5, steps: int = 10) -> float:
    """
        Minimize a function using Newton's method(single variable, using it's first and second derivative.

        Parameters
        ---------
        f: callable
            The function to perform optimization on.
        h: float, optional
            The step size, default = 1e-5.
        x_start: float
            Initial starting point for x.
        steps: int, optional
            Number of iterations to perform, default = 10.

        Returns
        ------
        float
            The x-value found at perfoming Newton's optimization.
    """
    x = x_start

    for _ in range(steps):
        second_derivative  = f_double_prime(f, x, h)
        if np.isclose(second_derivative, 0.0):
            raise ValueError(
                f"Curvature f'' is zero or near zero at x = {x}; "
                "Newton's method can not continue safely."
            )
        x = x - (f_prime(f, x, h)/ second_derivative)

    return x