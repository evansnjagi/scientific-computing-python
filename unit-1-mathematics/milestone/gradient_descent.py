# Import
import numpy as np

# Function definitions
def partial_derivative_x(
        f,
        x: float, 
        y: float, 
        h: float = 1e-5
) -> float:
    """
        Approximate partial derivative of `f` with respect to x.

        Parameters
        ----------
        f: callable
            The function, f(x, y).
        x: float
            The value of x at which to evaluate the partial derivative
        y: float
            The value of y, held constant.
        h: float, optional
            The small step size, default = 1e-5.

    Returns
    -------
    float
        Partial derivative of f with respect to x.
    """
    return (f(x + h, y) - f(x, y)) / h

def partial_derivative_y(
    f,
    x: float,
    y: float, 
    h: float = 1e-5
) -> float:
    """
        Approximate partial derivative of function `f` with respect to y.

        Parameters
        ----------
        f: callable
            The function, f(x, y).
        x: float
            The value of x, kept constant.
        y: float
            The value of y at which to evaluate the partial derivative.
        h: float, optional
            The small step, default = 1e-5
    """
    return (f(x, y + h) - f(x, y)) / h

def gradient(
    f,
    x: float,
    y: float,
    h: float = 1e-5
) -> tuple[float, float]:
    """
        Approximate the gradient, direction of the steepest increase of the two variable function, f(x, y) - (df/dx, df/dy).

        Parameters
        ----------
        f: callable
            A two variable function, f(x, y) to which gradient is computed.
        x: float
            The x value at which the function is evaluated, df/dx
        y: float
            The  value of y at which f(x, y) is evaluated to get df/dy.

        h: float, optional
            Small step size, default = 1e-5.  

    Returns
    -------
    tuple[float, float]
        The approximated gradient (df/dx, df/dy)
    """
    df_dx = partial_derivative_x(f, x, y, h)
    df_dy = partial_derivative_y(f, x, y, h)

    # Return 
    return (df_dx, df_dy)

def gradient_descent(
    f,
    x_start: float,
    y_start: float,
    h: float = 1e-5,
    lr: float = 0.1,
    steps: int = 100
) -> tuple[float, float]:
    """
        Approximated gradient descent, computed using numerical derivative.

        Parameters
        ----------
        f: callable
            A two-variable function, f(x, y).
        x_start: float
            The starting value of x.
        y_start: float
            The starting value of y.
        h: float, optional
            A small step size, default = 1e-5.
        lr: float, optional
            The learning rate, alpha, default = 0.1.
        steps: int, optional
            Small steps towards STEEPEST slope, default = 100.

        Returns
        -------
        tuple[float, float]
            Local minimum value, (df_dx, df_dy)
    """
    x = x_start
    y = y_start
    # Loop
    for _ in range(steps):
        df_dx, df_dy = gradient(f, x, y)
        x = x -  lr * df_dx
        y = y - lr * df_dy

    # Return 
    return (x, y)
# Function f
def f(x: float, y: float) -> float:
    return x**2 + y**2 + 4*x - 6*y

function_solution = f(-2, 3)
print(f"Function solution: {function_solution}")
print(f"Match hand-derived solution: {np.isclose(function_solution, -13)}")

df_dx = partial_derivative_x(f, 1, 1)
print(f"Partial derivative with respect to x: {df_dx}")
print(f"Match hand-derived solution: {np.isclose(df_dx, 6)}")

df_dy = partial_derivative_y(f, 1, 1)
print(f"Partial derivative  with respect to y: {df_dy}")
print(f"Match hand-derived solution: {np.isclose(df_dy, -4)}")

gradient_point = gradient(f, 1, 1)
print(f"Gradient point: {gradient_point}")

converge_point = gradient_descent(f, 5, 8)
print(f"Convergence point: {converge_point}")

# Experiment
# 1. Learning rate near the threshold
converge_near_threshold = gradient_descent(f, 5, 8, lr=0.99)
print(f"Learning rate near threshold: {converge_near_threshold}")

# 2. few vs many steps
few_steps = gradient_descent(f, 5, 8, steps=50)
many_steps = gradient_descent(f, 5, 8, steps=1000)
print(f"Few vs many steps: few steps -> {few_steps}, many steps -> {many_steps}")