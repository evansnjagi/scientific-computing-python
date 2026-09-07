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

# Gradient
def gradient(f, x: float, y: float, h: float = 1e-5) -> tuple[float, float]:
    """
        Approximate gradient of a two variable function at a point.

        Parameters
        ----------
        f: callable
            A two variable function, f(x, y).
        x: float
            The x-value at which to evaluate the gradient.
        y: float
            The y-value at which to evaluate the gradient
        h: float, optional
            A small step size, default 1e-5.

        Returns
        --------
        tuple[float, float]
            The approximate gradient, (df/dx, df/dy) at (x, y).
    """
    # Partial derivative of f with respect to x
    df_dx =  partial_derivative_x(f, x, y, h)

    # Partial derivative of f with respect to y
    df_dy = partial_derivative_y(f, x, y, h)
    
    # Return
    return (df_dx, df_dy)

# Gradient descent
def gradient_descent(
        f,
        x_start: float, 
        y_start: float,
        learning_rate: float = 0.1,
        steps: int = 100
) -> tuple[float, float]:
    """
        Minimize a two variable function using gradient descent.

        Parameters
        ----------
        f: callable
            A two variable function, f(x, y), to minimize.
        x_start: float
            Initial guess for x.
        y_start: float
            Initial guess for y.
        learning_rate: float, optional
            Step size, alpha, by default 0.1.
        steps: int, optional
            Number of iterations to perform, by default 100.
        
        Returns
        -------
        tuple[float, float]
            The (x, y) point after minimization. 
    """
    x = x_start
    y = y_start


    # Loop
    for i in range(steps):
        df_dx, df_dy = gradient(f, x, y)
        x = x - learning_rate * df_dx
        y = y - learning_rate * df_dy

    # Return 
    return (x, y)

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

# Gradient 
def f2(x: float, y: float) -> float:
    return 5 * x**2 * y + 3 * y**3 - 2 * x

print(f2(2, 3))
print(gradient(f2, 2, 3))

# Gradient descent test
def bowl(x: float, y: float) -> float:
    return x**2 + y**2

result = gradient_descent(bowl, 5, 8)
print(result)

# Learning rate experiments
result_slow = gradient_descent(bowl, 5, 8, 0.001, 100)
print(result_slow)

result_slow_more_steps = gradient_descent(bowl, 5, 8, 0.001, 10000)
print(result_slow_more_steps)

result_fast = gradient_descent(bowl, 5, 8, 1.1, 100)
print(result_fast)

result_edge = gradient_descent(bowl, 5, 8, 0.99, 100)
print(result_edge)