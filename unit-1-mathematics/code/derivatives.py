"""
Scientific Computing with Python
    Unit-1-Mathematics
    ------------------
    Derivatives
"""
def numerical_derivative(f, x: float, h: float = 1e-5) -> float:
    """
        Approximate the limit of a function at a point using the limit defination of a derivative.

        Parameters
        ----------
        f: callable
            The function to differentiate.
        x: float 
            Point at which to evaluate the derivative.
        h: float
            A small step size, by default 1e-5.

    Returns
    -------
    float
        Approximate derivate of `f` at `x`.
    """
    return (f(x + h) - f(x)) / h

def square(x: float) -> float:
    return x ** 2

print(f"{numerical_derivative(square, 3, 1e-15)}")