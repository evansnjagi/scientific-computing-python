"""
    Scientific Computing with Python
    --------------------------------
    Lesson 3: Foundations - Functions
"""
# Import
import math

# Compute mean
def compute_mean(values: list[float]) -> float:
    """
        Compute the arithmetic mean of a list of numbers

        Parameters
        ----------
        values: list[float]
            A list of numeric values.

        Returns
        --------
        float
            The arithmetic mean of input values.

    """
    total = sum(values)
    return total / len(values)

# Compute standard deviation
def compute_std_dev(values: list[float]) -> float:
    """
        Compute standard deviation for a list of numbers.
        using Bessel's correction (n - 1).

        Parameters
        ----------
        values: list[float]
            A list of real numerical numbers 

        Returns
        -------
        float
            Computed standard deviation
    """
    # Get mean
    mean = compute_mean(values)

    # Compute squared difference
    sqrd_diff = 0.0
    for v in values:
        sqrd_diff += (v - mean) ** 2

    # Compute average - Variance
    var = sqrd_diff/ (len(values) - 1)

    # Compute standard deviation
    return math.sqrt(var)

# Variable initializing
temperatures = [20.1, 21.5, 19.8, 22.3, 18.9]

avg_temp = compute_mean(temperatures)
std_dev = compute_std_dev(temperatures)
print(f"Average temperature: {avg_temp}")
print(f"Temp standard deviation: {std_dev:.4f}")