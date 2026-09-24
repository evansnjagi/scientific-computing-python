"""
    Scientific Computing with Python Mastery
        Lesson 4: SciPy
        Introduction
"""
# Import
import numpy as np
from scipy import optimize

# Text function, bowl function
def bowl(point):
    x, y = point
    return (x**2) + (y**2)

# Get results
result = optimize.minimize(bowl, x0=[5, 8])

print(result.x)
print(result)

# Rosenbrock/banana benchmark function
def rosenbrock(point):
    x, y = point
    return (1 - x)**2 + 100*(y - x**2)**2

# Optimizing rosenbrock function
result_rosenbrock = optimize.minimize(rosenbrock, x0 = [-1, 1])
print(result_rosenbrock)
print(result_rosenbrock.x)
print(result_rosenbrock.success)
print(result_rosenbrock.nit)