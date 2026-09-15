"""
    Scientific Computing with Python
    -------------------------------
    Lesson 1: Float precision
"""
# Import
import numpy as np 

# Def num.
print("Float precision")
a = 0.1
b = 0.2
c = a + b

print(f"Numbers: a = {a} and b = {b}")
print(f" a + b = {c}")

# Wrong way to compare floats
print(f"Float precision check: {c == 3}")

# Correct way to compare floats 
torelance = 1e-9
print("Torelance: ", torelance)
print(f"Manual check using torelance: {abs(c - 0.3) < torelance}")

# Professinal way to check closeness
print(f"Usinp numpy.isclose method: {np.isclose(c, 0.3)}")