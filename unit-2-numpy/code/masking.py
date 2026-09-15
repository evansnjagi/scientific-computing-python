"""
    Scientific Computing with Python
        Lesson 2: NumPy - Boolean indexing/masking
"""

# Import
import numpy as np

# Temperature data
temps = np.array([18, 22, 15, 30, 25, 12, 28])
hot_days = temps[temps > 20]
print(hot_days)

# Example 2
temperatures = np.array([20.1, 21.5, 19.8, 22.3, 18.9])
mean_temp = np.mean(temperatures)
above_avg = temperatures[temperatures > mean_temp]
print(f"Above average temp: {above_avg}")

# Aggregation
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Sum: {np.sum(matrix)}")
print(f"Row sum: {np.sum(matrix, axis = 1)}")
print(f"Column sum: {np.sum(matrix, axis = 0)}")

# Matrix multiplication
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 0], [0, 1], [1, 1]])

C = A @ B

print(f"Shape of C = {C.shape}")
print(C)

# Reshaping arrays
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
reshaped = arr.reshape(3, -1)
print(f"Reshaped : \n{reshaped}")
print(f"Reshaped shape: {reshaped.shape}")