# Import 
import numpy as np
import time

a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(a)
print(type(a))

# Create a large dataset
size = 1_000_000
python_list = list(range(size))
numpy_array = np.array([python_list])

# Method one, Python list sum
start = time.time()
total = 0.0

for num in python_list:
    total += num
python_time = time.time() - start

# Method two, NumPy sum
start = time.time()
numpy_total = np.sum(numpy_array)
numpy_time = time.time() - start

print(f"Python loop time: {python_time:.5f} seconds.")
print(f"NumPy time: {numpy_time:.5f} seconds.")
print(f"Numpy is {python_time / numpy_time} x faster.")

# Numpy array operations
b = np.arange(0, 20, 2)
print(b)
print(b.shape)
print(b[2: 5])

# Vectorized operations
temp_celsius = np.array([0, 10, 20, 30, 40])
temp_farenheit = temp_celsius * 9 / 5 + 32

print(temp_farenheit)


# Matrix
matrix = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print(f"Matrix X = \n{matrix}")
print(matrix.shape)
print(matrix[1, 2])
print(matrix[0,:])
print(matrix[:, 2])

# Broadcasting
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row = np.array([10, 20, 30])
print(f"A = \n{matrix}")
print(f"row = \n{row}")

print(f" matrix + row = \n{matrix + row}")