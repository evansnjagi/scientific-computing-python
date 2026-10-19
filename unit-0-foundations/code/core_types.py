"""
    Scientific Computing with Python
    ------------------------------
    Lesson 2: Foundation - Core data types
"""
# 1. List
print("\n=== Core types ===\n")
temperatures = [20.1, 21.5, 19.8, 22.3, 18.9]
print(f"Temperatures: {temperatures}")
print(f"Type of temperatues: {type(temperatures)}")
print(f"Length of temperatures list: {len(temperatures)}")
print(f"First item: {temperatures[0]}")
print(f"Last item: {temperatures[-1]}")
print(f"1st to 3rd item: {temperatures[1:3]}")

# 2. Computing with a loop
total = 0.0
for t in temperatures:
    total = total + t 
avg_temp = total/len(temperatures)
print(f"Averange temperature: {avg_temp}")
print(f"Average type: {type(avg_temp)}")