"""
    Scientific Computing with Python
    Lesson 10: Introduction to pytest 
"""
# Import
from functions import compute_mean

# Testing function
def test_compute_mean_basic():
    result = compute_mean([1.0, 2.0, 3.0])
    assert result == 2.0

def test_compute_mean_single_value():
    assert compute_mean([5.0]) == 5.0

def test_compute_mean_negative_values():
    assert compute_mean([-1.0, -2.0, -3.0]) == -2.0

# def test_compute_mean_wrong_on_purpose():
#     assert compute_mean([1.0, 2.0, 3.0]) == 99.0 # Deliberate incorrect 