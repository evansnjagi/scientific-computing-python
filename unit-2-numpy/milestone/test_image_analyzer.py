"""
    Scientific Computing with Python
    unit 2: NumPY
    Tests
"""

# Import 
import numpy as np
import pytest 

from image_analyzer import (
    data_setup, 
    normalize, 
    threshold,
    row_col_stats,
    flatten_reshape
    )

# Test data setup
# 1. Normal test
def test_data_setup_normal():
    assert data_setup(0, 255).shape == (8, 8)

# Matrix testing
def test_data_setup_matrix_addition():
    A = data_setup(0, 150)
    B = data_setup(151, 255)
    assert np.all((A + B) == (B + A))

def test_data_setup_matrix_multiplication():
    A = data_setup(0, 8)
    B = data_setup(9, 16)
    assert not np.all(A @ B == B @ A)

# 2. Dimension test
def test_data_setup_dimensions():
    assert data_setup(0, 255, (4, 4)).shape == (4, 4)

# Image normalization test
# 1. Normal test
def test_normalize_normal():
    image = np.array([[231, 10], [204, 212]])
    assert np.allclose(normalize(image), np.array([[1, 0], [0.8778280543, 0.9140271493]]))

# 1. Edge case, (1, 1) grayscale image
def test_normalize_edge_case():
    image = data_setup(0, 255, (1, 1))
    assert np.isnan(normalize(image))


# Test Threshold function
def test_threshold_normal():
    image = np.array([[231, 10], [204, 212]])
    assert np.all(threshold(image, 210) == np.array([[1, 0], [0, 1]]))

# Test edge case
def test_threshold_edge_case():
    image = np.array([[231, 10], [204, 212]])
    assert np.all(threshold(image, 212) == np.array([[1, 0], [0, 0]]))
    
def test_threshold_edge_case_min():
    image = np.array([[231, 10], [204, 212]])
    assert np.all(threshold(image, 0) == np.array([[1, 1], [1, 1]]))

def test_threshold_edge_case_max():
    image = np.array([[231, 10], [204, 212]])
    assert np.all(threshold(image, 255) == np.array([[0, 0], [0, 0]]))

# Test row_col_stats
def test_row_col_stats_normal_dict():
    image = np.array([[231, 10], [204, 212]])
    assert type(row_col_stats(image)) == dict

def test_row_col_stats_axis_0_exist():
    image = np.array([[231, 10], [204, 212]])
    assert "axis_0" in row_col_stats(image)

def test_row_col_stats_axis_0_aggregate():
    image = np.array([[231, 10], [204, 212]])
    assert np.all(row_col_stats(image)["axis_0"] ==  np.array([217.5, 111]))

def test_row_col_stats_axis_1_exist():
    image = np.array([[231, 10], [204, 212]])
    assert "axis_1" in row_col_stats(image)

def test_row_col_stats_axis_1_aggregate():
    image = np.array([[231, 10], [204, 212]])
    assert np.all(row_col_stats(image)["axis_1"] ==  np.array([120.5, 208]))

# Test flatten_reshape function
def test_flatten_reshape_normal_shape():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    assert flatten_reshape(matrix, (2, -1)).shape == (2, 6)

def test_flatten_reshape_normal_matrix():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    assert np.all(flatten_reshape(matrix, (2, -1)) == np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]))

def test_flatten_reshape_error():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    with pytest.raises(ValueError):
        flatten_reshape(matrix, (2, 5))