"""
    Scientific Computing with Python
    Unit 2: NumPy
    Milestone Project
"""
# Import
import numpy as np

# Data setup
def data_setup(low: int, high: int, size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
        A NumPy 2D array, simulating a gray scale image.

        Parameters
        ----------
        low: int 
            Lowest pixel value to pick from.

        high: int
            The highest pixel value the image can have.
        size: tuple[int, int]
            The size of the matrix, height and width e.g. (8, 8) 

    Returns
    -------
    np.ndarray, optional
        2D array simulating image pixels, default (8, 8).
    """
    return np.random.randint(low = low, high = high, size = size)

# Functions
def normalize(image: np.ndarray) -> np.ndarray:
    """
        min-max normalization, X' = (x - min) / (max - min)

        Parameters
        -----------
        image: np.ndarray
            2D matrix of image pixels, i.e. (8, 8) grayscale image.

        Returns
        -------
        np.ndarray
            A normalized 2D matrix, values between 0 and 1.
    """
    min = np.min(image)
    max = np.max(image)

    # Return
    try:
        return (image - min) / (max - min)
    except Exception as err:
        print(err)
        return np.nan

def threshold(image: np.ndarray, cutoff: int) -> np.ndarray:
    """
        Return every pixel above cutoff as 1 and every pixel at or below as 0.

        Parameters
        ----------
        image: np.ndarray
            A 2D matrix containing gray scale image pixels
        cutoff: int
            The point, an integer, where the boundary starts.

        Returns
        -------
        np.ndarray
            A 2D array, with the same shape as the image, containing 0 and 1's as the entries.
    """
    return (image > cutoff - 1).astype(int)

def row_col_stats(image: np.ndarray) -> dict:
    """
        Axis based aggregation returned as a dictionary. The first entry is axis = 0, mean computed by collapsing columns of the image matrix. The second entry is axis = 1, where mean is computed by collapsing the rows of the image matrix.

        Parameters
        ----------
        image: np.ndarray
            A 2D array with pixel values of a gray scale image.
        
        Returns
        -------
        dict
            A dictionary with mean values on every axis of the image matrix.
    """
    return {
        "axis_0": np.mean(image, axis=0),
        "axis_1": np.mean(image, axis=1)
    }

def flatten_reshape(image: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    """
        Reshape the image pixel matrix into a new shape, (x, y) with `y` representing the vertical height of the matrix and `x` representing the horizontal width of the image matrix.

        Parameters
        ----------
        image: ndarray
            The 2D array of the image pixels.
        new_shape: tuple[int, int]
            The shape of the new matrix to be formed. E.g. (2, -1) an array with 2 rows and fitting number of columns.

        Returns
        -------
        np.ndarray
            A numpy 2D array of the size provided
    """
    try:
        return image.flatten().reshape(new_shape)
    except ValueError as err:
        raise ValueError(f"Cannot reshape array: {err}")  
    
print(data_setup(0,  255, (8, 8)))