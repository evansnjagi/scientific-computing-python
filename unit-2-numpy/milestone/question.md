# Image Like Data Analyzer
> The goal is to combine everything we have been learning from this unit; array creation, indexing, vectorized operations, 2D arrays, broadcasting, boolean masking, aggregation, matrix multiplication, reshaping, and automated tests.

## Data Setup
Create a NumPY array simulating a small gray scale image i.e. $8 \times 8$ matrix of integers between $0$ and $255$. 

Use: 

```Python
np.random.randint(0, 255, size = ((8, 8)))
```

## Functions to build
1. `normalize(image: np.ndarray) -> np.ndarray` -  Rescale all pixel values to range from 0 to 1. Use this formula: $X' = \frac{(x - min)}{(max - min)}$. This is a broadcasting exercise, using min-max normalization method.
2. `threshold(image: np.ndarray, cutoff: float) -> np.ndarray` - Using boolean indexing, return every pixel above the cutoff as $1$ and every pixel at or below becomes $0$.
3. `row_col_stats(image: np.ndarray) -> dict` - Using axis based aggregation, return a dictionary with the mean of each row and the mean of each column.
4. `flatten_reshape(image: np.ndarray, new_shape: tuple) -> np.ndarray` - Flatten the image, then reshape into new shape, raising a clear error if the total count doesn't match. 

## matrix Operations
Choose and two matrix and compute and operation, verify by hand first.

## Tests
Write a `test_image_analyzer.py` file with at least four tests covering the functions aboves.

## Summary
In `unit-2-numpy/milestone/summary.md` write a summary of what the module does, one real design decision you made and why, your matrix multiplication example, and one sentence connecting this work to real image processing in *computer visions*.

## File locations
`
unit-2-numpy/milestone/image_analyzer.py
unit-2-numpy/milestone/test_image_analyzer.py
unit-2-numpy/milestone/summary.md
unit-2-numpy/milestone/question.md
`

---
All the best, take your time.