# Foundation
## How computers represent numbers
In python, $0.1 + 0.2 \neq 0.3$. This happens because computers stores numbers in a fixed number of bits, and mostly decimal fractions e.g. $0.3$ can not be represented exactly in binary. Instead the computer stores the closest possible approximation, which introduces a tiny rounding error. Because of this, floating-points numbers should never be compared directly using equality (`==`). Instead, the correct way is to check whether two numbers are close enough withing a small torelance:  

```python
abs(c - 0.3) < 1e-9

# Professinally, using NumPy
np.isclose(c, 0.3)
```