# Summary
## Function
Working function for this project is:

$$
    f(x, y) = x^2 + y^2 + 4x - 6y
$$
## Partial derivatives
The partial derivative of $f(x, y)$ is derived as follows:

1. Partial derivative with respect to x:

$$
\frac{df}{dx}f(x, y) =  2x + 4
$$
Solving for the partial derivative of f(x, y) with respect to x at x = 1, we get:

$$
    \frac{df}{dx} = 2 \cdot 1 + 4 = 6
$$

2. Partial derivative with respect to y:

$$
    \frac{df}{dy}f(x, y) = 2y - 6
$$
Solving for the partial derivative at point of y = 1.

$$
    \frac{df}{dy} = 2 \cdot 1 - 6 = -4
$$

In `gradient_descent` script, The two partial derivative functions are approximated using the numerical derivative method.

If we solve the partial derivative of f(x, y) with respect to x, we expect the solution to be approximately close to $6$, Here is the output `6.000010000040134`. Using `np.isclose()` to compare the predicted derivative and the computed one, we get a TRUE, meaning that the two are technically equal.

The next natural step is computing the partial derivative of f(x, y) with respect to y. Using *power rule*, the solution to the partial derivative is $-4$, evaluating at $y = 1$. Here is Python output: `-3.999990000025377`. Checking closeness of Python computed approximation with the algebraic derived one, they are equal.
## Gradient
Gradient gives the point(x, y) to the STEEPEST direction. From the two-variable function above, we compute gradient algebraically by setting the derived equation, from the function, to zero and solving for respective values of x or y.

$$
    \nabla f(x, y) = \left(\frac{df}{dx}, \frac{df}{dy}\right) \\
    \nabla f(1, 1) = (6, -4)
$$

In Python, the `gradient()` function return this output: `(6.000010000040134, -3.999990000025377)` which implies that $\nabla f(1, 1) = (6.000010000040134, -3.999990000025377)$. 
## Gradient descent
Gradient descent is the local minimum values of x and y. We can compute the local minimum using algebraic technique, simply setting the derived equation to zero and solving it.

$$
    2x + 4 = 0 \quad \cdots \text{eq 1}\\
    x = \frac{-4}{2} = -2 \\
    2y - 6 = 0 \quad \cdots \text{eq 2}\\
    y = \frac{6}{2} = 3\\
$$
The local minimum point is at x = -2, and y = 3.  To compute gradient descent, the following steps are followed:

1. Start at a random point. In `gradient_descent.py` the starting point is at $(5, 8)$.
2. Compute gradient at that point

$$
\nabla f(5, 8) = (14, 10)
$$
3. Step the opposite direction

$$
    x = x -\alpha \frac{df}{dx} \\
    y = y - \alpha \frac{df}{dy}
$$

With a learning rate of $0.1$ the new x or y value will be:
$$
    x = 5 - 0.1 \cdot 14 = 3.6\\
    y = 8 - 0.1 \cdot 10 = 7
$$

4. Repeat step 3 many times. In `gradient_descent` script, we are repeating this process 100 times, although the step is optional, one can choose as many steps to iterate as they want.

**Experiment**

1. Learning rate near threshold `0.99`- The output is `(-1.0716674491025893, 3.663093440283477)`.  This learning rate is near threshold, the algorithm is oscillating rather than just moving slowly.
2. Few steps, `50` - Using few steps the convergence will not be achieve, we need more. Here is the comparison: 

Few steps output: `-1.9999050926305981, 3.0000663624649633)`

Many steps output: `(-2.000004999913045, 2.999995000100057)`

## ML connection

In neural network, gradient descent is used to minimize the loss function by repeatedly adjusting the model weights that reduces the error.