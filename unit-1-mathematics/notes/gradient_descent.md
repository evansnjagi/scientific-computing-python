# Gradient and Gradient descent

## Gradient
$$
    \nabla f(x, y) = \left(\frac{df}{dx}, \frac{df}{dx}\right)
$$

## Gradient descent
$$
    x = x - \alpha \cdot \frac{df}{dx}
$$

$$
    y = y - \alpha \cdot \frac{df}{dy}
$$

To get local minimum, repeat that process many steps. 

## Machine learning application
In neural network, gradient descent is used to minimize the loss function by updating weight in the direction of less error.