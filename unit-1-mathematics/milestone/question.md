# Mathematics
> The goal is to connect everything, derivatives, partial derivative, gradient and gradient descent. On this problem, find the solution by yourself without step by step guidance.

## The Task
1. Choose a two variable function to minimize.
2. By hand, or in your notebook, compute $\frac{df}{dy}$ and $\frac{df}{dx}$ using the function you choose.
3. In `unit-1-mathematics/milestone/gradient_descent_milestone.py` folder, Write a script that computes partial derivative of the function, $\frac{df}{dx}$ and $\frac{df}{dy}$, reuse gradient and gradient descent functions, don't rewrite them. 
4. Confirm the results match your hand-predicted minimum. Use `np.isclose`.
5. One experiment of your own i.e. learning rate near the instability threshold, very few steps vs. many, starting very far away, etc. Explain in writing what you observe.   
6. Write a summary `unit-1-mathematics/milestone/summary.md`. Write about the functions, hand-derived gradient, experiment and its result, One sentence connecting everything to how ML training works.