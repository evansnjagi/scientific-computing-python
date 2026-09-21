# A Summary of Image Analyzer
> The goal of this project is to use everything we have been learning from creating an array, broadcasting, aggregation, boolean masking, reshaping and automated tests. 

## Project Setup
The following is the project tree used: 

```bash
unit-2-numpy/milestone/
├── image_analyzer.py
├── question.md
├── summary.md
└── test_image_analyzer.py
```

## Functions
Each function written, has one single responsibility with a clear explained purpose in DOCSTRING and some TYPE HINTS. 

### 1. `data_setup(low, high, tuple[int, int]) -> np.ndarray` 
This function takes in three parameters and return a 2D array of pixels between *low* and *high*. The array is a simulation of a gray scale image. By default, the image should be of size ($8, 8$). 

The image pixels i.e. $A_{ij} \in \mathbb{R}$ (image array/matrix $\mathbf{A}$), is chosen randomly between LOW and HIGH values provided as parameters in the function definition.

Example:
```python
A = data_setup(0, 255, (8, 8))
```
$$
\mathbf{A} = 
\begin{pmatrix}
35 & 142 & 103 & 219 & 99 & 174 & 156 & 139 \\
6 & 54 & 151 & 2 & 65 & 175 & 158 & 24 \\
61 & 148 & 76 & 133 & 215 & 223 & 139 & 163 \\
70 & 246 & 245 & 214 & 138 & 245 & 24 & 144\\
144 & 27 & 22 & 87 & 93 & 75 & 133 & 52 \\
204 & 200 & 87 & 142 & 251 &  7 & 130 & 248 \\
82 & 249 & 72 & 86 & 187 & 97 & 112 & 46 \\
157 & 49 & 69 & 113 & 169 & 62 & 241 & 151 \\
\end{pmatrix}
$$
The random matrix above is created. We can perform any computation we want using the matrix $\mathbf{A}$.

Once the matrix is formed, several tests are performed asserting correct random entries $\forall A_{ij} \text{ such that } 0 \le A_{ij} \le 255$

The first test is to check the size of the ($8 \times 8$) matrix. The shape should align with our expected matrix $\mathbf{A} \in \mathbb{R}^{8 \times 8}$. 

Following that is the second and third test checking matrix addition and multiplication.

In matrix addition, the test expect that adition of two matrix is commutative i.e. $\mathbf{A} + \mathbf{B} = \mathbf{B} + \mathbf{A}$, here is an example: 
$$
\mathbf{A} = 
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix} \qquad 
\mathbf{B} = 
\begin{pmatrix}
4 & 3 \\
2 & 1
\end{pmatrix}
$$

$$
\mathbf{A} + \mathbf{B} = 
\begin{pmatrix}
(1 + 4) & (2 + 3) \\
(3 + 2) & (4 + 1)
\end{pmatrix} = 
\begin{pmatrix}
5 & 5 \\
5 & 5
\end{pmatrix}
$$
likewise,
$$
\mathbf{B} + \mathbf{A} = 
\begin{pmatrix}
(4 + 1) & (3 + 2)\\
(2 + 3) & (1 + 4)
\end{pmatrix} = 
\begin{pmatrix}
5 & 5 \\
5 & 5
\end{pmatrix}
$$
The two matrix are equal, they are the same $\blacksquare$.

For matrix multiplication, we expect $\mathbf{A} \mathbf{B} \neq \mathbf{B} \mathbf{A}$. Confirming, mathematically, using the two matrix above:
$$
\mathbf{A} \mathbf{B} = 
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}
\begin{pmatrix}
4 & 3 \\
2 & 1
\end{pmatrix} \\
\implies
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix} 
\begin{pmatrix}
4 \\ 2
\end{pmatrix} = 
\begin{pmatrix}
4 + 4 \\
12 + 8
\end{pmatrix}
=
\begin{pmatrix}
8 \\ 20
\end{pmatrix}\\
\implies 
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix} 
\begin{pmatrix}
3 \\ 1
\end{pmatrix} = 
\begin{pmatrix}
3 + 2 \\
9 + 4
\end{pmatrix}
=
\begin{pmatrix}
5 \\
13
\end{pmatrix}\\
\therefore 
\begin{pmatrix}
8 & 5 \\
20 & 13
\end{pmatrix}
$$
next step,
$$
\mathbf{B} \mathbf{A} = 
\begin{pmatrix}
4 & 3 \\
2 & 1
\end{pmatrix}
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix} \\
\implies
\begin{pmatrix}
4 & 3 \\
2 & 1
\end{pmatrix} 
\begin{pmatrix}
1 \\ 3
\end{pmatrix} = 
\begin{pmatrix}
4 + 9 \\
2 + 3
\end{pmatrix}
=
\begin{pmatrix}
13 \\ 5
\end{pmatrix}\\
\implies 
\begin{pmatrix}
4 & 3 \\
2 & 1
\end{pmatrix} 
\begin{pmatrix}
2 \\ 4
\end{pmatrix} = 
\begin{pmatrix}
8 + 12 \\
4 + 4
\end{pmatrix}
=
\begin{pmatrix}
20 \\ 8
\end{pmatrix}\\
\therefore 
\begin{pmatrix}
13 & 20 \\
5 & 8
\end{pmatrix}
$$
The two matrix are not similar. $\blacksquare$

But something is clear, taking transpose of both matrices, the solution will be the same i.e. $(\mathbf{A} \mathbf{B})^{T} = \mathbf{B}^T \mathbf{A}^T$, this is a conceptual followup understanding hence, not tested or proved.
## 2. `normalize(image: np.ndarray) -> np.ndarray`
This function takes in a 2D array, image matrix with pixel entires, and returns a normalized equivalent for each entry.

We will constantly use the image array, $\mathbf{A}$, formed by the `data_setup()`. 

In mathematics, *min-max* normalization is computed using the following formula:

$$
X' = \frac{x - min}{max - min}
$$

Example with $A_{11} = 35$:

From the matrix $\mathbf{A}$, $\text{max} = 2$  and $\text{min} = 251$

Normalizing $A_{11}$
$$
\implies \frac{35 - 2}{251 - 2} = \frac{33}{249} \approx 0.1325
$$

This process is repeated on the remaining entires until the entire matrix is normalized. A min-max normalized matrix should have values between $0$ and $1$.

Following normalization of $\mathbf{A}: 8 \times 8$, is testing. The first test is the normal test. Testing if the normalized matrix is the same as the hand computed one. After that, we test on an edge case scenario. Normalizing a $1 \times 1$ matrix e.g. $\mathbf{A} = \begin{pmatrix} 221 \end{pmatrix}$. Mathematically we would normalize $A_{11}$ as $\frac{221-221}{221-221} = \frac{0}{0}$. In programming, using NumPy, we expect the resultant output to be `np.nan`, because the result is surely an undefined one.

## 3. `threshold(image: np.ndarray, cutoff: int) -> np.ndarray`
This function takes in  image pixels combined together as a matrix, then the function checks if the matrix entry $A_{ij}$ is greater than the cutoff integer value, if true $A_{ij} = 1$ and if the entry is less than the cutoff value then $A_{ij} = 0$.

Let's work out an example. At $A_{11} = 35$ and cutoff = $150$, $A_{11}$ will be returned as a $0$ because 35 is less than 150. 

Using vectorized computation and masking, the entire threshold is computed and a resultant matrix returned.

The last step is always testing if the function is working correctly. In our case, we have three distinct tests to assert that our function is working as intended.

The first test is done with a simple matrix, checking if the output is a 2D array with only zeros and ones.

After the first test, we test with an edge case i.e minimum cutoff which is a 0 or maximum cutoff with a integer value of 255. 

Testing with minimum cutoff, we expect a matrix with every element equals 1, commonly known as a *ones matrix*.

i.e
$$
\mathbf{A} = 
\begin{pmatrix}
1 & 1 & \cdots & 1\\
1 & 1 & \cdots & 1\\
\vdots & \vdots & \ddots & \vdots \\
1 & 1 & \cdots & 1\\
\end{pmatrix}
$$

With the maximum cutoff, the output should be a 2D matrix with all zeros. This kind of a matrix is called a *zero matrix*.

$$
\mathbf{A} = 
\begin{pmatrix}
0 & 0 & \cdots & 0\\
0 & 0 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0\\
\end{pmatrix}
$$

## Design Decison 
In normalization function, we have one major edge case problem. Normalizing a $1 \times 1$ matrix, i.e. $\mathbf{A}\in \mathbb{R}^{1 \times 1}$. 

Mathematically, min-max normalization for a 1D matrix, with one entry, gives an *indeterminate* solution. For example, $\mathbf{A} = \begin{pmatrix} 200 \end{pmatrix}$

$$
\implies X' = \frac{x - \min}{\max - \min}
 \rightarrow\frac{200 - 200}{200 - 200} = \frac{0}{0}
$$

Computing min-max normalization with that same matrix, will surely render an error. NumPy handles this gracefully, returning an important RUNTIME error. 

While testing, we need to catch that particular error so that we know our function is working as intended. 

We can raise an EXCEPTION using normal Python code, but inside a vectorized operation, that part will never be reached. The reson is because, NumPy code does not crash raising an exception error. Where we have $\frac{0}{0}$ computation, it returns `np.nan`. 

To solve this problem, we can raise an exception using `np.errstate` setting `divide = "ignore"` and `invalid = "ignore"`. This solves our problem entirely and we can try catching the error in testing section. 

## ML Connection
Operations like noise reduction, image enhancement, and cropping, using NumPy, are used to prepare raw image pixels so machine learning models can receive clean and reliable data.

