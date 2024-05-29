## Gradient Calculation for 2-split
We will use superscripts to denote the vectors as indicated by the illustration. For example $X^2$ is $X2$ in the diagram. We will use subscript to denote the specific element of the matrix/vector. For example, $X^2_{3,4}$ is the element at the 3rd row and the 4th column of the matrix $X^2$.

By definition of the _ReLU_ layer $ReLU(x) = \max \{x, 0 \}$, and because _ReLU_ is a scalar function we get the derivatives

$$
    \frac{\partial Y^i}{\partial Z^i} = \delta (Z^i)
$$

where $\delta$ is an elementwise function s.t $\delta(V) = [\delta(v_i)]$ and the scalar $\delta(v_i)$ is defined by

$$
\delta(v_i) = \begin{cases}
    1 & v_i > 0 \\
    0 & otherwise
\end{cases}
$$

By definition of the _Linear_ layer $Z = WX + b$ (Remember that $X$ and $Z$ are vectors)

$$
    Z_i = \sum_{k=1}W_{ik} X_k + b_i
$$

Hence,

$$
    \frac{\partial Z_i}{\partial W_{jk}} = \begin{cases}
        X_k & i = j \\
        0 & otherwise
    \end{cases}
    ~~~;~~~
    \frac{\partial Z_i}{\partial b_j} = \begin{cases}
        1 & i = j \\
        0 & otherwise
    \end{cases}
$$

By the chain rule

$$
    \frac{\partial C}{\partial W_{ij}} =
    \frac{\partial C}{\partial Y} \begin{bmatrix}
        \frac{\partial Y^1}{\partial Z^1} \cdot \frac{\partial Z^1}{\partial W_{ij}} \\
        \frac{\partial Y^2}{\partial Z^2} \cdot \frac{\partial Z^2}{\partial W_{ij}} \\
    \end{bmatrix} = 
    \frac{\partial C}{\partial Y} \begin{bmatrix}
        \delta(Z^1) \cdot \frac{\partial Z^1}{\partial W_{ij}} \\
        \delta(Z^2) \cdot \frac{\partial Z^2}{\partial W_{ij}} \\
    \end{bmatrix} =
    \frac{\partial C}{\partial Y} \begin{bmatrix}
        0 \\
        \vdots \\
        0 \\
        \delta(Z^1_i) X^1_j \\
        0 \\
        \vdots \\
        0 \\
        \delta(Z^2_i) X^2_j \\
        0 \\
        \vdots \\
        0
    \end{bmatrix}
$$

Where the non-zero entries are the $i$ and the $\frac{m}{2} + i$ entries respectively.

Also by the chain rule

$$
    \frac{\partial C}{\partial b_{i}} =
    \frac{\partial C}{\partial Y} \begin{bmatrix}
        \frac{\partial Y^1}{\partial Z^1} \cdot \frac{\partial Z^1}{\partial b_i} \\
        \frac{\partial Y^2}{\partial Z^2} \cdot \frac{\partial Z^2}{\partial b_i} \\
    \end{bmatrix} = 
    \frac{\partial C}{\partial Y} \begin{bmatrix}
        \delta(Z^1) \cdot \frac{\partial Z^1}{\partial b_i} \\
        \delta(Z^2) \cdot \frac{\partial Z^2}{\partial b_i} \\
    \end{bmatrix} =
    \frac{\partial C}{\partial Y} \begin{bmatrix}
        0 \\
        \vdots \\
        0 \\
        \delta(Z^1_i) \\
        0 \\
        \vdots \\
        0 \\
        \delta(Z^2_i) \\
        0 \\
        \vdots \\
        0
    \end{bmatrix}
$$

Where again the non-zero entries are the $i$ and the $\frac{m}{2} + i$ entries respectively.

## Extension of the Gradient for 4-split

For an input the is split to 4 components $X_1, X_2, X_3$ and $X_4$ we get

$$
    \frac{\partial C}{\partial W_{ij}} =
    \frac{\partial C}{\partial Y} \begin{bmatrix}
        \frac{\partial Y^1}{\partial Z^1} \cdot \frac{\partial Z^1}{\partial W_{ij}} \\
        \frac{\partial Y^2}{\partial Z^2} \cdot \frac{\partial Z^2}{\partial W_{ij}} \\
    \end{bmatrix} = 
    \frac{\partial C}{\partial Y} \begin{bmatrix}
        \delta(Z^1) \cdot \frac{\partial Z^1}{\partial W_{ij}} \\
        \delta(Z^2) \cdot \frac{\partial Z^2}{\partial W_{ij}} \\
    \end{bmatrix} =
    \frac{\partial C}{\partial Y} \begin{bmatrix}
        0 \\
        \vdots \\
        0 \\
        \delta(Z^1_i) X^1_j \\
        0 \\
        \vdots \\
        0 \\
        \delta(Z^2_i) X^2_j \\
        0 \\
        \vdots \\
        0 \\
        \delta(Z^3_i) X^3_j \\
        0 \\
        \vdots \\
        0 \\
        \delta(Z^4_i) X^4_j \\
        0 \\
        \vdots \\
        0
    \end{bmatrix}
~~~;~~~
    \frac{\partial C}{\partial b_{i}} =
    \frac{\partial C}{\partial Y} \begin{bmatrix}
        \frac{\partial Y^1}{\partial Z^1} \cdot \frac{\partial Z^1}{\partial b_i} \\
        \frac{\partial Y^2}{\partial Z^2} \cdot \frac{\partial Z^2}{\partial b_i} \\
    \end{bmatrix} = 
    \frac{\partial C}{\partial Y} \begin{bmatrix}
        \delta(Z^1) \cdot \frac{\partial Z^1}{\partial b_i} \\
        \delta(Z^2) \cdot \frac{\partial Z^2}{\partial b_i} \\
    \end{bmatrix} =
    \frac{\partial C}{\partial Y} \begin{bmatrix}
        0 \\
        \vdots \\
        0 \\
        \delta(Z^1_i) \\
        0 \\
        \vdots \\
        0 \\
        \delta(Z^2_i) \\
        0 \\
        \vdots \\
        0 \\
        \delta(Z^3_i) \\
        0 \\
        \vdots \\
        0 \\
        \delta(Z^4_i) \\
        0 \\
        \vdots \\
        0
    \end{bmatrix}
$$