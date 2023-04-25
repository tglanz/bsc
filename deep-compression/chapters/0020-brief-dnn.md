## A brief on deep neural networks

In this section we briefly review the basics of neural networks and define common terms used throughout this paper.

### Matrix Operations

For every two matrices of same size $A=(a_{ij})$ and $B=(b_{ij})$, the elementwise product $A \odot B$ is defined to be the matrix $C = (a_{ij} \cdot b_{ij})$.

### Fully Connected Layer

A **fully connected** layer is a layer the connected each of the input neurons $x = (x_i)_{1}^{n}$ to each output neuron $y = (y_j)_{1}^{m}$ with connections $W = (w_{ij}) \in \mathbb{R}^{n \times m}$ such that $w_{ij}$ connects input $x_i$ with output $y_j$. Every output neuron $y_j$ is assigned a bias $b_j$ and is computed by $\sum_{i=1}^{n}{x_i y_j} + b_j$. In matrix form, we write

$$
    y = xW + b   
$$

A weights matrix $W$ is said to be sparse if it contain zeroes. Given $s$ is the number of zero elements, we say that $\frac{s}{n \cdot m}$ is its **sparsity**.

A **sparsely connected** layer is a fully connected layer with part of its weights remove.

![Fully connected layer](assets/diagrams-fully-connected.drawio.png){width=50%}

![Sparesly connected layer](assets/diagrams-sparsely-connected.drawio.png){width=50%}