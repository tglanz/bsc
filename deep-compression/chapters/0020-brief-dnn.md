# A brief on deep neural networks

In this section, we will discuss the basics of neural networks to provide sufficient understanding and definitions which will be used throughout this work.

The book "Artificial Intelligence, A modern approach" gives a general view of neural networks: A neural network is a **computation graph** with directed edges that indicate data flow and 3 different node types:

- Input nodes, which contain the input of the network.
- Weight nodes, which contain the weights of the networks and are tuned during **learning**.
- Operation nodes, perform some computation on the data from its incoming edges and produce the result to its outgoing edges.

![Neural network computation graph. Source: "Artificial intelligence, A modern approach"](assets/ann-computation-graph.png){width=50%}

We use the formalization in [[P2; 2.1, 2.2]](#ref-p2) to denote and differentiate neural network architecture from models - A neural network *architecture* $f(x, \cdot)$ is the fixed set of operations to be performed on the input $x$ and its weights. A neural network *model* $f(x, W)$ is a parameterization of $f(x, \cdot)$ with specific parameters $W$. 

## Learning

Let $f(x, W)$ be a neural network model.

Given the **training set** of input-output pairs $(x, y)$ we can define a **loss function** that measures the distance from $f$'s prediction for a given input to the real output:

$$
    Loss(y, f(x, W))
$$

For example, we can use the **squared distance** $L_2(a, b) = (a - b)^2$ as a metric and have the loss function

$$
    Loss(y, f(x, W)) = L_2(y, f(x, W)) = (y - f(x, W))^2
$$

**Training** a neural network is the process of minimizing the loss function by tuning the weights, which there are many algorithms for.

**Backpropagation** is the prevalent algorithm to tune the weights of a neural network to minimize a loss function. The algorithm works by iteratively feeding $f$ with a sample from the training set and then computing the gradient of the loss function for each of the weights. For each of the weights, tune the weight by performing a single step in the direction of the respective gradient.

The step size is linearly dependent on a learning parameter called the **learning rate**. The learning rate isn't always constant during the entire learning phase and can be changed according to the specific algorithm in use. For example, an algorithm can choose to have set a learning rate $\alpha(k) = \frac{1}{k}$ where $k$ is the current learning iteration.

A general learning algorithm might look similar to the following:

\begin{algorithm}[H]
\DontPrintSemicolon
\SetAlgoLined
\SetKwInOut{Input}{Input}\SetKwInOut{Output}{Output}
\Input{Neural Network $f$, weights $W$ and a training sample (x, y)}
\BlankLine
\While{keepLearning}{
    \ForEach{w $\in$ W}{
        w \gets w - $\alpha \frac{\partial}{\partial w}Loss(y, f(x, W))$
    }
}
\caption{Backpropogation}
\end{algorithm}

The algorithm above is straightforward - As long as we want to keep learning we iterate over each parameter and adjust it according to the gradient. Concretely, different implementations use different conditions for "keep learning" such as the number of iterations or achieved accuracy.

## Matrices

The **sparsity** of a matrix $A = (a_{ij}) \in \mathbb{R}^{n \times m}$ is the ratio between the zero elements and the size of the matrix. Formally

$$
  sparsity(A) = \frac{|\{a_{ij} : a_{ij} = 0, 0 \leq i \leq n, 0 \leq j \leq m \}|}{n \cdot m}
$$

Throughout the work, we will use the term **sparse matrix** somewhat freely to indicate that the matrix has a notable sparsity.

The **elementwise product** of two equally sized matrices $(a_{ij})$ and $(b_{ij})$ is denoted and defined by

$$
    (a_{ij}) \odot (b_{ij}) := ((a_{ij} \cdot b_{ij})_{ij})
$$

## Common Operations

Deep neural networks are composed of many operations, also known as layers.

There are many operations and many variants of them - We will not list them here as this is out of the scope of this work but we will list the operations used throughout this work.

### Fully Connected Layer

A **fully connected** layer is a layer that connects each of the input neurons $x = (x_j)_{1}^{n}$ to each of the output neurons $y = (y_i)_{1}^{m}$ with connections $W = (w_{ij}) \in \mathbb{R}^{m \times n}$ such that $w_{ij}$ connects input $x_j$ with output $y_i$. We also assign a **bias** $b \in \mathbb{R}^{m}$ to the output. For illustration, refer to Figure \ref{fully-connected-layer}.

The value assigned to an output $y_i$ is the sum of the products $x_j \cdot w_{ij}$ and the bias $b_i$.

In vector form, given $W_i$ is the notation of the $i-th$ row in $W$, we write:

$$
    y_i = W_ix + b_i
$$

In matrix form, we write

$$
    y = Wx + b   
$$

The connections $W$ are known as the _weights_ or the _parameters_.

A **sparsely connected** layer is a fully connected layer with a sparse weights matrix.

![Fully connected layer\label{fully-connected-layer}](assets/diagrams-fully-connected.drawio.png){width=50%}

![Sparsely connected layer](assets/diagrams-sparsely-connected.drawio.png){width=50%}

### ReLU

The ReLU is an activation function (i.e. non-linear operation) and is defined by:

$$
    ReLU(x) := \max(0, x)
$$