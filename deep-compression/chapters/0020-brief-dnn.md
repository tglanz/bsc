## A brief on deep neural networks

In this section we briefly review the basics of neural networks and define common terms used throughout this paper.

### Matrices

The **sparsity** of a matrix $A = (a_{ij}) \in \mathbb{R}^{n \times m}$ is the ratio between the zero elements and the size of the matrix. Formally

$$
  sparsity(A) = \frac{|\{a_{ij} : a_{ij} = 0, 0 \leq i \leq n, 0 \leq j \leq m \}|}{n \cdot m}
$$

Throughout the work, we will use the term **sparse matrix** somewhat freely to indicate that the matrix has a notable sparsity.

The **elementwise product** of two equally sized matrices $(a_{ij})$ and $(b_{ij})$ is denoted and defined by

$$
    (a_{ij}) \odot (b_{ij}) := ((a_{ij} \cdot b_{ij})_{ij})
$$

### Fully Connected Layer

A **fully connected** layer is a layer which connects each of the input neurons $x = (x_i)_{1}^{n}$ to each of the output neurons $y = (y_j)_{1}^{m}$ with connections $W = (w_{ij}) \in \mathbb{R}^{n \times m}$ such that $w_{ij}$ connects input $x_i$ with output $y_j$. A bias $b \in \mathbb{R}^{m}$ is added to the output $y$. In matrix form, we write

$$
    y = xW + b   
$$

The connections $W$ are known as the _weights_ or the _parameters_.

A **sparsely connected** layer is a fully connected layer with a sparse weights matrix.

![Fully connected layer](assets/diagrams-fully-connected.drawio.png){width=50%}

![Sparesly connected layer](assets/diagrams-sparsely-connected.drawio.png){width=50%}

### Neural Networks

In this section we will discuss the basics of neural networks to provide sufficient understanding and definitions which will be used throughout this work.

The book "Artificial intelligence, A modern approach" gives a general perspective for neural networks: A neural network is a **computation graph** with directed edges that indicated data flow and 3 different node types:

- Input nodes which contain the input of the network.
- Weight nodes which contain the weights of the networks and are tuned during **learning**.
- Operation nodes that perform some computation on the data from it's incoming edges and produce the result to its outgoing edges.

![Neural network computation graph from "Artificial intelligence, A modern approach"](assets/ann-computation-graph.png){width=50%}

We use the formalization in [[7; 2.1 ,2.2]](#ref-7) to denote differentiate neural network architecture from models - A neural network *architecture* $f(x, \cdot)$ is the fixed set of operations to be performed on the input $x$ and it's weights. A neural network *model* $f(x, W)$ is a parameterization of $f(x, \cdot)$ with specific parameters $W$. 

### Learning

Let $f(x, W)$ be a neural network model.

Given the **training set** of input output pairs $(x, y)$ we can define a **loss function** that measures a distance from $f$'s prediction for a given input to the real output:

$$
    Loss(y, f(x, W))
$$

For example, we can use the **squared distance** $L_2(a, b) = (a - b)^2$ as a metric and have the loss function

$$
    Loss(y, f(x, W)) = L_2(y, f(x, W)) = (y - f(x, W))^2
$$

**Training** a neural network is the process of minimizing the loss function by tuning the weights which there are many algorithms for.

**Backpropogation** is the prevalent algorithm to tune the weights of a neural networks in order to minimize a loss function. The algorithm works by iteratively feeding $f$ with sample from the training set and then compute the gradient of the loss function with respect to each of the weights. For each of the weights, tune the weight by performing a single step in the direction of the respected gradient.

The step size is linearly dependant on a learning parameter called the **learning rate**. The learning rate isn't always constant during the entire learning phase and can be changed according to the specific algorithm in use. For example, an algorithm can choose to have set a learning rate $\alpha(k) = \frac{1}{k}$ where $k$ is the current learning iteration.

A general learning algorithm might look similiar to the following:

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