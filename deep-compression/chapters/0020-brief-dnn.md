# A brief on deep neural networks

In this section, we will discuss the basics of neural networks to provide sufficient understanding and definitions which will be used throughout this work.

The book "Artificial Intelligence, A modern approach" gives a general view of neural networks: A neural network is a **computation graph** with directed edges that indicate data flow and 3 different node types:

- Input nodes, which contain the input of the network.
- Weight nodes, which contain the weights of the networks and are tuned during **learning**.
- Operation nodes, perform some computation on the data from its incoming edges and produce the result to its outgoing edges.

![Neural network computation graph. Source: "Artificial intelligence, A modern approach"](assets/ann-computation-graph.png){width=50%}

We use the formalization in [[P2; 2.1, 2.2]](#ref-p2) to denote and differentiate neural network architecture from models - A neural network *architecture* $f(x, \cdot)$ is the fixed set of operations to be performed on the input $x$ and its weights. A neural network *model* $f(x, W)$ is a parameterization of $f(x, \cdot)$ with specific parameters $W$. 

## Learning

The process of learning includes the following:

A randomly initialized model $f(x, W)$.

A **Training Set** consisting of $n$ tuples $(x_i, y_i)$ of the input $x_i$ and its corresponding **Label**.

A **Loss Function** $L(y, y')$ is a function that provides a distance between a Label $y$ to an inferred output $y' = f(x, W)$.

The purpose of the learning process is to minimize $L$ on average over the whole training set.

Analytically, if $L$ is differentiable, we can compute its derivatives for each of $w \in W$. In multivariable calculus, the vector of those derivatives is known as the **Gradient**, which is an operator that is notated using $\nabla$. Again, from multivariable calculus, it is known that the Gradient at some point $W$ points to the direction of the steepest ascent. Therefore, if we would like to make small adjustments to the parameters $W$ such that they will minimize $L$, we can adjust them in a small step to the negative direction of $\nabla L$. The size of the small step is determined by the **Learning Rate** parameter which we will notate using $\mu$.

This approach leads us to the basic learning algorithm, in its most basic variant, known as the **Gradient Descent**:

- As long as the average loss is higher than some desired accuracy or we exceeded the maximum number of iterations (a.k.a **Epochs**):
  - For each training set sample $(x, y)$
    - **Forward Pass**: Compute the output $y' = f(x, W)$ 
    - **Backward Pass**: Compute the gradient $\nabla L(y, y')$ and adjust the weights by setting $W \leftarrow W - \mu \nabla L(y, y')$

To save computation power, the **Batch Gradient Descent** variant has been proposed. In this variant, the loss function is modified to apply to a set of samples. Then, instead of applying it sample by sample, we apply it to the entire training set at once.

Another proposed variant is the **Mini-Batch Gradient Descent**. We predefine partitions of the training sets which we call mini-batches. Then, we apply the process of the forward and backward passes for each mini-batch independently.

The **Stochastic Gradient Descent (SGD)** is the most common variant of Gradient Descent. SGD works by randomly selecting samples at each epoch, shuffling them and performing the forward and backward passes on them. The selected samples are also known as mini-batches.

SGD aims to increase generalization by introducing randomness to the process at the cost of learning time.