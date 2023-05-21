# Pruning

## Motivation

**Pruning** is a compression technique to reduce the size of a model by forcefully removing part of its parameters and/or neurons.

Oxford's definitions for pruning are both very fitting

> The activity of cutting off some of the branches from a tree, bush, etc. so that it will grow better and stronger

Pruning has been shown to increase their robustness by reducing the tendency to overfit [[1]](#ref-1) and increasing generalization.

> The act of making something smaller by removing parts; the act of cutting out parts of something

More importantly, pruning reduces the size of the model. Smaller models have multiple benefits

**Fewer computations**

The fewer connections the fewer the theoretical amount of computations required - Fewer computations lead to faster inferences.

**Smaller memory footprint**

A smaller memory footprint allows for larger parts of the network to fit in memory, reducing storage/device/network I/O and increasing GPU utilization.

**Less storage footprint**

Smaller models can fit in resource-limited devices such as Mobile devices, Chips, FPGAs etc.

**Smaller power consumption**

Smaller models require less memory access, reducing the power consumption required to process the model (either in inference or training phases). More efficient power consumption can drastically reduce costs (especially for huge models running in data centers) and increase the life duration of smaller devices.

---

Pruning, however, does not come without its merits. Firstly, software and hardware must adapt their logic to achieve full utilization - the pruned parameters/neurons should be tracked and skipped during computations efficiently. Secondly, and perhaps most important, pruning a network is the same as deleting part of its information and can potentially lead to accuracy loss. 

Now that we have an idea of **why** pruning might be beneficial, in the next sections we will discuss **what** is pruning and **how** it's being performed.

## What is pruning?

We define **pruning** to be a function, mapping a model $f(x, W)$ to a _pruned_ model $f(x, M \odot W')$ where $W'$ is some set of parameters (different from $W$) and $M \in \{0, 1\}^{|W'|}$ is a binary mask.

Most of the pruning methods are variants of the general algorithm outline described in [[1; 3]](#ref-1).

1. Learn the model parameters via _training_ as normal.
2. Prune the parameters having a lower value than some predefined threshold.
3. Retrain the network, to learn the final parameters.

See Figure \ref{pruning-outline} for illustration.

![Pruning algorithm outline. Source: "Learning both Weights and Connections for Efficient Neural Networks"\label{pruning-outline}](assets/pruning-algorithm-outline.png){width=90%}

## Pruning Methods

Different pruning methods differ in the following criteria

### Structure

**Un-Structured Pruning** is the method of pruning weights with no constraint regarding their relation to each other.

Usually, the hardware performs computations in groups. For example, a unit of computation in a matrix multiplication can be an entire row, in convolution operation a group can be the filters' channels - It depends on the specific algorithm and the hardware. Therefore, without specific design and programming, the hardware cannot achieve proper utilization of the network since they are optimized for dense networks[[10]](#ref-10).

**Structured Pruning** is the method of pruning structural groups of parameters. An example of such groups could be entire rows or columns of a matrix, convolutional filters, channels, etc...

Structurally pruned networks are relatively straightforward to accelerate by hardware - Simply ignore the groups that were pruned.

### Scoring (a.k.a Mask Criteria)

Each parameter is assigned a score which is used to prioritize which parameter to prune.

**Local Scoring** is when we compare scores of parameters only within their substructure (e.g. Layer, Filter, etc...).

**Global Scoring** is when we compare scores of parameters across the entire network.

There are many ways to assign scores to parameters

- According to some random distribution
- According to the magnitude of the parameter after training
- According to the magnitude of the parameter at initialization
- According to the change in magnitude of the parameter before and after initialization

The standard scoring method is to use the magnitude of the parameter after training which is known as **Magnitude Pruning**.

### Scheduling

Scheduling determines whether we prune to a target sparsity at once or we break the process into iterations.

**One-Shot Pruning** is the method of pruning to a target sparsity at once.

**Iterative Pruning** is the method of pruning to a target sparsity in iterations, each iteration prunes a smaller amount of parameters gradually increasing the sparsity.

### Retraining Methods

According to the pruning algorithm, we retrain the network again after we have pruned the parameters. Retraining methods refer to what adjustments should we make to the network state before we retrain it.

**Fine Tuning** is the method of keeping the parameters as they are.

This is the most widely used method and is shown to achieve high accuracy and is somewhat intuitive - After we remove some parameters we delete part of the network information effectively reducing the accuracy. To increase the accuracy all we need is to keep training.

**Learning Rate Rewinding** is the method of re-initializing the learning rate, keeping the parameters as is and then retraining the network.

The idea behind this method is to allow for bigger _error margins_ during the stochastic gradient descent process (TODO: describe SDG in the brief to NNs).

**Weight Rewinding** is the method of re-initializing the weights to their values at initialization before training and pruning - Only then retrain the network.

This method was suggested by *Jonathan Frankle* in his paper *The Lottery Ticket Hypothesis* which is a topic that we will discuss later. In his paper he shows he can achieve high accuracies using this method which suggests that the sparse, sub-networks could have been initialized from the start to achieve the same accuracies as the original networks - This hypothesis was revolutionary at the time and initiated large amounts of further research.

From a practical standpoint, it is not a storage-efficient technique because we need to keep the initial state of
all of the parameters. But again, the theoretical implications are great.

## Evaluation

We prune networks to be more efficient in the following metrics:

- Accuracy
- Storage
- Memory
- Power
- Computations

As discussed in the previous sections there are many ways in which we can prune a given network. Different methods affect different metrics. Therefore, there is no "one technique is better than the other" scenario - we need to prioritize the metrics according to our scenario and choose the techniques that will optimize this choice.

The main statement by Jonathan Frankle in [[7]](#ref-7) is that the literature is currently fragmented - Pruning research papers compare slightly different metrics, slightly different architectures and different datasets. This leads to difficulties in asserting different techniques.

However, some results are consistent - Pruning keeps accuracy relatively high up to a certain point in which the accuracy falls off drastically. In the paper, Frankle used ShrinkWeb - an OSS he implemented to consistently evaluate network pruning techniques. He performed pruning experiments over different architectures using different scoring methods.

Below are his results. 

The Figures \ref{pruning-results-1}, \ref{pruning-results-2}, \ref{pruning-results-3}, \ref{pruning-results-4} and \ref{pruning-results-5} are split into architecture+dataset and a metric to correlate with the measured accuracy.

The metrics are compression ratio and theoretical speedup

- **Compression Ratio** is the ratio between the size of the original network and the pruned network.

- **Theoretical Speedup** is the ratio between the number of MACs (Multiply And aCcumulate) in the original network and the pruned network.
  - Interestingly it is named _theoretical speedup_, not _speedup_ due to the reasoning we discussed regarding proper hardware/software utilization.

Furthermore, the diagrams show the trends of evaluations using the following different scoring methods

- **Global weight.** Parameters are scored based on their magnitude and compared globally.
- **Layer weight.** Parameters are scored based on their magnitude and compared locally.
- **Global gradient.** Parameters are scored based on their gradient's magnitude and compared globally.
- **Local gradient.** Parameters are scored based on their gradient's magnitude and compared locally.
- **Random.** Parameters are scored according to a random distribution.

We can gather the following conclusions from the results:

1. Random scoring doesn't work well - The accuracy falls off immediately in all architectures and datasets.

2. Weight magnitudes (Globally and Locally) mostly outperform gradient magnitudes (Globally and Locally).

![Frankle's results for the Resnet18 architecture, using the Imagenet dataset. Source: What is the State of Neural Network Pruning?\label{pruning-results-1}](assets/pruning-results-resnet18-imagenet.png)

![Frankle's results for the Resnet20 architecture, using the Cifar dataset. Source: What is the State of Neural Network Pruning?\label{pruning-results-2}](assets/pruning-results-resnet20-cifar.png)

![Frankle's results for the Resnet56 architecture, using the Cifar dataset. Source: What is the State of Neural Network Pruning?\label{pruning-results-3}](assets/pruning-results-resnet56-cifar.png)

![Frankle's results for the Resnet110 architecture, using the Cifar dataset. Source: What is the State of Neural Network Pruning?\label{pruning-results-4}](assets/pruning-results-resnet110-cifar.png)

![Frankle's results for the VGG architecture, using the Cifar dataset. Source: What is the State of Neural Network Pruning?\label{pruning-results-5}](assets/pruning-results-vgg-cifar.png)