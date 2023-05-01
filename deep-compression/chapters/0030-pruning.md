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

![Pruning algorithm outline](assets/pruning-algorithm-outline.png){width=90%}

## Pruning Methods

Different pruning methods differ in the following criteria

### Structure

**Un-Structured Pruning** is the method of pruning weights with no constraint regarding their relation to each other.

Usually, hardware perform computations in groups. For example, a unit of computation in a matrix multiplication can be an entire row, in convolution operation a group can be the filters' channels - It depends on the specific algorithm and the hardware. Therefore, without specific design and programming, the hardware cannot achieve proper utilization of the network since they are optimized for dense networks[[10]](#ref-10).

**Structured Pruning** is the method of pruning groups of neurons or weights - Group of weights could be rows or columns of a matrix, convolutional filters, channels, etc... 

Structurely pruned networks are relatively straight-forward to accelerate by hardware - Simply ignore the groups that was pruned.

### Scoring (a.k.a Mask Criteria)

Each parameter is assigned a score which is used to prioritize which parameter to prune.

**Local Scoring** is when we compare scores of parameters only withing their substructure (e.g; Layer, Filter, etc...).

**Global Scoring** is when we compare scores of paramaters accross the entire network.

There are many ways to assign scores to parameters

- According to some random distribution
- According to the magnitude of the parameter after training
- According to the magnitude of the parameter at initialization
- According to the change in magnitude of the parameter before and after initialization

The standard scoring method is to use the magnitude of the parameter after training which is known as **Magnitude Pruning**.

### Scheduling

Scheduling determines whether we prune to a target sparsity at once or we break the process into iterations.

**One-Shot Pruning** is the method of pruning to a target sparsity at once.

**Iterative Pruning** is the method of pruning to a target sparsity in iterations, each iteration prune a smaller amount of parameters.

### Retraining Methods

Tuning refers to how we prepare the network before we retrain it, after we have pruned it. 

**Fine Tuning** is the method of keeping the parameters as is (after pruning) and retraining the network.

This is the most widely used method and is shown to achieve high accuracies.

**Learning Rate Rewinding** is the method of re-initializing the learning rate, keeping the parameters as is and then retraining the network.

The idea behind this method is to allow for bigger _error margins_ during the stochastic gradient descent process (TODO: describe SDG in the brief to NNs).

**Weight Rewinding** is the method of re-initializing the weights to their values at initialization prior to training and pruning - Only then retrain the network.

This method was suggested by *Jonathan Frankle* in his paper *The Lottery Ticket Hypothesis* which is a topic we will discussed later. In his paper he shows he can achieve high accuracies using this method which suggests that the sparse, sub-networks could have been initialized from the start to achieve the same accuracies as the original networks - This hypothesis was revolutionary at the time and initiated large amounts of further research.

## Leftover from the outline (will be removed)

The Lottery Ticket Hypothesis and the following tickets research, compare different pruning approaches. Therefore, we need to fully grasp pruning before discussing the hypothesis.

- What is pruning?
  - Impact on Inference vs. Impact on Training
  - Sparsity in hardware
- Definitions
- Techniques
  - Weights, Neurons, Layers
  - Structured vs Unstructured
  - One Shot vs Iterative
  - Rewind

