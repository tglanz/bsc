# Pruning

## Motivation

**Pruning** is a compression technique to reduce the size of a model by forcefully removing part of its parameters and/or neurons.

Oxford's definitions for pruning are both very fitting

> The activity of cutting off some of the branches from a tree, bush, etc. so that it will grow better and stronger

Pruning has been shown to increase their robustness by reducing the tendency to overfit [[1]](#ref-1) and increasing generalization.

> The act of making something smaller by removing parts; the act of cutting out parts of something

More importantly, pruning reduces the size of the model. Smaller models have multiple benefits

**Fewer computations**

Fewer computations can lead to faster inferences.

**Smaller memory footprint**

A smaller memory footprint allows for larger parts of the network to fit in memory, reducing storage/device/network I/O and increasing GPU utilization.

**Less storage footprint**

Smaller models can fit in resource-limited devices such as Mobile devices, Chips, FPGAs etc.

**Smaller power consumption**

Smaller models require less memory access, reducing the power consumption required to process the model (either in inference or training phases). More efficient power consumption can drastically reduce costs (especially for huge models running in data centers) and increase the life duration of smaller devices.

---

Pruning, however, does not come without its merits. Firstly, software and hardware must adapt their logic to achieve full utilization - the pruned parameters/neurons should be tracked and skipped during computations efficiently. Secondly, and perhaps most important, pruning a network is the same as deleting part of its information and can potentially lead to accuracy loss. 

The rest of this section will be dedicated to evaluating different pruning methods.

## Overview

To evaluate different pruning methods, we need to have a formal framework containing the relevant definitions. We use the formalization in [[7; 2.1, 2.2]](#ref-7).

A neural network *architecture* $f(x, \cdot)$ is the fixed set of operations to be performed on the input $x$ and its weights. A neural network *model* $f(x, W)$ is a parameterization of $f(x, \cdot)$ with specific parameters $W$. 

**Pruning** is defined as a function, mapping a model $f(x, W)$ to a pruned model $f(x, M \odot W')$ where $W'$ is some set of parameters (different from $W$), $M \in \{0, 1\}^{|W'|}$ is a binary mask.

Most of the pruning methods are variants of the general algorithm outline described in [[1; 3]](#ref-1).

1. Learn the model parameters via training as normal.
2. Prune the parameters having a lower value than some predefined threshold.
3. Retrain the network, to learn the final parameters.

![Pruning algorithm outline](assets/pruning-algorithm-outline.png){width=90%}

## Pruning Methods

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

