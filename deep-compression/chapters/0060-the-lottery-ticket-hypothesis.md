# The Lottery Ticket Hypothesis

Let's get back to pruning.

As previously discussed, magnitude pruning and fine-tuning, produce a subnetwork that achieves similar accuracy as the original network.

It begs the question then - Couldn't we just train a simpler architectured network right from the start? Experience and research answer this question negatively. It was assumed that learning a model inherently required more space than inferencing it. Effectively, as exemplified by the process of pruning, besides learning the weights, a model optimizes its connectivity.

In theory, then, we could have used the subnetwork from the start since it has optimal connections. In his paper ["The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"](#ref-l1), Jonathan Frankle states that in practice, this is indeed the case - he shows that the initialization for the weights plays a major role.

His hypothesis is known as **The Lottery Ticket Hypothesis**: (Quote) _A randomly initialized, dense neural network contains a subnetwork that is initialized such that - when trained in isolation - it can match the test accuracy of the original network after training for at most the same number of iterations._

The hypothesis provides two very strong claims about such subnetworks:

1. They are trainable within at most the same number of iterations as the original network
2. They have about the same accuracy as the original network

A way to find such subnetworks is in the way our initial intuition dictated - through pruning! Although simple retrospectively, the key insight Frankle made is the importance of weight initialization. The main reason that previous experience showed that those subnetworks are not trainable is due to the choice of weight initialization method. In contrast to fine-tuning (which is the common practice), Frankle showed that one-shot pruned networks are trainable if using weight-rewinding to initialize the weights (we discussed those methods in the chapter about pruning).

To summarize: Either one-shot or iterative pruning yields subnetworks, which, when having their weights rewound, are trainable. The subnetworks can reach similar accuracy as the original network with fewer training iterations.

We refer to such subnetworks as **winning tickets**, they learn faster and generalize better than their original networks.

## Results

In this section, we will review some of the results in [[L1]](#ref-l1) that back the hypothesis.

| _Network_          | **Lenex**    | **Conv-2** | **Conv-4**              | **Conv-6**                           | **Resnet-18**                    | **VGG-19**                                           |
|--------------------|--------------|------------|-------------------------|--------------------------------------|----------------------------------|------------------------------------------------------|
| Convolutions       |              | 64,64,pool | 64,64,pool 128,128,pool | 64,64,pool 128,128,pool 256,256,pool | 16,3x[16,16] 3x[32,32] 3x[64,64] | 2x64 pool 2x128 pool, 4x256, pool 4x512, pool, 4x512 |
| Fully Connecteds   | 300, 100, 10 | 256,256,10 | 256,256,10              | 256,256,10                           | avg-pool,10                      | avg-pool,10                                          |
| _All/Conv Weights_ | 266K         | 4.3M / 38K | 2.4M / 260K             | 1.7M / 1.1M                          | 274K / 270K                      | 20.0M                                                |

The convolution filters in the table have an implicit size of 3x3.

### Importance of architecture

Figure \ref{lotter-weights-vs-iterations} compares winning tickets (solid lines) and randomly sampled sparse subnetworks (dashed lines) for different architectures (color).

The two left figures correlate sparsity (x-axis) with the number of training iterations (y-axis). For the Lenet architecture, It is empirically shown, that down to 7% weights remaining, winning tickets require fewer training iterations than the original network (100% weights remaining). For the Conv networks, the number of weights is down even to 3%.

The two right figures correlate sparsity (x-axis) to accuracy (y-axis). For both the Lenet architecture and the Conv architectures we see the same breakpoints of 7% and 3% respectively. Up to those breakpoints, the winning tickets yield higher accuracy than the original network!

We can conclude, that, at least for these architectures, winning tickets of sparsities of up to 97% perform better than the original network in terms of accuracy and faster training.

If this indeed holds for modern networks, the difference in size is huge. Assume a network of size 100GB - Such a network can be replaced by a 3GB sized network, that can effectively be held in most commodity RAMs (Remember that we can further reduce the sizes through quantization and encoding).

![Remaining weights vs. the number of iterations. Source: "The lottery ticket hypothesis:
Finding sparse, trainable neural networks"\label{lotter-weights-vs-iterations}](assets/lottery-weights-remaining-vs-iterations.png){width=90%}

### Importance of initialization

Figure \ref{lottery-random-vs-winning-initialization} shows experiments that compare how winning tickets behave if their weights were initialized randomly (Random Reinit) or using the proposed technique of weight rewinding (Winning Ticket).

Another dimension is whether the tickets were identified using one-shot or iterative pruning.

Firstly, we can tell that the better method to find winning tickets is using iterative pruning. From the top left figure, we see that the Iterative method retains fewer training iterations much better than the Oneshot. In the rest of the figures, it performs slightly better.

Secondly, when we compare Iterative Winning Ticket and Iterative Random Reinint (blue vs. orange) in the 3 top figures, we see that although Random Reinit has the same architecture as the Winning Ticket, its performance drop immediately respective to its sparsity. However, the Iterative Winning Ticket keeps its performance and even improves on it, up to about 95% sparsity.

![Comparison of winning tickets that were initialized randomly (red and orange) and winning tickets that were rewound (green and blue). Source: "The lottery ticket hypothesis:
Finding sparse, trainable neural networks"\label{lottery-random-vs-winning-initialization}](assets/lottery-random-vs-winning-initialization.png){width=90%}

## TODO - Below are only leftovers from the outline

This chapter is the core of our work - Discuss the hypothesis and showcase further results.

- Initial hypothesis
- Adjustments for bigger networks
  - Linear mode connectivity
- Impact of the Hypothesis
  - Practical
  - Theoretical: 
- Characteristics of a lottery ticket?

