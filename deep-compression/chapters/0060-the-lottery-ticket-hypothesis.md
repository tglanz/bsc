# The Lottery Ticket Hypothesis

Let's get back to pruning.

As discussed in the Pruning chapter, Iterative Magnitude Pruning, can produce a subnetwork that achieves similar accuracy as the original network.

It begs the question then - Couldn't we just train a simpler architectured network right from the start? Experience and research have previously answered this question negatively. It was assumed that learning a model inherently required more space than inferencing it. Effectively, as exemplified by the process of pruning, besides learning the weights, a model optimizes its connectivity.

In theory, then, we could have used the subnetwork from the start since it has optimal connections. In his paper ["The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"](#ref-l1), Jonathan Frankle states that in practice, this is indeed the case - he shows that the initialization for the weights plays a major role.

His hypothesis is known as **The Lottery Ticket Hypothesis**: (Quote) _A randomly initialized, dense neural network contains a subnetwork that is initialized such that - when trained in isolation - it can match the test accuracy of the original network after training for at most the same number of iterations._

The hypothesis provides two very strong claims about such subnetworks:

1. They are trainable within at most the same number of iterations as the original network
2. They have about the same accuracy as the original network

A way to find such subnetworks is as our initial intuition dictated - through pruning! Although simple retrospectively, the key insight Frankle made is the importance of weight initialization. The main reason that previous experience showed that those subnetworks are not trainable is due to the choice of weight initialization method. In contrast to fine-tuning (which is the common practice), Frankle showed that one-shot pruned networks are trainable if using weight-rewinding to initialize the weights (we discussed those methods in the chapter about pruning). Later, he showed that fine-tuning yields even better results.

To summarize: Either one-shot or iterative pruning yields subnetworks, which, when having their weights rewound, are trainable. The subnetworks can reach similar accuracy as the original network with fewer training iterations.

We refer to such subnetworks as **Winning Tickets**, they learn faster and generalize better than the original networks.

## First results

In this section, we will review some of the results in [[L1]](#ref-l1) that back the hypothesis.

The table below shows the architectures used throughout the following experiments:

+----------------------+------------+------------+--------------+--------------+
| **Network**          | Lenet      | Conv-2     | Conv-4       | Conv-6       |
+======================+============+:==========:+:============:+:============:+
| **Convolutions**     |            | 64,64,pool | 64,64,pool   | 64,64,pool   |
|                      |            | 64,64,pool | 128,128,pool | 128,128,pool |
|                      |            |            |              | 256,256,pool |
+----------------------+------------+------------+--------------+--------------+
| **Fully Connected**  | 300,100,10 | 256,256,10 | 256,256,10   | 256,256,10   |
+----------------------+------------+------------+--------------+--------------+
| **All/Conv weights** | 266K       | 4.3M / 38K | 2.4M / 260K  | 1.7M / 1.1M  |
+----------------------+------------+------------+--------------+--------------+

The convolution filters in the tables have an implicit size of 3x3.

### Importance of architecture

Figure \ref{lotter-weights-vs-iterations} compares winning tickets (solid lines) and randomly sampled sparse subnetworks (dashed lines) for different architectures (color).

The two left figures correlate sparsity (x-axis) with the number of training iterations (y-axis). For the Lenet architecture, it is empirically shown, that all the down to 7% weights remaining, winning tickets require fewer training iterations than the original network (100% weights remaining). For the Conv networks, the number of weights is down even to 3%.

The two right figures correlate sparsity (x-axis) to accuracy (y-axis). For both the Lenet architecture and the Conv architectures we see the same breakpoints of 7% and 3% respectively. Up to those breakpoints, the winning tickets yield higher accuracy than the original network!

We can conclude, that, at least for these architectures, winning tickets of sparsities of up to 97% perform better than the original network in terms of accuracy and faster training.

If this indeed holds for modern networks, the difference in size is huge. Assume a network of size 100GB - Such a network can be replaced by a 3GB sized network, that can effectively be held in most commodity RAMs (Remember that we can further reduce the sizes through quantization and encoding).

![Remaining weights vs. the number of iterations. Source: "The lottery ticket hypothesis:
Finding sparse, trainable neural networks"\label{lotter-weights-vs-iterations}](assets/lottery-weights-remaining-vs-iterations.png){width=100%}

### Importance of initialization

Figure \ref{lottery-random-vs-winning-initialization} shows experiments that compare how winning tickets behave if their weights were initialized randomly (Random Reinit) or using the proposed technique of weight rewinding (Winning Ticket).

Another dimension is whether the tickets were identified using one-shot or iterative pruning.

Firstly, we can tell that the better method to find winning tickets is using iterative pruning. From the top left figure, we see that the Iterative method retains fewer training iterations much better than the Oneshot. In the rest of the figures, it performs slightly better.

Secondly, when we compare Iterative Winning Ticket and Iterative Random Reinint (blue vs. orange) in the 3 top figures, we see that although Random Reinit has the same architecture as the Winning Ticket, its performance drop immediately respective to its sparsity. However, the Iterative Winning Ticket keeps its performance and even improves on it, up to about 95% sparsity.

![Comparison of winning tickets that were initialized randomly (red and orange) and winning tickets that were rewound (green and blue). Source: "The lottery ticket hypothesis:
Finding sparse, trainable neural networks"\label{lottery-random-vs-winning-initialization}](assets/lottery-random-vs-winning-initialization.png){width=100%}

## Instability analysis

### Additional Results

The architectures examined above are relatively small. Additional experiments were performed on a bit more complex networks as described below:

+----------------------+--------------+--------------------+
| **Network**          | Resnet-18    | VGG-19             |
+======================+:============:+:==================:+
| **Convolutions**     | 16,3x[16,16] | 2x64 pool 2x128    |
|                      | 3x[32,32]    | pool, 4x256, pool  |
|                      | 3x[64,64]    | 4x512, pool, 4x512 |
+----------------------+--------------+--------------------+
| **Fully Connected**  | avg-pool,10  | avg-pool,10        |
+----------------------+--------------+--------------------+
| **All/Conv weights** | 274K / 270K  | 20.0M              |
+----------------------+--------------+--------------------+

> Note that even those architectures are only CNNs and are nowhere near modern, state-of-the art architectures.

Similar experiments from the previous section, on the above networks have shown that winning ticket identification behaves differently on the slightly more complex networks.

Figure \ref{lottery-vggs-winning-tickets} shows the results. In this figure:

- Solid and dashed lines refer to Winning Tickets and Random Reinit respectively
- Blue and orange lines refer to learning rates of 0.1 and 0.01 respectively, while the green color refers to a learning rate of 0.01 using warm-up (increasing the learning rate gradually).
- Each chart in the figure contains the result of the same experiment, using different amounts of iterations (30K, 60K and 112K).

![VGG/Resnet Winning Tickets. Source: "The lottery ticket hypothesis:
Finding sparse, trainable neural networks"\label{lottery-vggs-winning-tickets}](assets/lottery-vggs-winning-tickets.png){width=100%}

Our focus on the results should not be about the accuracy values but rather the difference between the performance of Winning Ticket and Random Reinit of the same learning rates.

Let's look at this difference at a learning rate of 0.1 and focus on the solid and dashed blue lines. The performance difference is unnoticeable. A Winning Ticket behaves like a Random Reinit! This is not what we saw in the previous section.

Now, let's look at the difference at a learning rate of 0.01 (orange lines). The performance difference, especially of 30K iterations is highly noticeable, the Winning Ticket again outperforms the Random Reinit!

What happened then? It seems like the learning rate plays a big role in Winning Ticket identification.

Amazingly, if we use the same learning rate of 0.1 (green lines), but now use warm-up (i.e. increasing the learning rate gradually), the Winning Ticket again outperforms the Random Reinit.

The last result hints at the fact that the "instability" occurs at some of the first iterations - probably before the parameters "roughly find their place".

This point remained open in his initial paper about lottery tickets - why Winning Tickets behave in such a way? Later, Frankle assumed that there are networks that are "stable" enough. He formalized what he called the Instability Analysis of neural networks.

### Theory

Instability Analysis is a theoretical framework to determine whether a neural network is stable to SGD noise.

In other terms, it is a process to determine whether a neural network, if retrained using SGD, will achieve similar results as different training sessions.

Because the SGD process is random, the answer to this question is not trivial at all.

We first take a neural network $N$ that was initialized randomly to values $W_0$. Then, we create two copies of it $N_1, N_2$ and train them independently ( - To be correct, we need to make sure the SGD samples different mini-batches). We will set $W_T^1$ and $W_T^2$ to be the trained parameters of $N_1$ and $N_2$ respectively.

If we use a good comparison method to compare and conclude that $W_T^1$ and $W_T^2$ are similar, we can safely assume that $N$ is stable to SGD since different, random processes, yielded similar results.

Now, there are multiple ways to compare between $W_T^1$ and $W_T^2$ such as $L_2$ distance, Cosine distance etc. We will use a different approach - we will analyze the landscape of the loss function (i.e. the error) along the line connecting $W_T^1$ and $W_T^2$. We will call the highest **increase** along this line the **linear interpolation instability** of $N$ to SGD (See figure \ref{instability}). When the instability is near zero it indicates that $W_T^1$ and $W_T^2$ have found a (approximately) linearly connected local minimum.

The **Linear Interpolation** of the weights $W_1$ and $W_2$ is a function defined by:

$$
  \Gamma(\gamma, W_1, W_2) = (1 - \gamma) W_1 + \gamma W_2
$$

For parameters $W$, we will mark the error of a network at $W$ by $\varepsilon(W)$.

Now, we will define the maximum error and the mean error by:

**Maximum Error**

$$
  \varepsilon_{sup}(W_1, W_2) = sup_{\gamma} \varepsilon(\Gamma(\gamma, W_1, W_2))
$$

**Mean Error**

$$
  \varepsilon_{mean}(W_1, W_2) = mean_{\gamma} \frac{1}{2}(\varepsilon(W_1) + \varepsilon(W_2))
$$

Finally, we achieve the **Error Barrier Height** by:

$$
  \varepsilon_{sup}(W_1, W_2) - \varepsilon_{mean}(W_1, W_2)
$$

This quantity is also known as the **Linear Interpolation Instability** and this is the main focus in this theory.

Two networks with parameters $W_1$ and $W_2$ are said to be **Mode Connected** if there exists a path (not necessarily linear) with an Error Barrier Height that is roughly 0. If there exists such a linear path, i.e. the Instability is roughly 0, they are said to be **Linear Mode Connected**.

A network is said to be **Stable to SGD** if the networks that result from Instability Analysis are Linear Mode Connected.

![Instability illustration. Source: "Linear Mode Connectivity and the Lottery Ticket Hypothesis"\label{instability}](assets/instability.png){width=100%}

### Experiments and Results

Instability Analysis was performed over multiple networks which are described in Figure \ref{instability-networks}.

There were two experiments. The first experiment was to perform Instability Analysis at initialization. The second experiment was to perform Instability Analysis after $k$ iterations. In the following experiments, networks with Instability below $2%$ are considered stable.

Figure \ref{instability-intialization} shows the results of the first experiment. The $x$-axis is the interpolation variable (notated by $\gamma$ in the theory section) and the $y$-axis is the error at that interpolation point. Putting in use the definition of Instability here, Instability is the difference between the highest point in the plot between the midpoint of the values at 0.0 and 1.0. The lines represent the mean and standard deviation from multiple samples.

From the results, we observe that only the simple network Lenet is stable at initialization! This is a hint for us, remember that previously it was shown that Lenet Winning Ticket identification can be done at initialization.

Figure \ref{instability-post-initialization} shows the results of the second experiment. In this experiment, we perform Instability Analysis after $k$ training iterations. The $x$-axis is $k$ and the $y$-axis is the measured Instability.

We can deduce from the results that after some iterations, the networks that were experimented on got stable! Remember the previous experiments about Winning Tickets - Identification could be made using learning rate warm-up, i.e. after some iterations.

![Networks that were Instability Analyzed. Source: "Linear Mode Connectivity and the Lottery Ticket Hypothesis"\label{instability-networks}](assets/lottery-instability-networks.png){height=200%}

![Instability Analysis at initialization. Source: "Linear Mode Connectivity and the Lottery Ticket Hypothesis"\label{instability-initialization}](assets/lottery-instability-analysis-initialization.png){width=100%}

![Instability Analysis at iteration $k$. Source: "Linear Mode Connectivity and the Lottery Ticket Hypothesis"\label{instability-post-initialization}](assets/lottery-instability-analysis-post-initialization.png){width=100%}

### Instability and Winning Tickets

Previously we stated that Winning Tickets are identified by performing Iterative Magnitude Pruning (IMP) to achieve a sparse subnetwork and then rewinding the parameters to the original network's state at iteration $k=0$. At iteration $k=0$ the initialization was random whereas in some iteration $k > 0$ the network learned some and the parameters are not considered random.

We call subnetworks that were identified using IMP and rewound to some $k > 0$ that behave like a Winning Ticket (in the sense of accuracy as discussed in the previous sections) as **Matching**.

We previously saw that the Lenet subnetworks that rewound to iteration $k=0$ are Matching. We also saw that subnetworks of VGG-19 and Resnet-18 that were identified either by a very low learning rate or using warm-up, are also Matching (See section "First Results").

The central findings in [[L2; 4]](#ref-l2), that somewhat explain the phenomena of finding Matching subnetworks is that subnetworks are Matching if, and only if, they are stable.

The paper provides multiple results that correlate between the instability of the subnetworks and whether it's matching or not. We already saw the networks are stabilizing with learning. How about the subnetworks?

Figure \ref{lottery-instability-subnetworks-iteration} show the Instability of subnetworks that were created from the network at learning iteration $k$ - The percentages are the amount of the remaining weights. We see that **IMP**, weight rewinding to iteration $k$ are stabilizing with $k$.

Figure \ref{lottery-error-subnetworks-iteration} show the Error of subnetworks that were created from the network at learning iteration $k$ - The percentages are the amount of the remaining weights. The gray line is the full network. We see that **IMP**, weight rewinding to iteration $k$, is Matching with $k$ (touches the full network's error).

The key point to look at is that at the same iterations where the subnetworks are getting stable, they are also Matching.

![Instability Analysis of subnetworks at iteration $k$. Source: "Linear Mode Connectivity and the Lottery Ticket Hypothesis"\label{lottery-instability-subnetworks-iteration}](assets/lottery-instability-subnetworks-iteration.png){width=100%}

![Error of subnetworks at iteration $k$. Source: "Linear Mode Connectivity and the Lottery Ticket Hypothesis"\label{lottery-error-subnetworks-iteration}](assets/lottery-error-subnetworks-iteration.png){width=100%}

### Final Notes

You might have noticed that all of the networks that were experimented on are convolutional networks. It is currently researched how the hypothesis holds on other, more complex and modern architectures like those found in NLP domains (RNNs, Transformers etc.). Existing results have shown that the hypothesis somewhat holds.

Another question regarding Winning Tickets is about what they encode and how can we learn from the to better architect new models. There are some results and insights regarding this topic that we haven't covered in this work.