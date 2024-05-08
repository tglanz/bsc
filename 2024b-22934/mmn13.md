---
title: 22934, Mmn13
author: Tal Glanzman
date: 03-05-2024
...

# Answer to question 1

Set $k = 80 \epsilon^{-2} = O(\frac{1}{\epsilon^2})$.

For every token $t$ define the indicator

$$
Y_t = \begin{cases}
    1 & h(t) < \frac{k}{(1 + \epsilon) d} \\
    0 & otherwise
\end{cases}
$$

From the uniformity and pariwise independence of the randomly chosen hash function $h$ we get that $Pr(Y_t) = \frac{k}{(1+\epsilon)d} = p$. We can immediately conclude that 

$$
    \mathbb{E}(Y_t) = p ~~,~~ Var(Y_t) = p(1-p)
$$

Denote by $D$ the set of **distinct tokens** in the stream and define $Y$ to be the number of hash values below the threshold $\frac{k}{(1+\epsilon)d}$ i.e.

$$
    Y = \sum_{t \in D}{Y_t}
$$

From the linearity of expectiation we get that 

$$
\mathbb{E}(Y) = d \mathbb{E}(Y_t) = \frac{k}{1+\epsilon}
$$

Because for every two different tokens $t_1, t_2$ the random variables $Y_{t_1}, Y_{t_2}$ are independent, we get that $Var(Y) = \sum_{t \in D}{Var(Y_t)} = dp(1-p) < dp$. We know that $0 \leq \epsilon \leq 1$ so we conclude that

$$
    Var(Y) \leq k
$$

To bound $Pr(X > (1+\epsilon)d)$ notice that

$$
    Pr(X > (1+\epsilon)d) = Pr(\frac{z_k}{k} > (1+\epsilon)d) = Pr(z_k < \frac{k}{(1+\epsilon)d})
$$

$z_k < \frac{k}{(1+\epsilon)d}$ is true iff the number of hash values below the threshold $\frac{k}{(1+\epsilon)d}$ is greater than k, meaning that

$$
    Pr(X > (1+\epsilon)d) = Pr(Y > k)
$$

Using Chebyshev's inequality we get that

\begin{align*}
    Pr(Y > k) &= Pr(Y - \mathbb{E}(Y) > k - \mathbb{E}(Y)) \\
    &\leq \frac{Var(Y)}{(k -\mathbb{E}(Y))^2} \\
    &\leq \frac{k}{(k - \frac{k}{1+\epsilon})^2} = \frac{(1+\epsilon)^2}{k\epsilon^2} \\
    &\leq \frac{4}{80 \epsilon^{-2} \epsilon^2} = 0.05
\end{align*}

# Answer to question 2

Define $p = \log m$ partitions of $M = [m] = \{1, 2, ..., m\}$:

$I_0 = \{1, 2, ..., m \}$

$I_1 = \{ \{1, 2 \}, \{3, 4\}, ..., \{m-1, m\} \}$

$...$

$I_{p-1} = \{ \{1, 2, ..., \frac{m}{2}\}, \{\frac{m}{2} + 1, \frac{m}{2} + 2, ..., m\} \}$

$I_p = \{ \{ 1, 2, ..., m \} \}$

Algorithm $A$ is defined by:

1. Set $\tilde{\epsilon} = \frac{\epsilon}{p}$ and $\tilde{\delta} = \frac{\delta}{p}$
1. Initialize $p$ copies of algorithm $CountMin(\tilde{\epsilon}, \tilde{\delta})$ and denote $A_i$ to be the $i$-th copy. The domain of tokens for each copy $A_i$ is the elements of the subset $I_i$.
    - Every token $t$ correspond to some element in $I_i$ according the mapping $t \mapsto \lceil  \frac{t}{2^i} \rceil$. For example, $t=3$ is mapped to the second element at $I_1$ which is $\{ 3, 4 \}$.
1. Initialize $n \leftarrow 0$ and $l_1 \leftarrow 0$
1. While there are more tokens, get the next token $(t, c)$ and do:
    1. Invoke $A_i(\frac{t}{2^i}, c)$ forall $i = 0, 1, ..., p$
    1. $l_1 \leftarrow l_1 + c$
    1. $n \leftarrow \max \{ n, l_1 \}$
1. Initialize $i \leftarrow 0$ and $\sigma \leftarrow 0$
1. for $j \leftarrow p$ to $1$ do:
    1. Set $f$ to be the frequency of $i$ as estimated by $A_j$
    1. If $\sigma + f < \frac{n}{2}$ then
        1. Set $i \leftarrow 2i + 1$ and $\sigma \leftarrow \sigma + f$
    1. Else if $\sigma + f > \frac{n}{2} + \epsilon n$ then
        1. Set $i \leftarrow 2i - 1$
    1. Else return the last token of the $i$-th element in the subset $I_j$ 
1. Return $i$

As a general explanation for the algorithm:
- Steps 1 to 4 delegates the tokens to $p$ copies of the $CountMin$ algorithm, a paradigm we already saw multiple times in previously learnt algorithms.
- Step 6 navigates through the elements of subsets $I_p, I_{p-1}, ..., I_0$ (as indicated by the variable $i$) in a manner that the $\sigma$ variable will converge within the interval $[\frac{n}{2}, \frac{n}{2} + \epsilon n]$. Therefore by the time it reaches the $I_0$ subset, which contains the original stream tokens, $i$ will point to the estimated token $i$ such that $\sum_{k=1}^i {f_k} \in [\frac{n}{2}, \frac{n}{2} + \epsilon n]$ (with some probability).

### Probabilistic analysis

The sum $\sum_{i=1}^t {f_i}$ (for the median $t$) is computed as part of step 6 in the algorithm. To bound the probability of this sum we will bound the probabilities of its operands.

Denote the followings:

- $s$ is the number of iterations performed in step 6
- $f_i$ and $\tilde{f_i}$ are the actual and estimated frequencies of the element at iteration $i$ respectively

Notice that

$$
    \bigcap_{i=1}^s f_i \leq \tilde{f_i} \leq f_i + \tilde{\epsilon} n \iff \sum_{i=1}^s f_i \leq \sum_{i=1}^s \tilde{f_i} \leq \sum_{i=1}^s (f_i + \tilde{\epsilon} n)
$$

Because $s \leq \log m$ we bound $\sum_{i=1}^s (f_i + \tilde{\epsilon} n) \leq \sum_{i=1}^s f_i + \epsilon n$

We know by the probabilitty analysis of $CountMin$ that $Pr(f_i \leq \tilde{f_i} \leq f_i + \tilde{\epsilon} ||f||_1) > 1 - \delta$ and because $||f||_1 \leq n$ we get that 

$$
    Pr(f_i \leq \tilde{f_i} \leq f_i + \tilde{\epsilon} n) \geq 1 - \delta
$$

Forall $i = 1, 2, ..., s$.

Therefore **(1)**

$$
    Pr ( \sum_{i=1}^s f_i \leq \sum_{i=1}^s \tilde{f_i} \leq \sum_{i=1}^s f_i + \epsilon ) \geq 1 - \delta
$$

On the other hand, according to the way the algorithm works, the sum of the estimated frequencies satisfies (according to the conditions in step 6) **(2)**:

$$
    \frac{n}{2} \leq \sum_{i=1}^s \tilde{f_i} \leq \frac{n}{2} + \epsilon n
$$

Combining the right hand side of **(2)** with **(1)** we get

$$
    Pr(\sum_{i=1}^sf_i \leq \frac{n}{2} + \epsilon n) \geq 1 - \delta
$$

Combining the left hand side of **(2)** with **(1)** we get

$$
    Pr(\frac{n}{2} \leq \sum_{i=1}^s f_i + \epsilon) = Pr(\frac{n}{2} - \epsilon \leq \sum_{i=1}^s f_i) \geq 1  - \delta
$$

Together

$$
    Pr(\frac{n}{2} - \epsilon \leq \sum_{i=1}^s f_i \leq \frac{n}{2} + \epsilon n) \geq 1  - \delta
$$

### Complexity analysis

- Maintain $\log m$ of $CountMin$ algorithms is $O(\log m \log \tilde{\delta}^{-1} (\tilde{\epsilon}^{-1} \log n + \log m))$

- The variables $n$, $l_1$ and $\sigma$ which are initialized in steps 3 and 5 are all $O(\log n)$ 
- The variable $j$ in step 6 is $O(\log \log m)$
- The variable $i$ defined in step 5 is $O(\log m)$

All in all, we get the space complexity of

$$
    O(\log^2 m \cdot \log (\frac{\log m}{\delta}) \cdot \frac{\log n}{\epsilon})
$$
