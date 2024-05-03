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