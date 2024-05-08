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

Define $p = \lg m$ partitions of $M = [m] = \{1, 2, ..., m\}$:

$I_0 = \{1, 2, ..., m \}$

$I_1 = \{ \{1, 2 \}, \{3, 4\}, ..., \{m-1, m\} \}$

$...$

$I_{p-1} = \{ \{1, 2, ..., \frac{m}{2}\}, \{\frac{m}{2} + 1, \frac{m}{2} + 2, ..., m\} \}$

$I_p = \{ \{ 1, 2, ..., m \} \}$

Algorithm $A$ is defined by:

1. Set $\tilde{\epsilon} = \frac{\epsilon}{p}$ and $\tilde{\delta} = \frac{\delta}{p}$
1. Initialize $p$ copies of algorithm $CountMin(\tilde{\epsilon}, \tilde{\delta})$ and denote $A_i$ to be the $i$-th copy. The domain of tokens for each copy $A_i$ is the elements of the subset $I_i$.
    - Every token $t$ correspond to some element in $I_i$ according the mapping $t \mapsto \lceil  \frac{t}{2^i} \rceil$. For example, $t=3$ is mapped to the second element at $I_1$ which is $\{ 3, 4 \}$. When we will pass tokens to the algorith $A_i$ we will **implictly** map the token to the relevant domain $I_i$.
1.  Initialize $n \leftarrow 0$ and $l_1 \leftarrow 0$
1. While there is more token $(t, c)$ do
    1. Invoke $A_i(t, c)$
    1. $l_1 \leftarrow l_1 + c$
    1. $n \leftarrow \max \{ n, l_1 \}$