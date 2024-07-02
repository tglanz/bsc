---
title: 22934, Mmn15
author: Tal Glanzman
date: 02/07/2024
...

# Answer to 1.a

First we should notice that

$$
    \sum_{i=1}^{n-1} t_i = \sum_{i=1}^{n}i - x
$$

By extracting $x$ we get that

$$
    x = \sum_{i=1}^{n}i - \sum_{i=1}^{n-1} t_i
$$

Finally, using the sum of arithmetic series we can simplify the above to

$$
    x = \frac{n (n+1)}{2} - \sum_{i=1}^{n-1}t_i
$$

This leads us to our algorithm, which will sum the tokens and count them. Finally, return $x$ using the above.

### Algorithm 1.a

1. let $n \leftarrow 0$ and $\sigma \leftarrow 0$
1. while there are more tokens do
    1. let $t$ be the next token
    1. $n \leftarrow n + 1$
    1. $\sigma \leftarrow \sigma + t$
1. return $\frac{n(n+1)}{2} - \sigma$

### Space complexity

1. $Space(n) = O(\log n)$
1. $Space(\frac{n(n+1)}{2}) = O(\log n^2) = O(2 \log n) = O(\log n)$
1. $Space(\sigma) \leq Space(\frac{n(n+1)}{2}) = O(\log n)$
1. Also note that the return value is $O(\log n)$ because it is a positive value less than $\frac{n(n+1)}{2}$

We conclude that 

$$
    Space(1.a) = 4 \cdot O(\log n) = O(\log n)
$$

# Answer to 1.b

We will follow the hint - We will formulate a system of $k$ equations with $k$ variables such that the solution are the missing elements of the stream. Concretely, we will use [Newton's identities](https://en.wikipedia.org/wiki/Newton%27s_identities) to construct a $k$-th order polynomial whose roots are the required solution.

We denote the $k$ missing values by $x_1, x_2, ..., x_k$.

Define the $k$ _power sums_:

$$
p_i = \sum_{j=1}^{k}{x_j}^i
$$

Denote the _[elementary symmetric polynomials](https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial)_

$$
e_i = \sum_{1 \leq j_1 \leq ... \leq j_i \leq n} {x_{j1} \cdot ... \cdot x_{ji}}
$$

i.e. such $i$-th polynomial is the sum of products of subsets having size $i$ of $\{ x_j \}_{j=1}^k$.

Newton's identities states that we can express and compute $e_i$ recursively in terms of $e_{j<i}$ and $p_{j \leq i}$:

$$
    e_i = \frac{\sum_{j=1}^i (-1)^{j-1} e_{i-j} p_j}{i}
$$

Now we can express the polynomial with roots $\{x_j\}_{j=1}^k$ by
$$
    P(x) = \prod_{j=1}^k (x - x_j) = \sum_{j=0}^k (-1)^j e_j x^{k-i} = e_0x^k - e_j x^{k-1} + ... + (-1)^k e_k
$$

Finally, factorizing the above polynomial will yield the missing elements.

Having said that, we suggest the following algorithm:

### Algorithm 1.b

1. let $n \leftarrow 0$
1. let $p_0 \leftarrow 0$, $p_1 \leftarrow 0$, $p_2 \leftarrow 0$, ..., $p_k \leftarrow 0$
1. while there are more tokens do
    1. let $t$ be the next token
    1. $n \leftarrow n + 1$
    1. for $i \leftarrow 1$ to $k$
        1. $p_i \leftarrow p_i + t^i$
1. for $i \leftarrow 1$ to $k$
    1. let $\sigma_i \leftarrow 0$
    1. for $j \leftarrow 0$ to $n$
        1. $\sigma_i \leftarrow \sigma_i + j^i$
    1. $p_i \leftarrow \sigma_i - p_i$
1. let $e_0 \leftarrow 0$
1. for $i \leftarrow 0$ to $k$
    1. let $e_i \leftarrow 0$
    1. for $j \leftarrow 1$ to $k$
        1. $e_i \leftarrow e_i + (-1)^{j-1}e_{i-j}p_j$
    1. $e_i \leftarrow \frac{e_i}{i}$
1. factorize and return the roots of the polynomial $P(x) = e_0x^k - e_j x^{k-1} + ... + (-1)^k e_k$

### Space complexity

$n$ requires $O(\log n)$ space.

For each token $t$, for all $i = 0, 1, ..., k$, the space required for $t^i$ is bounded by the space required for $t^k$. Therefore it is $O(k \log n)$ (we only keep one in memory at each point in time).

For all $i = 0, 1, ..., k$ the variables $p_i$ have value of at most $n \cdot n^k = n^{k+1}$. Therefore they require $O(k \log n)$ each. For the same reasons, the $\sigma_i$ variables require the same space. For all of the $p_i, \sigma_i$ variables we require $O(k^2 \log n)$ space.

Each elementary polynomial $e_i$ is bounded by the sum of products of $i$ variables. It is therefore bounded by $n \cdot n^k$ and as such requires $O(k \log n)$ space. Therefore, the $k$ such polynomials require $O(k^2 \log n)$

Summing the above, we conclude that 

$$
    Space(1.b) = O(k^2 \log n)
$$