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

# Answer to 2.a

I took the assumption that the graph is undirected. Meaning, an edge $(u, v)$ adds to the degree of both the vertices $u$ and $v$.

For intuition, if we will think of each original token $(u, v)$ as two different tokens $(u, 1)$ and $(v, 1)$ (of the form $(vertex, frequency)$) we can treat this scenario as the a cash register model. Doing so, we can emulate the way the $CountMinSketch$ algorithm works (Algorithm 3 in chapter 7 of the book) to provide an estimation of occurences of a specific vertex, yielding its estimated degree.

### Algorithm $2.a(\epsilon)$

1. Let $k$ be the least power of two such that $k \geq \frac{2}{\epsilon}$, and let $C$ be an array of size $k$ whose cells are all initially zero.
1. For $m$ sufficiently large to encode any vertex, Choose a random hash function ${h : [m] \rightarrow [k]}$ from a pairwise independent hash functions family $H$.
1. while there are more edges do
    1. Let $(u, v)$ be the next edge
    1. $C[h(u)] \leftarrow C[h(u)] + 1$
    1. $C[h(v)] \leftarrow C[h(v)] + 1$
1. **sketch:** The sketch consists of $C$ and $h$.
1. **query:** Given a vertex $v$, output $\tilde{d_v} = C[h(v)]$ as an estimate for $d_v$.

### Algorithm $2.a$ is a sketch

The proof is similar to that of the $CountMinSketch$ Algorithm.

Consider two streams $\sigma_1$ and $\sigma_2$, and let $C^1$, $C^2$ and $C^{12}$ denote the content of the array $C$ after **Algorithm 2.a** processes $\sigma_1$, $\sigma_2$ and the concatenation $\sigma_1 \cdot \sigma_2$, respectively.

Each cell of any such array $C$ contains the number of occurences of edges that touch a vertex that is assign through $h$ to this cell. Therefore, $C^{12}$ can be calculated based on the contents of $C^1$ and $C^2$ by adding together corresponding cells.

Because there exist some merge function $C^{12} = COMB(C^1, C^2)$, Algorithm 2.a is a sketch.

### Space Complexity

1. The degree of each vertex is bounded by $n - 1$ (complete graph). Therefore the space required for $C$ is ${O(k \log (n-1)) = O(\frac{2}{\epsilon} \log n) = O(\epsilon^{-1} \log n)}$

1. $h$ is a hash function $h : \{0,1\}^{\log m} \rightarrow \{0, 1\}^{\log k}$, according to Theorem 2 in Chapter 5, ${h = O(\log m + \log k) = O(\log m + \log n)}$ space where in the last equality we assumed that $\epsilon \geq \frac{1}{mn}$ (Otherwise, for smaller values of $\epsilon$, storing the sketch cost more than the stream istelf).

1. Each token, that is an edge $(u, v)$, requires $O(2 \cdot \log m) = O(\log m)$ space.

We get the total space complexity

$$
    Space(2.a) = O(\epsilon^{-1} \log n + \log m)
$$

If we also assume (the reasonable) assumption that $m \leq n$ then we get

$$
    Space(2.a) = O(\epsilon^{-1} \log n)
$$

### Error Estimates

We will show that when queried on a token (vertex) $v$, with probability of at least $\frac{1}{2}$, Algorithm $2.a$ outputs a value $\tilde{d_v}$ s.t. $d_v \leq \tilde{d_v} \leq d_v + \epsilon \cdot d$ where $d_v$ is the actual degree of $v$ and $d$ equals to the sum of all degrees in the graph.

Observe that
$$
    \tilde{d_v} = C[h(v)] = \sum_{\substack{u \in [m] \\ h(u) = h(v)}} d_u = d_v + \sum_{\substack{u \in [m]\setminus \{v\} \\ h(u) = h(v)}} d_u
$$

First, because we are in cash register model (specifically frequency is always 1), the addends of the rightmost term are all positive. Therefore we get that $d_v \leq \tilde{d_v}$.

From the last equalities, to prove that $\tilde{d_v} \leq d_v + \epsilon \cdot d$ we need to show that $\sum_{\substack{u \in [m]\setminus \{v\} \\ h(u) = h(v)}} d_u \leq \epsilon \cdot d$.

For every $u \in [m] \setminus \{v\}$ we define an indicator $X_u$ for the event that $h(u) = h(v)$. Because $h$ is drawn from a family of uniform hash functions onto $[k]$, we know that ${Pr(X_u = 1) = \frac{1}{k}}$. We can now rewrite

$$
\sum_{\substack{u \in [m]\setminus \{v\} \\ h(u) = h(v)}} d_u = \sum_{u \in [m] \setminus \{v\}} X_u \cdot d_u
$$

By linearity of expectation we get that

$$
\mathbb{E}[\sum_{\substack{u \in [m]\setminus \{v\} \\ h(u) = h(v)}} d_u] = 
\sum_{u \in [m] \setminus \{v\}} \mathbb{E}[X_u] \cdot d_u =
\frac{1}{k} \sum_{u \in [m] \setminus \{v\}} d_u \leq \frac{d}{k} \leq \frac{\epsilon}{2} d
$$

Using Markov's inequality

$$
Pr(\sum_{\substack{u \in [m]\setminus \{v\} \\ h(u) = h(v)}} d_u \geq \epsilon d) = 
Pr(\sum_{\substack{u \in [m]\setminus \{v\} \\ h(u) = h(v)}} d_u \geq 2 \mathbb{E}[\sum_{\substack{u \in [m]\setminus \{v\} \\ h(u) = h(v)}} d_u]) \leq
\frac{1}{2}
$$

Therefore

$$
Pr(\sum_{\substack{u \in [m]\setminus \{v\} \\ h(u) = h(v)}} d_u < \epsilon d) \geq \frac{1}{2}
$$

To conclude, we got that with probablity of at least $\frac{1}{2}$:
$$
d_v \leq \tilde{d_v} \leq d_v + \epsilon \cdot d
$$

# Answer to 2.b

Our intuition is to run multiple copies of the algorithm using different, randomly selected hash functions. Because the error is upperly bounded, we can take and yield the minimum of the results. The number of copies to run depends on $\delta$.

Algorithm $2.b(\epsilon, \delta)$

1. Let $k$ be the least power of two such that $k \geq \frac{2}{\epsilon}$, and let $r = \lceil \log_2 \delta^{-1} \rceil$
1. Let $C$ be an array of size $r \times k$ whose cells are all initially zero.
1. For an $m$ sufficiently large to encode any vertex in the graph, independently choose $r$ random hash functions ${h_1, h_2, ..., h_r : [m] \rightarrow [k]}$ from a pairwise independent hash functions family $H$.
1. while there are more edges do
    1. Let $(u, v)$ be the next edge
    1. for $i = 1$ to $r$ do
        1. $C[i, h_i(u)] \leftarrow C[i, h_i(u)] + 1$
        1. $C[i, h_i(v)] \leftarrow C[i, h_i(v)] + 1$
1. **sketch:** consists of $C$ and the hash functions $h_1, h_2, ..., h_r$
1. **query:** given a vertex $v$, output $\tilde{d_v} = \min_{1 \leq i \leq r} C[i, h_i(v)]$ as an estimate for $d_v$

### Algorithm $2.b$ is a sketch

We will treat each row of $C$ as a different copy of algorithm $2.a$ which we know how to combine as described previously.

Consider two streams $\sigma_1$ and $\sigma_2$, and let $C^1$, $C^2$ and $C^{12}$ denote the content of the array $C$ after **Algorithm 2.b** processes $\sigma_1$, $\sigma_2$ and the concatenation $\sigma_1 \cdot \sigma_2$, respectively.

For all $i = 1, 2, ..., r$ we will combine the $i$-th row of $C^1$ and $C^2$ to the $i$-th row of $C^{12}$ through elementwise addition.

Because all elements of the combination are positive, the minimum of a specific column across the row dimension will yield the required estimate (the query procedure for some vertex).

Because there exist some merge function $C^{12} = COMB(C^1, C^2)$, Algorithm $2.b$ is a sketch.

### Space Complexity

Realisticly, we have $r$ sketches as in algorithm $2.a$. Therefore the space complexity is
$$
O(r \epsilon^{-1} \log n) = O(\log \delta^{-1} \epsilon^{-1} \log n)
$$

### Error Estimate

We want to show that for every vertex $v$, whp ($1 - \delta$) Algorithm $2.b$ returns and estimate $\tilde{d_v}$ for $d_v$ such that ${d_v \leq \tilde{d_v} \leq d_v + \epsilon \cdot d}$.

Algorithm $2.b$ effectively runs $r$ copies of algorithm $2.a$. Denote by $\tilde{d_v^i}$ the estimate produced by the $i$-th copy.

Because $d_v \leq \tilde{d_v^i}$ for all $i = 1, ..., r$ and $\tilde{d_v} = \min_{1 \leq i \leq r}{\tilde{d_v^i}}$, we conclude that $d_v \leq \tilde{d_v}$.

Algorithm $2.a$ gaurantees that $\tilde{d_v^i} \leq d_v + \epsilon \cdot d$ with probability of at least $\frac{1}{2}$. Therefore, because we run $r$ different, independent copies of that algorith, we get that with probability of at least $1 - \frac{1}{2}^r$ there is some $i$ such that $\tilde{d_v^i} \leq d_v + \epsilon \cdot d$. Because $\tilde{d_v} = \min_{1 \leq i \leq r} \tilde{d_v^i}$ we conclude that

$$
    Pr(\tilde{d_v} \leq d_v + \epsilon \cdot d) \geq 1 - 2^{-r} = 1 - 2^{- \log_2 \delta^{-1}} = 1 - \delta
$$

In conclusion

$$
Pr(d_v \leq \tilde{d_v} \leq d_v + \epsilon \cdot d) \geq 1 - \delta
$$