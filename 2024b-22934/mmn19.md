---
title: 22934, Mmn19
author: Tal Glanzman, 302800354
date: 21/07/2024
...

\newpage
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
1. factorize and return the roots of the polynomial ${P(x) = e_0x^k - e_j x^{k-1} + ... + (-1)^k e_k}$

### Space complexity

$n$ requires $O(\log n)$ space.

For each token $t$, for all $i = 0, 1, ..., k$, the space required for $t^i$ is bounded by the space required for $t^k$. Therefore it is $O(k \log n)$ (we only keep one in memory at each point in time).

For all $i = 0, 1, ..., k$ the variables $p_i$ have value of at most $n \cdot n^k = n^{k+1}$. Therefore they require $O(k \log n)$ each. For the same reasons, the $\sigma_i$ variables require the same space. For all of the $p_i, \sigma_i$ variables we require $O(k^2 \log n)$ space.

Each elementary polynomial $e_i$ is bounded by the sum of products of $i$ variables. It is therefore bounded by $n \cdot n^k$ and as such requires $O(k \log n)$ space. Therefore, the $k$ such polynomials require $O(k^2 \log n)$

Summing the above, we conclude that 

$$
    Space(1.b) = O(k^2 \log n)
$$

\newpage
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

### Correctness

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

### Correctness

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

\newpage
# Answer to 3

The intuition for the following algorithm is that we will iteratively sample $h$ points and compute the distance between them and reference points that act as representative of their corresponding clusters. If any such distance is larger than $b$ for all clusters, it indicates that there is another cluster. The algorithm's answer will depend on whether or not we have detected more than $k$ clusters in that way.

Algorithm $3$

1. let $h \leftarrow \epsilon^{-1} \ln 3k$
1. let $x_1 \sim U(X)$ (i.e. $x_1$ is a point sampled from a uniform distribution on $X$)
1. let $newCluster \leftarrow true$
1. let $i \leftarrow 1$
1. while $i \leq k$ and $newCluster = true$ do
    1. let $newCluster = false$
    1. for $h$ times and while $newCluster = false$ do
        1. let $y \sim U(X)$
        1. for $j \leftarrow 1$ to $i$ and while $newCluster = false$ do
            1. if $dist(x_j, y) > b$ then
                1. set $x_{i+1} \leftarrow y$
                1. set $i \leftarrow i + 1$
                1. set $newCluster = true$
1. return $i \leq k$

In the algorithm, $i$ represents the number of clusters we detected and $x_i$ are their representatives respectively. Trivially, at the start of the loop at step $5$ there is one cluster which $x_1$ is its representative.

The purpose of the loop at step $5$ is to find additional clusters and it iterates as long as we found additional clusters (up to a maximum of $k+1$ clusters because it is all we need to disprove $k$ clusterability).

The purpose of the loop at step $5.2$ is to find a point which is not $b$ close to any of the existing $i$ clusters. It does so by drawing $h$ samples uniformly and check whether the distance of any of those is larger than $b$ from any of the $i$ currently detected clusters. If there is such distant point, it indicates for a new cluster, let this point be this cluster's representative and continue.

If at the end of the algorithm (step $6$) we detected more than $k$ clusters than we return $false$. Because $i$ is the number of detected cluster, the fact that there are more than $k$ clusters detected is indicated by $i > k$.

## Number of distance queries

The distance queries are performed in step $5.2.2.1$ which is nested in the loops

- at step $5$ which iterates at most $k$ times
- at step $5.2$ which iterates at most $h$ times
- at step $5.2.2$ which iterates at most $j \leq i \leq k$ times

Therefore, the maximum number of queries is given by $k^2 \cdot h$, asymptotically:

$$
    O(\epsilon^{-1} k^2 \ln 3k)
$$

## Correctness

**Case 1: $X$ is $(k, b)$-diameter clusterable**

The algorithm won't be able to detect more than $k$ clusters because there are non. At "the worst case", after the algorithm detected $k$ clusters, there won't be any point in $X$ that its distance from all of those clusters is larger than $b$ and therefore the condition at step $5.2.2.1$ won't be met.

Meaning, at step $6$ it will alway be true that $i \leq k$ and the algorithm will return $true$.

**Case 2: $X$ is $\epsilon$-far from being $(k, 2b)$-diameter clusterable**

Given that the algorithm had already detected $i$ clusters, let $y \sim U(X)$. For any $x_i$ it is true that
$$
    Pr(dist(x_i, y) > 2b) \geq \epsilon
$$

because there are at least $\epsilon |X|$ such points.

And therefore
$$
    Pr(dist(x_i, y) \leq 2b) < 1 - \epsilon
$$

This is the probability for some point **to not satisfy** the condition at step $5.2.2.1$.

The loop at step $5.2$ draws $h$ points. From the above, and because the points are independent, the probability for all such points to not satisfy the condition is at most
$$
    (1 - \epsilon)^h \leq \exp (- \epsilon h) = \exp (- \epsilon \cdot \epsilon^{-1} \cdot \ln 3k) = \frac{1}{3k}
$$

The loop at step $5$ iterates at most $k$ times. If we denote by $A_l$ the event that in iteration $l$ none of the $h$ points sampled satisty the condition at $5.2.2.1$ we can bound the union of those events by
$$
    \bigcup_{l\leq k} Pr(A_l) \leq \sum_{l \leq k} Pr(A_l) \leq k \frac{1}{3k} = \frac{1}{3}
$$

which is the probability that the algorithm will return $true$ (because it will yield $i \leq k$).

Therefore by complement, the probability for the algorithm to return $false$ is at least $\frac{2}{3}$.

**Conclusion**

In the case $X$ is $(k, b)$-diameter clusterable, Algorithm $3$ will always return $true$.

In the case $X$ is $\epsilon$-far from being $(k, 2b)$-diameter clusterable, Algorithm $3$ will return $false$ whp.

\newpage
# Answer to 4

Note: This is heavily inspired by Michal Parnas and Dana Ron's paper "Testing the Diameter of Graphs".

The $C$-neighborhood of some vertex $v$ is defined to be the set of vertices that are of distance less than $C$ from $v$ and we denote it by:
$$
    \Gamma_C(v) = \{ u | dist(u, v) \leq C \}
$$

Algorithm $4(G, D, n, m, \epsilon)$

1. Let $\epsilon_{n,m} \leftarrow \frac{m}{n} \epsilon$
1. let $k \leftarrow \frac{2}{\epsilon_{n, m}}$
1. Let $S$ be a set of $s = 4 \frac{1}{\epsilon_{n, m}}$ vertices sampled uniformly and independently from $V$
1. For each $v \in S$ do
    1. Construct $\Gamma_D(v)$ by running BFS from $v$ with a depth limit of $D$ and neighbors limit of $k$
    1. if $|\Gamma_D(v)| < k$ then
        1. return $false$
1. return $true$

## Query Complexity

The queries to the graph are performed when constructing $\Gamma_D(v)$ in step $4.1$ by doing BFS. There are $s = \frac{4}{\epsilon_{n, m}}$ vertices from which we are doing BFS. For each vertex $v$, the depth is limited to $D$ with a maximum of $k$ neighbours.

Therefore the query complexity is
$$
    O(s k D) = O(\frac{4}{\epsilon_{n, m}} \frac{2}{\epsilon_{n, m}} D) = O(D \cdot \frac{n^2}{m^2} \cdot \epsilon^{-2})
$$

(Alternatively we could formulate it more tight using $\min \{D, k\}$, the result is the same since $D$ will be treated as a constant.)

Recall that $D$ is bounded by some constant $d$ and also note that because the graph is connected $n < m + 1$.

Thus we get the query complexity

$$
    O(\epsilon^{-2})
$$

## Correctness

**Case 1: $diam(G) \leq D$** 

By definition of the diameter we get for all $v \in V$ that $\Gamma_D(v) = G$. Therefore setp $4.2.1$ will never be reached and the algorithm will return $true$ in step $5$.

**Case 2: $G$ is $\epsilon$-far from having diameter $4D + 2$**

First we will prove the following _Lemma_.

---

_Lemma:_ If the $C$-neighborhood of at least $(1 - \frac{1}{k}) n$ of the vertices contains at least $k$ vertices, then the graph can be transformed into a graph with diameter at most $4C + 2$ by adding at most $\frac{2}{k} n$ edges.


_Proof:_ We will prove by construction.

Define a ball of radius $r$ with center at vertex $v$ to be $\Gamma_r(v)$. Now, we will partially cover the graph with disjoint balls using the following iterative process:

1. The center of the first ball is any vertex $v$ s.t. $|\Gamma_C(v)| \geq k$
1. At any subsequent step, the next center selected is any vertex $u$ s.t.
    - $|\Gamma_C(u)| \geq k$
    - $u$ is not contained in any of the previous balls
    - The distance from $u$ to any any of the previous balls' centers is at least $2C$
1. Stop the process once there are no vertices that match the criteria in step $2$

By the way of construction:
- The balls are disjoint
- The number of centers (and balls) is at most $\frac{1}{k} \cdot n$ because each ball contains at least $k$ vertices
- There are at most $\frac{1}{k} \cdot n$ uncovered vertices that are at distance greater than $2C$ from any center

Finally, pick some arbitrary center $v$ and:
- Connect the rest of the centers to $v$ (adding at most $\frac{n}{k} - 1$ edges). 
- Connect the the remaining uncovered vertices that are at distance greater than $2C$ from any center to $v$ (adding at most ${\frac{n}{k} - 1}$ edges)

We now have a graph with diameter of at most $4C + 2$ and the total number of edges added is at most $\frac{2}{k} \cdot n$.

---

Now, we want to show that for every graph that is $\epsilon$-far from diameter $4D + 2$ Algorithm $4$ will return $false$ whp (at least $\frac{2}{3}$).

If the $D$-neighborhood of at most $\frac{1}{k}n = \frac{\epsilon_{n, m}}{2}n$ vertices contains less than $k = \frac{2}{\epsilon_{n, m}}$ vertices, it follows that the $D$-neighborhood of at least $(1 - \frac{1}{k})n$ vertices contains at least $k$ vertices. In this case, by the _Lemma_ we get that the graph can be transformed into a graph with diameter at most $4D+2$ by adding at most $\frac{1}{k}\cdot n = \frac{\epsilon_{n, m}}{2} \cdot n = \frac{m}{2n} \epsilon \cdot n \leq \epsilon \cdot m$ edges i.e. it is $\epsilon$-far from diameter $4D + 2$.

The last paragraph implies that a graph that is $\epsilon$-far from diameter $4D + 2$ has more than $\frac{1}{k} \cdot n = \frac{\epsilon_{n, m}}{2}n$ vertices whose $D$-neighborhood contains less than $k$ vertices. The probability not to sample $s = \frac{4}{\epsilon_{n, m}}$ such vertices is at most
$$
    (1 - \frac{\epsilon_{n, m}}{2})^s \leq \exp(-\frac{\epsilon_{n, m}}{2} \cdot \frac{4}{\epsilon_{n, m}}) = e^{-2} \sim 0.135 < \frac{1}{3}
$$

Thus, the probability that Algorithm $4$ such vertex (and return $false$ in step $4.2.1$) is at least $\frac{2}{3}$.