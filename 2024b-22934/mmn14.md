---
title: 22934, Mmn14
author: Tal Glanzman
date: 20/05/2024
...

# Answer to question 1

We will use Algorithm 8 presented in chapter 8 with a small variation - The original algorithm initializes a set with all vertices of the graph, but from what I understand from the question requirements, working in the "model of edge insertions only" does not allow us to do this. We can however create the initial set $S$ using the two vertices $s,t \in V$. Then in a similar fashion to Algorithm 8, on each token that is the edge $(u, v)$, we expand and unify the relevant connected components. Finally, we will check if there is a component in $S$ that contains both $s$ and $t$ - If there is, $s$ and $t$ are connected.

We define Algorithm $A(s, t)$ as follows:

1. Let $C \leftarrow \{ \{s\}, \{t\} \}$
1. **while** there are more edges **do**
    1. Let $(u, v)$ be the next edge
    1. **if** there is no set containing $u$ in $C$ **then**
        1. $C \leftarrow C \cup \{u\}$
    1. **if** there is no set containing $v$ in $C$ **then**
        1. $C \leftarrow C \cup \{v\}$
    1. Let $C_u$ and $C_v$ be sets in $C$ containing $u$ and $v$, respectively
    1. **if** $C_u \neq C_v$ **then**
        1. Remove $C_u$ and $C_v$ from $C$ and add $C_u \cup C_v$ to $C$ instead
1. Declare $s$ and $t$ to be connected if there is a set $C_{uv} \in C$ that contains both $u$ and $v$, and disconnected otherwise

## Space Complexity

We require only $2 \log n$ bits for the For the edge $(u, v)$.

After each iteration of algorithm $A$, for every vertex $v \in V$ there is **at most** one set $C_x$ in $C$ such that $x \in C_v$. Thus, in the worst case, the number of vertices we keep track of in $C$, which is the sum of the sizes of sets in $C$, is $n$. Formally $\sum_{c \in C} |c| \leq n$. We require at most $n \log n$ bits for $C$.

As analyzed in the original Algorithm 8, $C_u$ and $C_v$ are pointers to a set in $C$. There are at most $n$ sets in $C$ so $\log n$ bits are sufficient for such pointers. They both require $2 \log n$ bits.

We conclude that

$$
    Space(A) = O(n \cdot\log n)
$$

and algorithm $A$ is a semi-streaming algorithm

# Answer to question 2

By convention, given a graph $H$, if $(u, v) \notin H$ we set $d_H(u, v) = \infty$.

Denote $\alpha = 2k - 1$.

We define Algorithm $B(s, t, \alpha)$ as follows

1. Let $H \leftarrow \phi$
1. **while** there are more edges **do**
    1. Let $(u, v)$ be the next edge in the input graph $G$
    1. **if** $d_H(u, v) > \alpha$ **then**
        1. $H \leftarrow H \cup \{(u, v)\}$
1. Return $d_H(s, t)$

## Approximation Factor

Denote $G = (V_G, E_G)$.

On the one hand, because Algorithm $B$ adds only a subset of edges of $E_G$ that satisfy the condition in $2.2$ it holds that $H \subseteq E_G$. Thus, all paths that exist in $H$ also exist in $E_G$, therefore

$$
    d_G(s, t) \leq d_H(s, t) = d_{Alg}(s, t)
$$

On the second hand, Let $p$ be a shortest path in $G$ from $s$ to $t$. The path $p$ is comprised of $d_G(s, t)$ edges - Every such edge is given to algorithm $B$ as part of the stream. For every such edge $(u, v)$, condition $2.2$ makes sure that $d_H(u, v) \leq \alpha$. Therefore

$$
    d_{Alg}(s, t) = d_H(s, t) \leq \alpha \cdot d_G(s, t)
$$

We conclude that

$$
    d_G(s, t) \leq d_{Alg}(s, t) \leq (2k - 1) \cdot d_G(s, t)
$$

## Space Complexity

There are no cycles of length less than $2k$ in $H$. Assume by contradiction that there is such a cycle - It means that at some iteration of step $2$, there is a non-cycle path $p$ in $H$ that is of length at most $2k - 1$ and the edge $(u, v) \in E_G$ at $2.1$ satisfies that $p \cup (u, v)$ is a cycle of length less than $2k$. Notice that $u$ and $v$ along the path $p$ (it is easy to imagine them as the ends but it is not mandatory that they are), therefore $d_H(u, v) \leq 2k - 1$ which contradicts the fact that $(u, v)$ was inserted to the graph because it should have satisified condition $2.2$. Now, according to McGregor's Lemma we know that $H$ has $O(n^{1 + \frac{1}{k}})$ edges. Each edge requires at $O(\log n)$ bits and thus $Space(H) = O(\log n \cdot n^{1 + \frac{1}{k}})$.

Holding the edge $(u, v)$ requires $2 \cdot \log n$ bits, i.e. $Space((u, v)) = O(\log n)$. 

To compute $d_H(s, t)$ in step $3$ we will use the $BFS$ algorithm which requires $O(n)$ space.

All in all

$$
    Space(B) = O(\log n \cdot n^{1+\frac{1}{k}})
$$

and algorithm $B$ is a semi-streaming algorithm.