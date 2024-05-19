---
title: 22934, Mmn14
author: Tal Glanzman
date: 
...

# Answer to question 1

We will use Algorithm 8 presented in chapter 8 with a small variation - The original algorithm initializes a set with all vertices of the graph, but from what I understand from the question requirements, working in the "model of edge insertions only" does not allow us to do this. We can however create the initial set $S$ using the two vertices $s,t \in V$. Then in a similar fashion to Algorithm 8, on each token that is the edge $(u, v)$, we expand and unify the relevant connected components. Finally, we will check if there is a component in $S$ that contains both $s$ and $t$ - If there is, $s$ and $t$ are connected.

The algorithm $A$:

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

After each iteration of algorithm $A$, for every vertex $v \in V$ there is **at most** one set $C_v$ in $C$ such that $v \in C_v$. Thus, at the worst case, the number of vertices we keep track of in $C$, which is the sum of the sizes of sets in $C$, is $n$. Formally $\sum_{c \in C} |c| \leq n$. Thus we require at most $n \log n$ bits for $C$.

As analyzed in the original Algorithm 8, $C_u$ and $C_v$ are pointers to a set in $C$. There are at most $n$ sets in $C$ so $\log n$ bits are sufficient for such pointers. They both require $2 \log n$ bits.

We conclude that

$$
    Space(A) = O(n \cdot\log n)
$$