---
title: 22934, Mmn14
author: Tal Glanzman
date: 20/05/2024
...

# Answer to question 1

We define the distance on strings the same way we define distance on lists in the List Model.

Define the required algorithm $A(s, \epsilon)$ as follows:

1. **let** $h \leftarrow \frac{1}{\epsilon} \ln 3$
1. **do** $h$ times
    1. Pick uniformly a random character $c \in s$
    1. **if** $c == "1"$ **then**
        1. **return** "No"
1. **return** "Yes" 

## Correctness and probabilistic analysis

### Case 1: $s$ is the zero string

If $s$ is the zero string condition "2.2" will never be true and the algorithm will always terminate in step "3" returning a "Yes".

### Case 2: $s$ is $\epsilon$ far from the zero string

We want to show that w.p. of at least $\frac{2}{3}$ the algorithm will return a "No". i.e., we want to show that $Pr(A = "No") \geq \frac{2}{3}$.

Because there are at least $\epsilon n$ "1" characters in the string, the probability to uniformly pick a "0" is at most $1 - \frac{\epsilon n}{n} = 1 - \epsilon$. Thus, the probability to sample $h$ "0" characters is at most $(1 - \epsilon)^h$.

Meaning that

$$
    Pr(A = "Yes") \leq (1-\epsilon)^h \leq \exp(- \epsilon h) = \exp(\ln \frac{1}{3}) = \frac{1}{3}
$$

Concluding that

$$
    Pr(A = "No") = 1 - Pr(A="Yes") \geq \frac{2}{3}
$$

## Run complexity analysis

All of the operations in the algorithm are in constant time except the loop of $h$ iterations.

$$
    A(s) = O(\frac{1}{\epsilon})
$$

# Answer to question 2

We use the List Model s.t. the list's element at index $1 \leq i \leq n$ is $f(i)$.

The following suggested algorithm works in a somewhat similar manner to the algorithm proposed in the book to test for half plane.

Algorithm $B(f)$ is defined by

1. **if** $f(0) = 1 \land f(n) = 0$ **then**
    1. **return** "No"
1. **else if** $f(0) = f(1)$ **then**
    1. **let** $h \leftarrow \frac{1}{\epsilon} \ln 4$
    1. **do** $h$ times
        1. Sample a uniform index $i \in \{2, 3, ..., n - 1\}$
        1. **if** $f(0) \neq f(i)$
            1. return "No"
    1. **return** "Yes"
1. **else**
    1. **let** $h \leftarrow \frac{2}{\epsilon} \ln 4$
    1. **let** $k \leftarrow \log_2 \frac{1}{\epsilon} + 1$
    1. **let** $l \leftarrow 0$ and $r \leftarrow n$.
    1. **do** $k$ times
        1. **let** $m \leftarrow \lfloor \frac{l + r}{2} \rfloor$
        1. **if** $f(m) = 0$ **then set** 
            $l \leftarrow m$
        1. **else**
            1. $r \leftarrow m$
    1. **do** $h$ times
        1. Uniformly sample an index $i$ in the range $\{1, 2, ..., l\} \cup \{r, r + 1, ..., n \}$
        1. **if** $i \geq r \land f(i) = 0$ **then**
            1. **return** "No"
        1. **if** $i \leq l \land f(i) = 1$ **then**
            1. **return** "No"
    1. **return** "Yes"

## Correctness

Algorithm $B$ is comprised of 3 cases, we will analyze them seperately.

### Case 1: $f(0) = 1 \land f(n) = 0$

$f$ is not monotone by definition and the algorithm will accurately return "No" in step $1.1$.

### Case 2: $f(0) = f(1)$

We will assume without loss of generality that $f(0) = f(1) = 0$.

**If $f$ is monotone** then forall $i \in \{ 2, 3, ..., n - 1 \}$ it will be true that $0 = f(0) \leq f(i) \leq f(n) = 0$ which means that $f(i) = 0 = f(0)$. Because $f(i) = f(0)$ the condition at step $2.2.2$ will never be satisfied and the algorithm will accurately return "Yes" in step $2.2.3$.

**If $f$ is $\epsilon$-far from monotone** there are at least $\epsilon n$ elements that are 1. Algorithm $B$ will return "Yes" (step $2.2.3$) only if it uniformly sampled $h$ indices s.t. $f=1$ at those indices. Therefore

$$
    Pr(A = "Yes") \leq (1 - \frac{\epsilon \cancel n}{\cancel n})^h \leq e^{-\epsilon h} = e^{- \ln 4} = \frac{1}{4}
$$

Finally we get that

$$
    Pr(A = "No") \geq \frac{3}{4}
$$

### Case 3: $f(0) = 0 \land f(n)=1$

This case applies to step $3$ in the algorithm.

The loop at step $3.4$ runs $\log_2 \frac{1}{\epsilon}+1$ times, halving the range each iteration and setting $l$ and $r$ to be the edges of the range while maintining the invariance that $f(l) = 0$ and $f(r) = 1$. The size of the final range $[l, r]$ is given by

$$
    n \cdot \frac{1}{2} \cdot \frac{1}{2}^{\log_2 \frac{1}{\epsilon}} = n \cdot \frac{1}{2} \cdot 2^{\log_2 \epsilon} = \frac{\epsilon n}{2}
$$

**If $f$ is monotone** the algorithm will surely return "Yes" because the conditions at $3.5.2$ and $3.5.3$ will never be satisfied.

**If $f$ is $\epsilon$-far from monotone** then there are more than $\epsilon n$ out of order values.

Denote the set $X = \{ 1, ..., l - 1 \} \cup \{ r + 1, ..., n \}$.

Because $|[l, r]| \leq \frac{\epsilon n}{2}$, it doesn't matter for how many indices in the range $[l, r]$ the function $f$ assume out of order values, there will be more than $\frac{\epsilon n}{2}$ indices in the set $X$ s.t. $f$ assume out order values for. Because we know that $f(l) = 0$ and $f(r) = 1$, out of order values of $f$ in $X$ can only mean one of two options: Either
$i \leq l \land f(i) = 1$ or $i \geq r \land f(i) = 0$. In the loop at step $5$ we sample $h$ indices from $X$ and check whether there are out of order values according to this criteria.

The probability of uniformly sampling a single in order index from $X$ is at most

$$
    \frac{n - \frac{\epsilon n}{2}}{n} = 1 - \frac{\epsilon}{2}
$$

Hence

$$
    Pr(B = "Yes") \leq (1 - \frac{\epsilon}{2})^h \leq e^{- \frac{\epsilon h}{2}} = \frac{1}{4}
$$

Finally we conclude that

$$
    Pr(B = "No") \geq 1 - Pr(B = "Yes") \geq \frac{3}{4}
$$

## Run complexity

Algorithm $B$ is comprised of 3 cases, we will analyze them seperately.

### Case 1: $f(0) = 1 \land f(n) = 0$

Run in constant time $O(1)$

### Case 2: $f(0) = f(1)$

Run in $O(h) = O(\frac{1}{\epsilon})$

### Case 3: $f(0) = 0 \land f(n)=1$

Run in $O(k + h) = O(\log \frac{1}{\epsilon} + \frac{1}{\epsilon}) = O(\frac{1}{\epsilon})$

In any case, $O(B) = \frac{1}{poly(\epsilon)}$ time.