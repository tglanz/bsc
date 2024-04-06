---
title: 22943, Mmn11
author: Tal Glanzman
date: 06-04-2024
...

# Answer 1

We define $B$ to be the following algorithm:

- For the input $x$, execute $A(x)$ for $c = \ln \frac{1}{\alpha \beta}$ times such that $\alpha >= 12$.

- If the majority of executions yielded $True$, return $True$.

- Return $False$.

To analyze the error probability of $B$ we will define the following random variables:

Forall $i = 1, 2, ..., c$, let $X_i$ be indicator indicating whether the execution of $A$ at iteration $i$ is wrong. By definition of $A$ we get that $Pr(X_i = 1) = \frac{1}{4}$ thus $X_i \sim Ind(\frac{1}{4})$.

Let $X$ be the binomial random variable $X = \sum_{i = 1}^{c} X_i \sim Bin(c, \frac{1}{4})$. From the binomial distribution properties we know that $\mathbb{E}(X) = \frac{1}{4}c$.

The algorithm $B$ is wrong iff the majority of executions of $A$ were wrong, meaning $B~is~wrong \iff X \geq \frac{c}{2}$. Thus, the probability that $B$ is wrong is given by

$$
    Pr(B~is~wrong) = Pr(X \geq \frac{c}{2})
$$

Denote $\delta = 1$ and notice that we get that $(1+\delta)\mathbb{E}(X) = 2 \frac{c}{4} = \frac{c}{2}$. Thus, by the Chernoff Inequality we get that

$$
    Pr(B~is~wrong) = Pr(X \geq \frac{c}{2}) = Pr(X \geq (1+\delta)\mathbb{E}(X)) \leq e^{- \frac{\mathbb{E}(X)}{3}} = e^{-\frac{c}{12}}
$$


So

$$
    Pr(B~is~wrong) \leq e^{ - \frac{\ln \frac{1}{\alpha \beta}}{12} } = \frac{\alpha \beta}{12}
$$

Because we set $\alpha >= 12$ we can conclude that

$$
    Pr(B~is~wrong) \leq \beta
$$

Regarding execution time of $B$ - $B$ ran the algorithm $A$ for $c = O(\ln \frac{1}{\beta})$ times. Each execution of $A$ is $T(n)$ time. We get that $B$ runs in $O(T(n) \ln \frac{1}{\beta})$ time.

# Answer 2

Denote the domain $D = [\frac{f(x)}{1 + \varepsilon}, (1 + \varepsilon)f(X)]$.

Define the algorithm $B$ on input $x$:

1. Execute $k = 12 \ln \delta^{-1}$ randomly independent copies of $A(x)$ and denote them by $A_i(x)$ forall $i = 1, 2, ..., k$.
2. Return the median of the results $med \{ A_i(x) \}_{i = 1}^{k} \}$.

We will now show that the error probability of $B$ is at most $\delta$.

Forall $i = 1, 2, ..., k$ denote $Q_i$ an indicator such that $Q_i = 1 \iff A_i(x) \notin D$. By definition of $A$, it is clear that $Pr(Q_i) < 1 - \frac{3}{4} = \frac{1}{4} = p$.

Let $Q = \sum_{i = 1}^{k}{ Q_i(x) }$ be a random variable that counts the number of results that are not in $D$. By its definition, $Q \sim Bin(p, k)$.

Now, to evaluate the error probability of $B$ notice that

$$
    Pr(B \notin D) = Pr(Q \geq \frac{k}{2})
$$

Using Chernoff Bound

\begin{align*}
Pr(B \notin D) &= Pr(Q \geq \frac{k}{2}) \\
    &= Pr(Q \geq (1 + \frac{1  - 2p}{2p})pk) \\
    &< e^{- \frac{1 - 2p}{2p} \frac{pk}{3}} = e^{- \frac{1 - 2p}{6}k} \\
    &= e^{- \frac{1}{12} k } = e^{- \frac{1}{12} 12 \ln \delta^{-1}} \\
    &= \delta
\end{align*}

i.e. the probability of $Pr(B \notin D)$ is at most $\delta$ hence 

$$
    Pr(B \in [\frac{f(x)}{1 + \varepsilon}, (1 + \varepsilon)f(X)]) > 1 - \delta
$$

Algorithm $A$ runs in polynomial time in $\frac{1}{\epsilon}$ and $|x|$. $B$ runs $k = 12 \ln \delta^{-1} = O(\ln \delta^{-1})$ executions of $A$. Therefore, $B$ runs in $O(\frac{1}{\epsilon} |x| \ln \delta^{-1})$ time.

# Answer 3

Forall $i = 0, 1, ..., n-1$, let $X_i$ be a random variable which counts the number of flips made until the $(i + 1)^{st}$ side was tossed, given that there are $i$ sides that has been previously tossed. Because there are $n - i$ sides to choose from, we get that the probability to flip a new side is $\frac{n-i}{n}$. Thus, $X_i$ is drawn from a geometric distribution such that $X_i \sim G(\frac{n-i}{n})$.

Let $X$ be a random variable that counts the number of flips made until all of the $n$ sides where tossed. By this definition, we get that $X = \sum_{i=0}^{n-1}{X_i}$ and therefore, from the linearity of the expectation we get that $\mathbb{E}(X) = \sum_{i=0}^{n-1}{\mathbb{E}(X_i)}$.

According to known properties of the geometric distribution we conclude that

$$
    \mathbb{E}(X) = \sum_{i=0}^{n-1}{ \frac{n}{n-1} } = n \sum_{i=1}^{n}{ \frac{1}{n} }
$$
