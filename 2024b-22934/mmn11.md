---
title: 22943, Mmn11
author: Tal Glanzman
date: 02-04-2024
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
