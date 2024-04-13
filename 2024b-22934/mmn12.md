---
title: 22943, Mmn12
author: Tal Glanzman
date: 13-04-2024
...

# Answer 1

The algorithm $A$ will run as follows:

1. Set $k = \epsilon^{-2}$
2. Maintain a multiset $R$ with $k$ tokens in the exact same way as the algorithm _"Uniform Sampling Without Replacements"_

At any point of the stream, the **Query($S$)** operation will return $\frac{| R \cap S |}{k}$ (Using multiset semantics).

The space complexity of $A$ is exactly like the "Uniform Sapmpling Without Replacements" algorithm - $O(\lg n + k \lg m) = O(\lg n m^{\epsilon^{-2}}$ based on:

- $O(\lg n)$ to keep track of $n$
- $O(k \lg m) = O(\epsilon^{-2} \lg m)$ to maintain $R$


Denote the tokens of $R = \{ t_i \}_{i=1}^{k}$.

Forall $i = 1, 2, ..., k$ we define an indicator $X_i$ for the event that $t_i \in S$ which has the probability $\frac{|S|}{m}$.

Let $X$ be a random variable that counts the tokens that are in $R \cap S$, i.e.:

$$
    X = \sum_{i=1}^{k} X_i \sim B(k, \frac{|S|}{m})
$$

Lastly denote the random variable $A = \frac{X}{k}$ which is the result of the algorithn $A$. We want to show that $Pr(|A - \mathbb{E}(A)| > \epsilon) < \frac{1}{2}$.

Using Chebyshev:

\begin{align*}
    Pr(|A - \mathbb{E}(A)| > \epsilon) &< \frac{Var(A)}{\epsilon^2} = \frac{1}{\epsilon^2 k^2} Var(X) \\
                                       &= \frac{1}{k} \frac{k |S|}{m} (1 - \frac{|S|}{m}) \frac{|S|}{m}(1 - \frac{|S|}{m}) \\
                                       &< \frac{1}{2}
\end{align*}

where the last inequality results from the fact that $p (1 - p) < \frac{1}{2}$ forall $0 \leq p \leq 1$.

# Answer 2

## a)

The required algorithm $B$ is defined as follows:

1. Initialize $k = 2$ (token, counter) pairs
1. **while** there are more tokens **do**
    1. **Let** $t$ be the next token
    1. Increase the counter of $t$ by one
    1. **If** there are at least 2 non-zero counters **do**
        1. Decrease all non-zero counters by 1
1. Return the token of the single non-zero counter

The suggested algorithm relies on the fact that there is a token $t$ that appears more than $\frac{n}{2}$ times in the stream. Similiarly to the proof of **Lemma 1**, we know that $t$ will be in the final counters set. The others tokens must apper less that $\frac{n}{2}$ so they won't survive the $\frac{n}{2}$ decrements.

Effectively, we run the first pass of the "Frequent Elements Algorithm (k)" from chapter 1, with $k = 2$ and return the token at the single non-zero counter of $F$.

Each counter holds two variables:

1. A token $t \in \{1, 2, ..., m \}$ which the counter tracks 
1. A value $c \in \{1, 2, ..., n \}$ which is the counter's value

Thus, the space complexity of the algorithm is $O(\lg m + \lg n) = O( \lg mn )$.

## b)

The algorithm $C$ utilizes the algorithms $A$ and $B$ (with $k = \epsilon^{-2}$ from the previous sections. By utilizing $B$, algorithm $C$ yields the token $t \in \{ 1, 2, ..., m \}$ with frequency larger than $\frac{n}{2}$. By utilizing $A$, using the query $S = \{ t \}$, it returns the fraction of tokens that are in $S$ - all we have left is to multiply that result by $n$ to get the estimated frequency. 

The reason that approximation of the algorithm is $\pm \epsilon$ with probability $\frac{1}{2}$ follows immediately from its usage of algorithm $A$.

The overall space requirements of the algorithm is the sum of the space requirements of the algorithms $A$ and $B$:

$$
    O(\lg mn + \lg m^{\epsilon^{-2}}n) = O(\lg m^{\epsilon^{-2}}n)
$$
