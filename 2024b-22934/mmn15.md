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

