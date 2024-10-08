# Mmn14, 22913

Author: Tal Glanzman

Date: 2024/09/27

# Answer to 1

## 8.17

> Question 8.17 in the book

We compute the subinterval according the algorithm

![img](./q1.jpg)

and get the final message: $0.144$.

## 8.18

> Question 8.18 in the book

Back tracking sub intervals like in the previous eercise, we get that the message is eaii!.

![img](./q1.2.jpg)

\newpage
# Answer to 2

## 8,7

> Question 8.7 in the book

Denote the symbols by $x_i$ and thei respective probabilities $p_i$ for ${i = 1, ..., q}$.

We want to show that $\log q - H \geq 0$ and equality is acheived if and only if $p_i = \frac{1}{q}$ for all $1 \leq i \leq q$.

Recall that $\sum_{i=1}^q p_i = 1$ and observe that
\begin{align*}
\log q - H &= \log q + \sum_{i=1}^q p_i \log p_i \tag{By 8-6} \\
&= \log q \sum_{i=1}^q p_i + \sum_{i=1}^q p_i \log p_i \\
&= \sum_{i=1}^q p_i \log q + \sum_{i=1}^q p_i \log p_i \\
&= \sum_{i=1}^q p_i \log q + p_i \log p_i \\
&= \sum_{i=1}^q p_i ( \log q + \log p_i) \\
&= \sum_{i=1}^q p_i \log q p_i \\
\end{align*}

From logarithm base conversion rules, we know that
$$
\log_2 x = \frac{1}{\log_e 2} log_e x
$$

In standard notation:
$$
\log x = \frac{\ln x}{\ln 2}
$$

We get that

$$
\log q - H = \ln 2 \cdot \sum_{i=1}^q p_i \ln q p_i
$$

Based to the inequality $\ln x \leq x -1$ $(*)$ we get that
$$
\ln \frac{1}{q p_i} \leq \frac{1}{q p_i} - 1 \Longrightarrow \ln q p_i \geq 1 - \frac{1}{q p_i}
$$

and thus we get
\begin{align*}
\log q - H & \geq \ln 2 \cdot \sum_{i=1}^q p_i (1 - \frac{1}{q p_i}) \\
&= \ln 2 \Biggr[\sum_{i=1}^q p_i - \sum_{i=1}^q \frac{1}{q} \Biggl] \\
&= \ln 2 \Biggr[1 - 1 \Biggl] \\
&= 0
\end{align*}

and we have shown that $H \geq \log q$.

Furthermore, equality in $(*)$ is achieved **if and only if** $x = 1$. Putting it in our expressions, we conclude that $H = \log q$ if and only if $\frac{1}{q p_i} = q$ which happens if and only if $p_i = \frac{1}{q}$ for all $1 \leq i \leq q$.

## 8.8

> Question 8.8 in the book

Let $x_1, x_2, x_3$ be the symbols and let $p_i$ be their probabilities respectively.

There are $3! = 6$ ways to permute $x_1, x_2, x_3$ such that
$$
x_i \leq x_j \leq x_k
$$

which will yield 6 encodings, however they are not unique.

The number of unique Huffman codes will be 2 becuase the algorithm will work by merging the 2 lower probability symbols and the assigning ${b \in \{ 0, 1 \}}$ to the third symbol. The two-bit code assigned to the pair is directly related to $b$. Because there are 2 options to assign $b$, there are 2 options for the code.


Assume w.l.g. that $p_1 < p_2 < p_3$.

**Case 1** $b = 0$:

$p_3 \rightarrow 0$

$p_2 \rightarrow 00$

$p_1 \rightarrow 10$

**Case 2** $b = 1$:

$p_3 \rightarrow 1$

$p_2 \rightarrow 01$

$p_1 \rightarrow 11$

\newpage
# Answer to 3

> Question 8.19 in the book

According to 7-bit ascii, we initialize the dictionary as follows

Location | Entry
-|-
0 | $\cdot$
$\vdots$ | $\cdot$
97 | a
$\vdots$ | $\cdot$
177 | $\cdot$

where $\cdot$ is a symbol we don't care about.

Let'e encode the string "$a^{11}$".

Reading "a" at index 1, we will emit 97 to the code and add "aa" to location 178 in the dictionary.

Reading "a" at indices 2, 3, we will emit 178 to the code and add "aaa" to location 179 in the dictionary.

Reading "a" at indices 4, 5, 6, will emit 179 to the code and add "aaaa" to location 180 in the dictionary.

Reading "a" at indices 7, 8, 9, 10, we will emit 180 to the code and add "aaaa" to location 181 in the dictionary.

Reading "a" at index 11, we will emit 97 to the code.

Finally, the code will be:
$$
97, 178, 179, 180, 97
$$

and the dictionary:

\newpage
Location | Entry
-|-
0 | $\cdot$
$\vdots$ | $\cdot$
97 | a
$\vdots$ | $\cdot$
177 | $\cdot$
178 | aa
179 | aaa
180 | aaaa
181 | aaaaa

\newpage
# Answer to 4

> Question 8.5 in the book

a)

According to Shannon's noiseless theorem
$$
L_{avg} \geq H = 5.3 \frac{bits}{pixel}
$$

thus, the compression ratio is at most 
$$
\frac{1024 \times 1024\times 8}{1024 \times 1024 \times 5.3} \approx 1.51
$$

and the compressed image is of size of at least
$$
\lceil 1024 \times 1024 \times 5.3 \rceil ~bits
$$

b) 

No due to the required ceil.

Assuming it is obtained, we could represent fractional bits which is not the case.

c)

Huffman encodes symbols one at a time. Thus, it has to spatial awareness. In case the image has any spatial correlation Huffman code won't capture it.

On the contrary, LZW has spatial awareness and it can map repeating sequences to one-symbol.

Thus, I suggest to perform an LZW compression followed by Huffman. The LZW emits a fixed-length code which then the Huffman encodes to a variable-length code. In that way we have a compression that can leverage both spatial correlations and symbol frequencies.