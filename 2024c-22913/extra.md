# Question 4.28, page 311

> Multiple fourier pairs. Impulse, phasor, trig.

By convention, in all of the answers below we will notate:

$$
\sum_{x} = \sum_{x=0}^{M-1} ~~ ; ~~ \
\sum_{y} = \sum_{y=0}^{N-1} ~~ ; ~~ \
\sum_{u} = \sum_{u=0}^{M-1} ~~ ; ~~ \
\sum_{v} = \sum_{v=0}^{N-1} ~~ ; ~~ \
$$

We will also allow ourselves to use two variables under the same summation sign, meaning independent summations according to the above conventions. For example
$$
    \sum_{xy} = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1}
$$

## a)

We will show that
$$
\delta(x, y) \Longleftrightarrow 1
$$

\begin{align*}
\mathcal{F}[\delta(x, y)] 
=& \sum_{xy}\delta(x, y) e^{-2\pi i (ux/M + vy/N)} && \text{By 4-67} \\
=& e^{-2\pi i (u0/M + v0/N)} && \text{By 4-58} \\
=& 1
\end{align*}

## b)

We will show that
$$
1 \Longleftrightarrow MN \delta(u, v)
$$

\begin{align*}
\mathcal{F}^{-1}[MN\delta(u,v)]
&= \frac{1}{MN} \sum_{uv} MN \delta(u,v) e^{2\pi i (ux/M + vy/N)} && \text{By 4-68} \\
&= e^{2\pi i (0x/M + 0y/N)} && \text{By 4-58} \\
&= 1
\end{align*}

## c)

We will show that
$$
\delta(x - x_0, y - y_0) \Longleftrightarrow e^{-j2\pi(ux_0/M + vy_0/N)}
$$

\begin{align*}
\mathcal{F}[\delta(x-x_0, y-y_0)] 
=& \sum_{xy}\delta(x-x_0, y-y_0) e^{-2\pi i (ux/M + vy/N)} && \text{By 4-67} \\
=& e^{-2\pi i (ux_0/M + vy_0/N)} && \text{By 4-58}
\end{align*}

## d)

We will show that
$$
e^{j2\pi(u_0x/M + v_0y/N)} \Longleftrightarrow MN \delta(u - u_0, v - v_0)
$$

\begin{align*}
\mathcal{F}^{-1}[MN\delta(u-u_0,v-v_0)]
&= \frac{1}{MN} \sum_{uv} MN \delta(u-u_0,v-v_0) e^{2\pi i (ux/M + vy/N)} && \text{By 4-68} \\
&= e^{2\pi i (u_0x/M + v_0y/N)} && \text{By 4-58}
\end{align*}

## e)

We will show that
$$
\cos(2\pi u_0 x/M + 2\pi v_0 y/N) \Longleftrightarrow (MN/2)[\delta(u + u_0, v + v_0) + \delta(u - u_0, v - v_0)]
$$

Observe that
$$
\frac{e^{i\theta} + e^{-i\theta}}{2} = \frac{\cos(\theta) + i\sin(\theta) + \cos(-\theta) + i\sin(-\theta)}{2} = \cos(\theta)
$$

So
\begin{align*}
&\mathcal{F}[\cos(2\pi(u_0x/M + v_0y/N))] \\
&= \sum_{xy} \frac{e^{2\pi i (u_0x/M + v_0y/N)} + e^{- 2\pi i (u_0x/M + v_0y/N)} }{2} e^{- 2\pi i (ux/M + vy/N)} && \text{By 4-67} \\
&= \frac{1}{2} \sum_{xy} e^{2\pi i (u_0x/M + v_0y/N)} e^{- 2\pi i (ux/M + vy/N)} + \frac{1}{2} \sum_{xy} e^{- 2\pi i (u_0x/M + v_0y/N)} e^{- 2\pi i (ux/M + vy/N)} \\
&= \frac{MN}{2} \delta(u-u_0, v-v_0) + \frac{MN}{2} \delta(u+u_0, v+v_0) && \text{By answer to 4.d} \\
&= \frac{MN}{2} [ \delta(u+u_0, v+v_0) + \delta(u-u_0, v-v_0) ]
\end{align*}

## f)

We will show that
$$
\sin(2\pi u_0 x/M + 2\pi v_0 y/N) \Longleftrightarrow (jMN/2)[\delta(u + u_0, v + v_0) + \delta(u - u_0, v - v_0)]
$$

Observe that
$$
\frac{e^{i\theta} - e^{-i\theta}}{2i} = \frac{\cos(\theta) + i\sin(\theta) - \cos(-\theta) - i\sin(-\theta)}{2i} = \sin(\theta)
$$

So
\begin{align*}
& \mathcal{F}[\sin(2\pi(u_0x/M + v_0y/N))] \\
&= \sum_{xy} \frac{e^{2\pi i (u_0x/M + v_0y/N)} - e^{- 2\pi i (u_0x/M + v_0y/N)} }{2i} e^{- 2\pi i (ux/M + vy/N)} && \text{By 4-67} \\
&= \frac{1}{2i} \sum_{xy} e^{2\pi i (u_0x/M + v_0y/N)} e^{- 2\pi i (ux/M + vy/N)} - \frac{1}{2i} \sum_{xy} e^{- 2\pi i (u_0x/M + v_0y/N)} e^{- 2\pi i (ux/M + vy/N)} \\
&= \frac{MN}{2i} \delta(u-u_0, v-v_0) - \frac{MN}{2i} \delta(u+u_0, v+v_0) && \text{By answer to 4.d} \\
&= \frac{i MN}{2} [ \delta(u+u_0, v+v_0) - \delta(u-u_0, v-v_0)]
\end{align*}


## Question 8.7, page 632

> Prove that for zero memory source with $q$ symbols, the maxium value of the entropy is $\log q$, which is acheived if and only if all source symbols are quiprobable

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

## Question 8.8, page 632

> How many huffman codes, and construct them, are for a three symbol source?

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

# Question 8.19, page 633

> Use the LZW coding algorithm to encode the 7-bit ASCII string "aaaaaaaaaaa" (11 times a)

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

# Question 8.5, page 632

> A $1024 \times 1024$ 8-bit image with 5.3 bits/pixel entropy [computed from eq 8-7] is to be huffman coded.

### a) What is the maximumm compression that can be expected?

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

### b) Will it be obtained? 

No due to the required ceil.

Assuming it is obtained, we could represent fractional bits which is not the case.

### c) If a greater lossless compression is required, what else can be done?

Huffman encodes symbols one at a time. Thus, it has to spatial awareness. In case the image has any spatial correlation Huffman code won't capture it.

On the contrary, LZW has spatial awareness and it can map repeating sequences to one-symbol.

Thus, I suggest to perform an LZW compression followed by Huffman. The LZW emits a fixed-length code which then the Huffman encodes to a variable-length code. In that way we have a compression that can leverage both spatial correlations and symbol frequencies.