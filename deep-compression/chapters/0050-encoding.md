# Weight Encoding

Pruning and Quantization are both techniques that reduce neural network size by modifying the network or its representation. **Weight Encoding** is a technique to further reduce the network size by efficiently encoding and compressing its parameters.

There are two main classifications for encoding algorithms: **Lossless encoding** is the class of encoding/compression algorithms that preserves the data to its fullness - No bit is lost (hence, Lossless). In contrast, **Lossy encoding** is the class of encoding/compression algorithms that lose some information.

The **Deep Compression** framework suggested using Huffman Coding for parameter compression after the Pruning and Quantization phases [[2; 4]](#ref-1).

Although most of the research was made regarding Pruning and Quantization, some of the research was made regarding weight compression. 

## Weightless

In contrast to (the lossless) Huffman Coding, **Weightless** [[C2]](#ref-c2) is a lossy encoding method - It utilizes the probabilistic data structure:

**Bloomier Filter** [[C2; 3.1]](#ref-c2) is a probabilistic data structure for space-efficient encoding of mappings.

Formally, let $S$ be a subset of some domain $U$ and a mapping $f : S \rightarrow R[0, 2^r)$. Given an input $v$, A bloomier filter will return $f(v)$ for every $v \in S$ and for every $v \in D/S$ it will return $null$ with some small error probability in which it will return a value in $R$. 

The bloomier filter is composed of $k$ hash functions $H_0, H_2, ..., H_{k-1}$ and a hash table $X$ with $m > |S|$ entries, each containing a value with $t > r$ bits. The hash functions $H_0, H_2, ..., H_{k-2}$ are defined as mappings from the input domain $S$ to a cell of $X$, i.e. $H_{1,2,..., k-2} : S \rightarrow [0, m)$ and the last hash acts as a mask $H_{k-1} : S \rightarrow [0, 2^r)$ and will also be denoted as $H_M$.

$X$ is constructed in such a way that for every $v \in S$:

$$
    H_M(v) \bigoplus_{i=0}^{k-2}{X_{H_i(v)}} = f(v)
$$

Meaning, we access the relevant entries in $X$ according to the mappings and XOR them together alongside the mask.

For every $v \notin S$ we get that $H_M(v) \bigoplus_{i=0}^{k-2}{X_{H_i(v)}}$ is distributed randomly over the range $[0, 2^t]$. If $f(v) is not in the sub-range $[0, 2^r]$ we return $null$ but if it does, we return the error. 

**Weightless** proposed to use this space-efficient data structure to compactly store the weights by encoding a mapping from the original weights $W$ to a more compact representation $W'$. This is done by associating each **non-zero** weight $w_{ij}$ to $w'_{ij}$ - we encode $S$ to be the set of indices $(i, j)$ such that $w_{ij} \neq 0$. For every $v \in S$ the mapping $f(v)$ yields a correct result. For originally zero-weights, $f(v)$ returns $null$ (which is treated as 0) in a high probability - in the case of error, it will return some arbitrary number.

In Figure \ref{bloomier-weights}, we see an illustration of the processes of querying correctly and a false positive.

![Encoding weights using Bloomier Filter. Source: "Weightless: Lossy weight encoding for deep neural network compression"\label{bloomier-weights}](assets/bloomier-weights.png){width=90%}

In Bloomier Filter, the more we increase $t$ the more information each table cell can encode, lowering the error probability. The lower the error probability, the more accurately $W'$ models $W$, leading to higher accuracy of the model.

Figure \ref{bloomier-relations} shows this inverse relationship between the number of false positives (red) and the model's accuracy (blue) which are both exponentially related to $t$.

![Inverse relation of the false positives and the model's accuracy. Source: "Weightless: Lossy weight encoding for deep neural network compression"\label{bloomier-relations}](assets/bloomier-false-positives-vs-accuracy.png){width=90%}