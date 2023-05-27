# Quantization

According to _Oxford_, the meaning of the term quantization is:

> A method of producing a set of discrete or quantized values that represents a continuous quantity

In the domain of neural networks, **Quantization** is the field of reducing the representation size of parameters from $n$ bits to $m$ bits such that $m < n$. To do so, the Quantization process embeds the space of real numbers into the space of integers.

Most neural networks today, before Quantization, use real numbers for computation. In contrast to integers, real numbers representations are far more complex and they are very costly compute-wise (see the [next section](#real-numbers-representations)).

Assuming the Quantization process can be performed without impacting the model's accuracy, it is highly valuable both from size/memory and computational resources perspectives.

## Real numbers representations

To understand the benefits of quantization we need to better understand how machines represent real numbers.

In machines, *real numbers* are usually represented using one of the formats:

- Fixed Point
- Floating Point

### Fixed Point

According to the fixed point format, given $n$ bits, A real number $x$ is represented by (from MSB to LSB):

- The **Sign** bit $s$.
- The **Integer Part** is the set of bits that are notated by $I$
- The rest of the bits, $m$ are known as the **Fractional Part** or the **Mantissa**.

Each system also has an **Implicit Exponent** notated by $E$.

Simply, we interpret a value $x$ by:

$$
    x = uint(I) + (uint(m) \cdot E)
$$

Where we notate $uint(X)$ to be the unsigned number represented by $X$.

In practice, $uint(I)$ is the integer to the left of the radix point and $uint(m)$ is the integer such that $uint(m) \cdot E$ is the integer to the right of the radix point.

See figure \ref{32-bits-fixed-point} for illustration.

For example, given we implicitly use $E = 0.001$, we can encode the value $- 12.064$ with:

- $S = 1$
- $I = 1100_2 = 12_{10}$
- $uint(m) = \frac{0.064}{0.001} \Rightarrow m=1000000_2 = 64_{10}$

### Floating Point

According to floating point IEEE 754 specification, given $n$ bits, A real number $x$ is represented by (From MSB to LSB):

- A bit, notated by $s$ that is known as the **Sign**
- Bits, notated by $E$ are known as the **Exponent**
- The rest of the bits, notated by $m$ are known as the **Mantissa**, or the **Significand**
x

See figure \ref{32-bits-single-precision} for illustration.

We also notate an exponent **Bias** by $B = 2^{e-1} - 1$, i.e. its half-range value.

When $E$ bits are not all zeros or all ones, we interpret $x$ as a **Normal Value** by:

$$
    x = (-1)^s \cdot (1 + frac(m)) \cdot (uint(E) - Bias)
$$

Where 

- $uint(E)$ is the unsigned integer represented by $E$ (in binary). i.e. $uint(E) = \sum_{i=0}^{|E|-1}{{E_i}2^i}$
- $frac(m)$ is the fractional value represented by $m$. i.e $frac(m) = \sum_{i=0}^{|m|-1}{{m_i}2^{i-|m|}}$

For example, given $n = 32$ and $|E| = 8$ we encode the value $x = 6$ with the representation $01000000110000000000000000000000_2$ since:

- The **Sign** is $0$
- The **Mantissa** is $10000000000000000000000_2 = 2^{-1} = 0.5_{10}$
- The **Exponent** is $10000001_2 = 129_{10}$
- The **Bias** is $2^7 - 1 = 127$

![The layout of a fixed point number. Source: "Neural Network Quantization for Efficient Inference".\label{32-bits-fixed-point}](assets/32-bits-fixed-point.png){width=90%}

And we get the formula:

$$
    -1^0 \cdot (1 + 0.5) \cdot 2^{129 - 127} = 1.5 \cdot 2^2 = 6
$$

If the $E = 0$, we interpret $x$ as a **Denormalized Value**. Denormalized values are also known as **Subnormal Values** because they represent values smaller than the smallest possible normalized value.

The only change in representation is that we don't implicitly phase shift the Mantissa by 1:

$$
    x = (-1)^s \cdot frac(m) \cdot 2^{-Bias}
$$

In addition, if both the Sign and the Mantissa are also 0, $x$ is interpreted as 0.

If the Exponent bits are all 1's, we interpret $x$ as one of the multiple special values, depending on the Sign and the Mantissa:

- **Infinity**. If the Mantissa is 0, $x$ is $\pm \infty$ when $Sign = 1, 0$ respectively.
- **NaN**. If the Mantissa is not 0, $x$ is considered to be not a number.

In neural networks, it is very common to use 32-bit numbers with an exponent of 8 bits - a.k.a single-precision numbers.

![The layout of a single precision number. Source: "Neural Network Quantization for Efficient Inference".\label{32-bits-single-precision}](assets/32-bits-single-precision.png){width=90%}

## The Quantization Mapping

As mentioned, quantization is a mapping $Q \vcentcolon \mathbb{R} \rightarrow \mathbb{N}$.

There are many ways to define $Q$. A sensible way to define it, which is the most common, is to define it as a uniform, linear mapping.

> By uniform, we mean that different subsets of the domain of the same size will be mapped to subsets of the range of the same size.

We call such a quantization scheme **Uniform affine quantization**, a.k.a **Asymmetric quantization** and it is defined by:

$$
    x_{int} = Q(x) = int(\frac{x}{s}) - z
$$

Where $s \in \mathbb{R}$ is known as the **scale factor** and $z \in \mathbb{N}$ is the **zero-Point**.

To get the initial real-number $x$ back from the integer $x_{int}$ we use the inverse process, known as **de-quantization**:

$$
    x = Q^{-1}(x_{int}) = s\cdot x_{int} + z
$$

Note that $Q$ is not a one-to-one mapping! Therefore, $Q^{-1}$ is an approximation. We lose information when performing quantization.

When we set $z=0$ we call the quantization **Symmetric** - it is a specific case of asymmetric quantizations. Without the need to offset by the zero point we reduce the computations required by de-quantization. However, asymmetric quantization has the potential to retain higher accuracy than symmetric quantization since we can offset imbalances in the range of the weights using the zero point. Such imbalances appear due to many reasons such as zero padding (to match tensor sizes), outputs of ReLU activations etc.