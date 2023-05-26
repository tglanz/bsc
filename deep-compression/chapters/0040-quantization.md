# Quantization

According to _Oxford_, the meaning of the term quantization is:

> A method of producing a set of discrete or quantized values that represents a continuous quantity

In the domain of neural networks, **Quantization** is the field of reducing the representation size of parameters from $n$ bits to $m$ bits such that $m < n$.

To understand how we can decrease the representation bits we first need to understand the common representations in use.

In machines, *real numbers* are usually represented using one of the formats:

- Floating Point
- Fixed Point

### Floating Point

According to floating point IEEE 754 specification, given $n$ bits, A real number $x$ is represented by:

- A 1 bit, notated by $s$ is known as the **Sign**
- $|E|$ bits, notated by $E$ are known as the **Exponent**
- The rest of the bits, $m$ are known as the **Mantissa**, or the **Significand**
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

- **Sign** is $0$
- **Mantissa** is $10000000000000000000000_2 = 2^{-1} = 0.5_{10}$
- **Exponent** is $10000001_2 = 129_{10}$
- **Bias** is $2^7 - 1 = 127$

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