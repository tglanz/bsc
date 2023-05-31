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

As mentioned, quantization is a mapping $Q: \mathbb{R} \rightarrow \mathbb{N}$.

There are many ways to define $Q$. A sensible way to define it, which is the most common, is to define it as a uniform, linear mapping. There are ways to define non-uniform mappings - we won't touch those here.

> By uniform, we mean that different subsets of the domain of the same size will be mapped to subsets of the range of the same size.

We call such a quantization scheme **Uniform affine quantization**:

$$
    x_{int} = Q(x) = int(\frac{x}{s}) - z
$$

$z \in \mathbb{N}$ is the **zero point** and it is used to shift the domain according to needs. Specifically, we can map a range with negative values to a domain of non-negatives allowing us to represent it using unsigned integers.

$s \in \mathbb{R}$ is the **scale factor**. It is the measure by which we shrink the range into the domain. Effectively, it can be thought of as the bucket size containing real numbers that are quantized to a single integer.

To get the initial, real-number $x$ back from the integer $x_{int}$ we use the inverse process, known as **de-quantization**:

$$
    x = Q^{-1}(x_{int}) = s\cdot x_{int} + z
$$

Note that $Q$ is not a one-to-one mapping! Therefore, $Q^{-1}$ is an approximation. We lose information when performing quantization.

For such quantization, we also define a **clipping range** $[\alpha, \beta] \subseteq \mathbb{R}$. The clipping range is a subset of the domain which we clip the input with.

We set the scale factor using the clipping range using:

$$
    S = \frac{\beta - \alpha}{2^b-1}
$$

Where $b \in \mathbb{R}$ is known as the **bit range** and it determines the size of the domain.

## Calibration

The process of determining the clipping range is a big part of quantization and it is known as **calibration**. 

When we calibrate the clipping range to be symmetric around 0, i.e. $\alpha = -\beta$, we say that the quantization is **symmetric** - non-symmetric quantizations are known as **asymmetric**. Symmetric quantizations allow us to choose $z = 0$, sparing computations at the cost of the clipping range being much wider than the input domain. Therefore, asymmetric quantizations, which have a potentially tight clipping range on the input domain, are mostly used on imbalanced operations such as ReLU (which is non-negative by definition).

Given an input domain $[x_{min}, x_{max}]$ we can calibrate the clipping range in multiple ways.

Most commonly, we set $\alpha = x_{min}$ and $\beta = x_{max}$ which is a symmetric quantization. A popular calibration for asymmetric quantization is done by setting $\beta = \max \{|x_{min}|, |x_{max}|\} = -\alpha$.

Neither of the calibrations above takes into account the distribution of the values within the input domain. To take into account this distribution, we can calibrate the clipping range according to specified percentiles (e.g. set $\alpha=percentile(1\%)$ and $\beta=percentile(99\%)$).

## Granularity

Given a specific architecture, it doesn't make real sense to quantize all of its tensors using the same quantization parameters. It is safe to assume that different tensors, at different layers of the architecture have different value ranges.

To give an example, refer to the fictive architecture in Figure \ref{diagrams-simple-cnn-architecture}. In this architecture, the input is a 3-channel tensor of values ranging from 0 to 255, potentially modeling an RGB image. Tensor $X_1$ is the output of a 2D convolution with real values (could be negative). Tensor $X_2$ is the output of a TanH operation, having values in the $(-1, 1)$ range. Similarly, $X_5$ is the output of a Softmax operation having values in the $(0, 1)$ range. There is no sensible calibration to properly quantize all tensors.

Take the calibration $[0, 255]$ for example. This clipping range is too loose for $X_4$, resulting in very low accuracy. In addition, assuming a uniform distribution of values, such calibration will clip half of the values of $X_2$.

The calibration $[-1, 1]$ is also problematic since it clips too many values of the tensors $X_0, X_1, X_3$ and $X_4$.

Another possible calibration is $[-255, 255]$ which is also too loose for the activation maps.

We are convinced that global calibration is problematic for this simplistic architecture let alone modern, complex architecture. The alternative is to calibrate and assign different clipping ranges for different substructures in the model.

There are multiple approaches for **granular calibration**:

- **Layerwise**. Calibration is performed at the granularity of a layer.
- **Groupwise**. Calibration is performed at the granularity of channel subsets.
- **Channelwise**. The specific case of Groupwise calibration with subsets of size 1.

Uncommonly, we can also calibrate at the sub-channel granularity. Regarding fully connected layers, we could calibrate specific neuron subsets.

Calibrations at finer granularities lead to better results for both accuracy and size reduction because we can set tight clipping ranges without clipping too many relevant values. However, the finer the granularity leads to high overheads - more parameters and computations.

![Simple CNN architecture.\label{diagrams-simple-cnn-architecture}](assets/diagrams-simple-cnn-architecture.drawio.png){width=90%}

## Training

Previous research work suggests that although possible, training a quantized model from scratch leads to lower accuracy than quantizing and retraining an already trained model.

We will explore the 2 main approaches for training a quantized model:

- **Quantization-Aware Training (QAT)**
- **Post-Training Quantization (PTQ)**

Refer to \ref{quantization-procedures} for the outline.

![QAT vs. PTQ. Source: "Neural Network Quantization for Efficient Inference".\label{quantization-procedures}](assets/quantization-procedures.png){width=90%}

### Quantization-Aware Training

Quantization of a trained model introduces errors. Quantization-Aware Training is the process of retraining such quantized models to increase the model's accuracy.

In QAT, the retraining is performed by using floating point numbers for the forward and backward passes to accurately compute the gradients respective to each of the parameters. After adjusting the parameters using floating point numbers, we quantize them.

Calculating the gradients should involve the calculation of the derivatives of the quantization mappings. The quantization mappings, as defined above are zero almost everywhere because they are piecewise constant functions. To overcome this, we use a different function as an approximation to the quantization mapping. A common function used for this purpose is the **Straight Through Estimator (STE)** which is practically an identity function. For a given quantization $Q(x) = int(x/s) - z$, its STE is the function $STE(x) = x/s - z$ with the derivative $\frac{d}{dx}STE = \frac{1}{s}$.

Refer to Figure \ref{qat-ste} for visualization of the QAT process.

![QAT learning using STE as an approximation. Source: "A Survey of Quantization Methods for Efficient
Neural Network Inference".\label{qat-ste}](assets/qat-ste.png){width=90%}

### Post-Training Quantization

There are 2, main downsides of QAT:

1. It is time and computationally expensive
2. It requires access to the whole training set

Post-Training Quantization is an alternative approach. In PTQ we don't retrain the model after quantization at all.

We can however perform additional calibrations and small weight adjustments by using different methods.

One such method is **Analytical Clipping for Integer Quantization (ACIQ)** which is proposed in [Q4; 2](#ref-q4). **ACIQ** is a technique to analytically compute an optimal clipping range.

Let $X$ be a random variable of a real-valued tensor with a laplacian distribution $Laplace(0, b)$. Assume we want to quantize $X$ uniformly to an integer grid with bit-width $M$ (we discussed this notion previously).

The author provides an equation for $\alpha$, such that the clipping range $[-\alpha, \alpha]$ minimizes the expected mean-squared-error $\mathbb{E}[(X - Q(X))^2]$, which effectively provides a clipping range that minimizes the squared distance of a tensor to its quantized tensor.

Initially, it is shown that (Eq. 5 in the paper):

$$
    \mathbb{E}[(X - Q(X))^2] \approx 2b^2e^{-\frac{\alpha}{b}} + \frac{\alpha^2}{3 \cdot 2^{2M}}
$$

Finally, to find the optimal value of $\alpha$ we solve the DE:

$$
    \frac{d}{d\alpha}[(X - Q(X))^2] = \frac{2\alpha}{3 \cdot 2^{2M}} - 2be^{-\frac{a}{b}} = 0
$$

The author claims that in practice, $b = \mathbb{E}[|X - \mathbb{E}[X]|]$ is a good estimation.

There are similar analyses for non-laplacian distributions, specifically Gaussian.