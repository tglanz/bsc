---
title: 22913, Mmn11
author: Tal Glanzman, 302800354
date: 26/07/2024
---

# Answer to 1

## a)

![hsi](./q1-hsi.png)

**Hue component**. There are 4 blocks, each corresponds to a block of the RGB image. In general, we see that the Hues corresponding to different Red/Green/Blue color are different. The top left and top right blocks are equal since they both arise from the Green color. The top right and bottom left corresponds to the Hue of the Red and Blue colors respectively.

**Saturation** values are $1$. Hence, the component is white.

**Intensity** values are $\frac{1}{3}$. Hence, the component is gray ($1/3$ of the way from black to white).

## b + c)

![averaged](./q1-averaged.png)

Because the Saturation is constant, the average is constant as well with the average equal to the constant value $1$. Therefore the averaged component is completely white.

The Hue averaging strode between 4 constant values blocks, each of size $250 \times 250$. Therefore, we can see that the $125 \times 125$ sized blocks at the outmost corners are the constant values themselves. The center is the averaging of the 4 blocks (the more towards the center the more equally weighted each block is).

# Answer to 2

The sensitivity to edges is caused because the error diffusion takes into account only neighbours of distance 1. To smooth the edges we can increase the kernel size so that the error will be diffused to farther pixels (which will also blur the image a bit as a side effect).

Below is a comparison between the Original image, an output of the Floyd and Steinberg algorithm using their proposed kernel and another output based on a larger kernel proposed by J. F. Jarvis, C. N. Judice and W.H. Ninke.

![error diffusion](./q2.png)

For completeness, the kernels used:
$$
Original = \frac{1}{16} \begin{pmatrix}
- & \star & 7 \\
3 & 5 & 1
\end{pmatrix}
~~~~~~
Larger = \frac{1}{48} \begin{pmatrix}
- & - & \star & 7 & 5 \\
3 & 5 & 7 & 5 & 3 \\
1 & 3 & 5 & 3 & 1
\end{pmatrix}
$$