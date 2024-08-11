# Mmn12, 22913

Author: Tal Glanzman

Date: 2024/08/10

# Answer to 1

... TBD code reference

# Answer to 2

## 2.1

$$
\begin{matrix}
& & & -1 & 2 & -1 \\
3 & 4 & 2 & 1
\end{matrix}
$$

$\Rightarrow y_0 = -1 \cdot 1 = -1$

$$
\begin{matrix}
& & -1 & 2 & -1 \\
3 & 4 & 2 & 1
\end{matrix}
$$

$\Rightarrow y_1 = 2 \cdot 1 + (-1) \cdot 2 = 0$

$$
\begin{matrix}
& -1 & 2 & -1 \\
3 & 4 & 2 & 1
\end{matrix}
$$

$\Rightarrow y_2 = -1 \cdot 1 + 2 \cdot 2 + (-1) \cdot 4 = -1$

$$
\begin{matrix}
-1 & 2 & -1 \\
3 & 4 & 2 & 1
\end{matrix}
$$

$\Rightarrow y_3 = -1 \cdot 2 + 2 \cdot 4 + (-1) \cdot 3 = 3$

$$
\begin{matrix}
-1 & 2 & -1 \\
& 3 & 4 & 2 & 1
\end{matrix}
$$

$\Rightarrow y_4 = -1 \cdot 4 + 2 \cdot 3 = 2$

$$
\begin{matrix}
-1 & 2 & -1 \\
& & 3 & 4 & 2 & 1
\end{matrix}
$$

$\Rightarrow y_5 = -1 \cdot 3 = -3$

Therefore

$$
y = h \circledast x =
\begin{pmatrix} -1 \\ 2 \\ -1 \end{pmatrix}
\circledast
\begin{pmatrix} 1 \\ 2 \\ 4 \\ 3 \end{pmatrix}
=
\begin{pmatrix}
-1 \\ 0 \\ -1 \\ 3 \\ 2 \\ -3
\end{pmatrix}
$$

## 2.2

By definitions of the Fourier Pair

$$
f(x, y)
= \mathcal{F}^{-1}[\hat{f}(u, x)]
= \int_{-\infty}^{\infty} e^{- \frac{u^2 + v^2}{2 \sigma^2}} e^{j2\pi (ux + vy)} \mathrm{d}u \mathrm{d}v
= g(x) \cdot g(y)
$$

with $g(t)$ defined by:
$$
    g(t) = \int_{-\infty}^{\infty} e^{- \frac{\omega^2 - j4\pi\sigma^2t\omega}{2\sigma^2}} \mathrm{d}\omega
$$

Notice that by complementing to the square, we get that
$$
\omega^2 - j4\pi\sigma^2t\omega = (\omega - j2\pi\sigma^2 t)^2 + 4\pi^2\sigma^4 t^2
$$

and therefore
$$
g(t) 
= e^{- 2 \pi^2 \sigma^2 t^2} \cdot \int_{-\infty}^{\infty} e^{- \frac{(w - j2\pi \sigma t)^2}{2\sigma^2}} \mathrm{d}\omega
$$

By using the fact about Gaussian Integrals
$$
\int_{-\infty}^{\infty} e^{-a(t + b)^2}\mathrm{d}t = \sqrt{\frac{\pi}{a}}
$$

we compute
$$
\int_{-\infty}^{\infty} e^{- \frac{(\omega - j2\pi \sigma t)^2}{2 \sigma^2}} \mathrm{d}\omega = \sqrt{2 \sigma ^2 \pi} = \sigma \sqrt{2\pi}
$$

And we get that
$$
g(t) = \sigma \sqrt{2\pi} \cdot e^{- 2 \pi^2 \sigma^2 t^2}
$$

Finally, putting it all together we get that
$$
f(x, y) = g(x) \cdot g(y) = 2\pi \sigma^2 e^{- 2 \pi^2 \sigma^2 (x^2 + y^2)}
$$

as we wanted to show.