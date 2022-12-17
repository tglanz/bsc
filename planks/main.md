---
title: בעיית הקרשים הממשית
author:
- טל גלנצמן
- בהנחיית ד״ר בועז סלומקה
...

# הקדמה

## הקדמה

כמה מלבנים בגודל
$1 \times k$
נדרשים על מנת לכסות דיסק בעל קוטר $k$?

&nbsp;
להכניס איור

. . .

לפחות במישור, האינטואיציה תוביל אותנו להשערה שנדרשים לפחות
$k$
מלבנים כאלה

&nbsp;
להכניס איור

## הקדמה

**בעיית הקרשים** עוסקת בקיום ותכונות של כיסוי **גופים קמורים** באמצעות אובייקטים הנקראים **קרשים**.

## הקדמה

כאן נעסוק בכיסוי של **גופים קמורים**, ובעיקר **סימטריים**, במרחבים **ממשיים בעלי מימד סופי**.

# גופים קמורים

### הגדרה
**גוף קמור**
$K \in \mathbb{R}^d$
הוא קבוצה קומפקטית, בעלת פנים לא ריק כך שלכל
$x, y \in K$
ולכל
$\lambda \in [0, 1]$
$$(1 - \lambda)x + \lambda y \in K$$

---

## על-מישור תומך

### הגדרה

לכל
$y \in \mathbb{R}^d$
ו-
$w \in \mathbb{R}$
,
קבוצה $L$ מהצורה
$\{ x : \langle y, x \rangle = w \}$
תקרא **על-מישור**.

. . .

&nbsp;

**כיוון העל-מישור**
$L$
הוא וקטור הכיוון
$\frac{y}{||y||} \in \mathbb{S}^d$
.

. . .

&nbsp;

העל מישור $L$ חוצה את $\mathbb{R}^d$ לשני חצאי מרחב
$L^+ = \{ x : \langle y, x \rangle \geq w \}$
ו-
$L^- = \{ x : \langle y, x \rangle \leq w \}$
.

. . .

&nbsp;

על-מישור
$L$
ייקרא **על-מישור תומך** של
$K$
כאשר
$K \cap L \neq \phi$
וגם
$K \subseteq L^-$
או
$K \subseteq L^+$
.

&nbsp;
להכניס איור

::: notes

- Solutions of a linear system

- There are other definitions for hyperplanes, e.g CoDim1. CoDim1 gives a bit of insight about the "why plank" question

:::

## רוחב מכוון

### הגדרה

**התומך** של
$K$
הוא הפונקציונל הלינארי
$h_K(\theta) = \sup_{x \in K} \langle x, \theta \rangle$

. . .

&nbsp;

**הרוחב של $K$ בכיוון $\theta$** 
,
$w_\theta(K)$
הוא המרחק האוקלידי בין שני העל-מישורים
 התומכים $\{ x : x \cdot (\pm \theta) = h_K(\pm \theta) \}$
ומתקיים
$$w_\theta(K) = h_K(\theta) + h_K(- \theta)$$

. . .

אם $K$ סימטרי אז
$w_\theta(K) = 2h_K(\theta)$

::: notes

- התומך בכיוון טטה מודד את את המרחק אל העל מישור התומך בכיוון טטה

:::

&nbsp;
להכניס איור

## רוחב מינימלי

**הרוחב המינימלי** של
$K$
הוא הרוחב המכוון המינימלי של
$K$
מבין כל הכיוונים
$\theta \in \mathbb{S}^d$

$$w(K) = \min_{\theta \in \mathbb{S}^d}{w_\theta(K)}$$

. . .

###

וכאשר
$K$
סימטרי

$$w(K) = 2 \min_{\theta \in \mathbb{S}^d}{w_\theta(K)}$$

# בעיית הקרשים של $Tarski$

## קרש

### הגדרה

קבוצה
$P$
מהצורה
$\{ x : |\langle x, y \rangle - m| \leq w \}$
תקרא **קרש**.

. . .

&nbsp;

$P$
הוא התחום הכלוא בין שני העל-מישורים
$\{ x : \langle x, y \rangle = w \pm m \}$

. . .

&nbsp;

**כוון הקרש**
$P$
הוא וקטור הכיוון
$\frac{y}{||y||}$
.

&nbsp;

**רוחב הקרש**
$P$
$\frac{2w}{||y||}$
.

. . .

&nbsp;

**הרוחב של גוף קמור
$K$
ביחס לקרש
$P$**
מוגדר ע״י
$w_P(K) = \frac{w(P)}{w_K(\frac{y}{||y||})}$
.

כאשר
$K$
סימטרי
$w_P(K) =  \frac{\frac{2w}{||y||}}{2h_K(\frac{y}{||y||})} = \frac{w}{h_K(y)}$
.

::: notes

- why plank? Plank frees a single dimension to choose a fitting width. We can always replace a plank with a convex body translating a plank covering problem to a convex covering problem. 

:::

## 1930 Tarski,
ב-
$1930$
העלה המתמטיקאי
$Alfred~Tarski$
[^1]
את ההשערה הבאה

&nbsp;

### השערת $Tarski$

יהי
$K$
גוף קמור ו-
$A$
אוסף סופי של קרשים. אם
$K \subseteq \bigcup_{P \in A} A$
אז
$\sum_{P \in A}{w(P) \geq w(K)}$

. . .

&nbsp;

$Tarski$ הוכיח השערה זו למקרה הפרטי עבור דיסקים ב-$\mathbb{R}^2$

[^1]: https://en.wikipedia.org/wiki/Alfred_Tarski

## 1950 Bang,

ב-
$1950$
הוכיח המתמטיקאי
$Thoger~Bang$ 
את השערת $Tarski$.

. . .

&nbsp;

בהוכחתו, העלה השערה חדשה

&nbsp;

### השערת $Bang$

יהי
$K$
גוף קמור ו-
$P$
אוסף סופי של קרשים. אם
$K \subseteq \bigcup_{P \in A} P$
אז
$$\sum_{P \in A} w_A(K) \geq 1$$

## Bang 1950

ההשערה של
$Bang$
קשה יותר מ-
$Tarski$

$$1 \leq \sum_{P \in A}{w_A(K)} = \sum_{P \in A}{\frac{w(A)}{w_\theta(K)}} \leq \frac{1}{w(K)} \sum_{P \in A}{w(A)}$$

ולכן

$$w(K) \leq \sum_{P \in A}{w(A)}$$

::: notes

- Right inequality is by definition of **minimal** width

- Conceptually, Tarski measures the Planks with no relation to the body. Bang tightens the measurement.

:::

# משפט Ball

## 1990 Ball,

השערתו של 
$Bang$
נותרה פתוחה עד היום!

. . .

ב-
$1990$
הוכיח המתמטיקאי
$Keith~Ball$[^2]
את ההשערה של
$Bang$
במקרה הפרטי והחשוב עבור גופים סימטרים. 

. . .

&nbsp;

### משפט $Ball$

יהי
$K$
גוף **סימטרי** קמור ו-
$A$
אוסף **בן-מניה** של קרשים. אם
$K \subseteq \bigcup_{P \in A} P$
אז
$$\sum_{P \in A} w_P(K) \geq 1$$

&nbsp;

כדי להוכיח את המשפט, $Ball$ מתרגם את הבעיה תחילה לעולם האנליזה הפונקציונלית, ומשם לאגברה לינארית.

[^2]: https://en.wikipedia.org/wiki/Keith_Martin_Ball

## רדוקציה לאנליזה פונקציונלית Ball,

### משפט $Ball~1$

יהי
$(\phi_i)_{1}^{\infty}$
אוסף בן-מניה
של פונקציונלים
לינארים ב-
$\mathbb{R}^d$
בעלי נורמה 1
,
מספרים ממשיים
$(m_i)_{1}^{\infty}$
ומספרים חיוביים
$(w_i)_{1}^{\infty}$
כך שלכל
$x \in B_{||\cdot||}(1) \subset \mathbb{R}^d$
קיים
$i$
שעבורו
$$| \phi_i(x) - m_i | \leq w_i$$

אז
$$\sum_i w_i \geq 1$$

## רדוקציה לאנליזה פונקציונלית Ball,

### הגדרה

**הנורמה האופרטורית** של פונקצינל לינארי
$\phi : (\mathbb{R}^d, ||\cdot||) \rightarrow \mathbb{R}$
מוגדרת על ידי
$$||\phi||_{OP} = \sup_{||x|| \leq 1}{||\phi(x)||}$$
.

. . .

&nbsp;

כל פונקציונל לינארי
$\phi : \mathbb{R}^d \rightarrow \mathbb{R}$
ניתן להצגה כמכפלה פנימית
$\phi(x) = \langle x, y \rangle$
עבור
$y \in \mathbb{R}^d$
כלשהו

$$\{ x : |\langle x, y \rangle - m| \leq w \} = \{ x : |\phi(x) - m| \leq w \}$$

## רדוקציה לאנליזה פונקציונלית Ball,

### Ball 1 <= Ball

אבחנה מרכזית לשם הצגת השקילות היא שכל גוף קמור סימטרי $K$ מגדיר מרחב נורמי 
$(\mathbb{R}^d, ||\cdot||_K)$
ש-
$K$
הוא כדור היחידה שלו. כלומר,
$K = B_{||\cdot||_K}(1)$
ומתקיים
$||\phi||_{K_{OP}} = h_K(y)$
כאשר
$\phi(x) = \langle x, y \rangle$
.

&nbsp;

יהיו $K$
גוף קמור סימטרי
ו- 
$P_i = \{ x : |\psi_i(x) - m_i| \leq w_i \}$
סדרה בת-מנייה של קרשים כך שלכל
$x \in K$
קיים
$i$
עבורו
$|\psi_i(x) - m_i| \leq w_i$
.

נסמן
$\phi_i(x) = \frac{\psi_i(x)}{||\psi_i||_{K_{OP}}}$
פונקציונל יחידה ב-
$(\mathbb{R}^d, ||\cdot||_{K_{OP}})$.

מכך ש-
$$\{ x : |\phi_i(x) - \frac{m_i}{||\psi_i||_{K_{OP}}}| \leq \frac{w_i}{||\psi_i||_{K_{OP}}} \} = \{ x : |\psi_i(x) - m_i| \leq w_i \}$$

&nbsp;

נובע ממשפט
$Ball~1$
ש-
$\sum{\frac{w_i}{||\psi_i||_{K_{OP}}}} = \sum{\frac{w_i}{h_K(y_i)}} \geq 1$

\qed

## רדוקציה לאנליזה פונקציונלית Ball,

### משפט $Ball~2$

יהי
$(\phi_i)_{1}^{n}$
אוסף סופי של פונקציונלים לינארים ב-
$\mathbb{R}^d$
בעלי נורמה 1
,
מספרים ממשיים
$(m_i)_{1}^{n}$
ומספרים חיוביים
$(w_i)_{1}^{n}$
כך ש-
$\sum_{1}^{n}{w_i}=1$.

אז קיימת נקודה
$x \in B(1)_{||\cdot||} \subset \mathbb{R}^d$
כך שלכל
$1 \leq i \leq n$
מתקיים
$$|\phi_i(x) - m_i| \geq w_i$$

## רדוקציה לאנליזה פונקציונלית Ball,

משיקולי קומפקטיות, משפט
$Ball~1$
ומשפט
$Ball~2$
שקולים. בפרט,

### Ball 2 <= Ball 1

יהי
$(\phi_i)_{1}^{n}$
אוסף סופי של פונקציונלים לינארים ב-
$\mathbb{R}^d$
בעלי נורמה 1
,
מספרים ממשיים
$(m_i)_{1}^{n}$
ומספרים חיוביים
$(w_i)_{1}^{n}$
כך שלכל
$x \in B(1) \subseteq \mathbb{R}^d$
קיים
$i$
עבורו 
$|\phi(x) - m_i| \leq w_i$.

&nbsp;

נניח בשלילה
$\sum_{i=1}^{n}{w_i} = r < 1$
ולכל
$n \in \mathbb{N}$
נסמן
$\varepsilon_n = r - \sum_{I=1}^{n}{w_i}$.

הרי ש-
$\sum_{I=1}^{n}{\frac{w_i}{r - \varepsilon}} = 1$
ולכן ממשפט
$Ball~2$
לכל
$n$
קיים
$x_n \in B(1)$
כך שלכל
$1 \leq i \leq n$
$$|\phi(x_n) - m_i| \geq \frac{w_i}{r - \varepsilon_n} > \frac{w_i}{r}$$.

מהקומפקטיות של
$B(1)$
ומרציפות הגבול קיים
$x \in B(1)$
כך ש-

$|\phi(x) - m_i| \geq \frac{w_i}{r}$
לכל
$1 \leq i \leq n$
,
בסתירה.

\qed

## רדוקציה לאלגברה לינארית Ball,

### משפט $Ball~2'$

תהי
$A = (a_{ij})$
מטריצה מסדר 
$n \times n$
המקיימת
$a_{ii} = 1$,
סדרת ממשיים
$(m_i)_{1}^{n}$
וסדרת אי-שליליים
$(w_i)_{1}^{n}$
כך ש-
$\sum_{i=1}^{n} w_i \leq 1$.

&nbsp;

אז קיימת סדרת ממשיים
$(\lambda_i)_{1}^{n}$
כך ש-
$\sum_{j=1}^{n}{|\lambda_j|} \leq 1$

ולכל $i=1,2...n$
$$|\sum_{j=1}^{n}{a_{ij}\lambda_j - m_i}| \geq w_i$$

## רדוקציה לאלגברה לינארית Ball,

$Ball~2$
ומשפט
$Ball~2'$
שקולים. בפרט,

### Ball 2' <= Ball 2


יהי
$(\phi_i)_{1}^{n}$
אוסף סופי של פונקציונלים לינארים ב-
$\mathbb{R}^d$
בעלי נורמה 1
,
מספרים ממשיים
$(m_i)_{1}^{n}$
ומספרים חיוביים
$(w_i)_{1}^{n}$
כך ש-
$\sum_{1}^{n}{w_i}=1$.

&nbsp;

תהי
$(x_j)_1^n$
סדרת נקודות ב-
$\mathbb{R}^d$
כך ש-
$||x_j|| = 1$
ובהן
$\phi_j(x_j) = ||\phi_j||_{OP}$.

נגדיר את המטריצה
$A = (a_{ij}) = \phi_i(x_j)$
לכל
$1 \leq i, j \leq n$
.
ממשפט
$Ball~2'$
קיימת סדרת ממשיים
$(\lambda_j)_1^n$
כך ש-
$\sum_{j=1}^{n}|\lambda_j| \leq 1$.

&nbsp;

עבור
$x = \sum_{j=1}^{n}{\lambda_j x_j}$
מתקיים כי
$$||x|| = ||\sum_{j=1}^{n}{\lambda_j x_j}|| \leq \sum_{j=1}^{n}{|\lambda_j|||x_j||} = \sum_{j=1}^{n}{|\lambda_j|} \leq 1$$
כלומר,
$x \in B(1)$
.

---

מלינאריות
$$\phi_i(x) = \phi_i(\sum_{j=1}^{n}{\lambda_j x_j}) = \sum_{j=1}^{n}{\lambda_j \phi_i(x_j)} = \sum_{j=1}^{n}{a_{ij} \lambda_j}$$

ממשפט
$Ball~2'$
נובע שלכל
$i=1, 2, ..., n$
$$|\sum_{j=1}^{n}{a_{ij}\lambda_j} - m_i| \geq w_i$$

ולכן
$|\phi_i(x) - m_i| \geq w_i$.

\qed

::: notes

- there exists such points x_j according to operator norm properties. The norm is achieved on unit circle boundary

:::

# הוכחת משפט Ball

## הוכחת Ball

### למה 3 (Bang)


תהי
$H = (h_{ij})$
מטריצה ממשית וסימטרית מסדר
$n \times n$
כך ש-
$h_{ii} = 1$,
סדרת ממשיים
$(\mu_i)_{1}^{n}$
וסדרת מספרים אי-שליליים
$(\theta_i)_{1}^{n}$.

&nbsp;

אז קיימת סדרת סימנים
$(\varepsilon_j)_{1}^{n}$
כך שלכל
$i$,
$$
    \sum_{j=1}^{n}{|h_{ij}\varepsilon_j\theta_j - \mu_i| \geq \theta_i}
$$

## הוכחת Ball

### למה 4

תהי
$A \in \RR^{n \times n}$
כך ש-
$a_{i*} \neq 0$
לכל
$i$.

אז קיימת סדרת מספרים חיוביים
$(\theta_i)_{1}^{n}$
ומטריצה אורתוגונלית
$U$
כך שהמטריצה
$H = (h_{ij})$
הנתונה ע"י
\[
    h_{ij} = \theta_i(AU)_{ij}
\]
היא חיובית ובעלת אלכסון שכולו אחדות.

כלומר,
לכל
$x \in \RRN$
מתקיים ש-
$x^* H x \geq 0$
ו-
$h_{ii} = 1$
לכל
$i$.

<style> * { direction: rtl; } pre { direction: ltr; } span.math, span.math * { direction: ltr; }</style>
