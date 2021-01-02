# ממן 12

מגיש: טל גלנצמן  
ת.ז: 302800354

## שאלה 1

### סעיף א

```python
findMin(heap)
	if heap.size = 0
	    return -1
	   
	if heap.size = 1
        return heap[0]

    min = heap[1]
    for i in 2 to heap.size - 1
        if heap[i] < min
            min = heap[i]
        
	return min
```

### סעיף ב

### סעיף ג

### סעיף ד

## שאלה 2

### סעיף א

נרשום טבלת יאנג עבור אוסף המספרים $$ \{ 9, 16, 3, 2, 4, 8, 5, 14, 12 \} $$

$$
Y =
\begin{pmatrix}
2 & 3 & 4 & 5 \\
8 & 9 & 12 & 14 \\
16 & \infty & \infty & \infty \\
\infty & \infty & \infty & \infty
\end{pmatrix}
$$

כאשר אל \infty אנו מתייחסים כאל איבר שאין גדול ממנו

### סעיף ב

__עבור Y[1,1]=\infty__ נניח בשלילה כי קיימים i ו- j כך ש \i \cdot j \neq 1