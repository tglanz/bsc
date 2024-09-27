# Mmn14, 22913

Author: Tal Glanzman

Date: 2024/09/27

# Answer to 1

# Answer to 2

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

Finally, the code will be
$$
97, 178, 179, 180, 97
$$

and the dictionary

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