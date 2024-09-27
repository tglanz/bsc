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