# Mmn13, 22913

Author: Tal Glanzman

Date: 

# Answer to 1

TODO: PUT THE CODE HERE

# Answer to 2

We will answer using two approaches.

**Apprach 1**

The rows of $A$ are the eigenvectors of $C_x$ which is an orthogonal set, and therefore $A$ is orthogonal which implies that
$$
    A^T=A^{-1}
$$

We can now write
$$
C_y = A C_x A^{-1}
$$

and notice that $C_y$ and $C_x$ are in fact Similar Matrices and it is well known that similar matrices have the same eigenvalues.

**Approach 2**

By construction and according to (11-52):

$$
C_y = diag(\lambda_i)
$$

where $\lambda_i$ are the eigen values of $C_x$.

We now that eigenvalues of diagonal matrices are the diagonal itself so we conclude that the eigenvalues of $C_y$ and $C_x$ are the same (and are $\lambda_i$)

# Answer to 3

From
$$
A \ominus B = \{ z | B_z \subseteq A \}
$$

we get that if $z \in A \ominus B$ then forall $b \in B$
$$
    z + b \in A
$$

Observe that according to the definition of translation we get that
$$
    A_{-b} = \{ a - b | a \in A \} = \{ x | x + b \in A \}
$$

so we conclude that $z \in A_{-b}$ for all $b \in B$ and therefore
$$
    A \ominus B \subseteq \bigcap_{b \in B} A_{-b}
$$

On the other hand, if $z \in \bigcap_{b \in B} A_{-b}$ then for all $b \in B$
$$
    z \in A_{-b}
$$

and with similar arguments we get that forall $z + b \in A$ and therefore by definition
$$
    B_z \subseteq A
$$

which means that $z \in A \ominus B$ so
$$
    \bigcap_{b \in B} A_{-b} \subseteq A \ominus B
$$

Because we have shown inclusion from both sides we conclude equality.