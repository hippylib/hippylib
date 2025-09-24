# Adjoint method for time dependent PDEs using one step methods 

Here we consider the adjoint equations for gradient actions and Hessian actions when using a one-step method discretization of some PDE. Abstractly, the forward problem can be written as 

$$
\text{Given } u_0, \text{ find } u = \{u_n\}_{n=1}^{N} \text{ s.t. } \quad
\sum_{n=1}^{N} r(u_n, u_{n-1}, m, p_n, t_n) = 0 \quad \forall p_n
$$

## Adjoint equations for Gradient 
We define the Lagrangian

$$
L(u,m,p) = \sum_{n=1}^{N} q_n (u_n, m) + \sum_{n=1}^{N} r(u_n, u_{n-1}, m, p_n, t_n)
$$

The adjoint equation can be derived by differentiating the Lagrangian with respect to the state $u$. Considering each $u_k$ separately, $\partial_{u_k} L = 0$ becomes

$$
\partial_u q(u_k, m) \tilde{u} + \partial_1 r(u_k, u_{k-1}, m, p_k, t_k) \tilde{u}
+ \partial_2 r(u_{k+1}, u_k, m, p_{k+1}, t_{k+1}) \tilde{u} 
= 0,
$$

where the term involving $u_{k+1}, p_{k+1}$ does not appear when $k = N$ (i.e., for the final time step of the forward problem). In what's to follow, we won't spell this out directly and instead abuse notation with this fact in mind.

This corresponds to the system

$$
\partial_1 r(u_k, u_{k-1}, m, p_k) \tilde{u}
=
\underbrace{- \partial_2 r(u_{k+1}, u_k, m, p_{k+1}) \tilde{u}}_{\text{ handled within solveAdj}}
\; \underbrace{- \partial_u q_k (u_k, m) \tilde{u}}_{\text{ given by adj\_rhs}}
$$

### Simplification for implicit Euler methods 
When using a $\theta$ method for a PDE of the form $\partial_t u = f(u,m)$, we have  

$$
r(u_n, u_{n-1}, m, v_n)
= 
\frac{1}{\tau}
\langle 
u_n - u_{n-1}, p_n
\rangle
+
\theta \langle f(u_n, m), p_n \rangle 
+
(1 -\theta) \langle f(u_{n-1}, m), p_n \rangle ,
$$

Then,

$$
\partial_1 r(u_k, u_{k-1}, m, p_k, t_k) \tilde{u}
= 
\frac{1}{\tau} \langle p_k, \tilde{u} \rangle
+
\theta \langle f'(u_k, m)\tilde{u}, p_k \rangle 
$$ 

and

$$
\partial_2 r(u_{k+1}, u_{k}, m, p_{k+1}, t_{k+1}) \tilde{u}
= 
- \frac{1}{\tau} \langle p_{k+1}, \tilde{u} \rangle
+
(1-\theta) \langle f'(u_k, m)\tilde{u}, p_{k+1} \rangle.
$$ 

For implicit Euler ($\theta  = 1$), the second part of $\partial_2 r$ disappears. In this case, one can cheat a little in assembling the RHS by plugging in anything for $u_{k+1}, u_{k}$ in the form $\partial_2 r(u_{k+1}, u_{k}, m, p_{k}, t_k)$ since this form $r(u_{k+1}, u_{k}, m, p_{k+1}, t_{k+1})$ is always linear in the second argument $u_{k}$. This is currently implemented in `hippylib`.

For completeness, in this case, the adjoint equation is

$$
\partial_u q_k (u_k, m) \tilde{u} 
+ \frac{1}{\tau} \langle p_k - p_{k+1}, \tilde{u} \rangle
+ \theta \langle f'(u_k, m)\tilde{u}, p_k \rangle 
+ (1-\theta) \langle f'(u_k, m)\tilde{u}, p_{k+1} \rangle,
$$

or equivalently 

$$ 
\frac{1}{\tau} \langle p_k,  \tilde{u} \rangle
+ \theta \langle f'(u_k, m)\tilde{u}, p_k \rangle 
=
\frac{1}{\tau} \langle p_{k+1},  \tilde{u} \rangle
- (1-\theta) \langle f'(u_k, m)\tilde{u}, p_{k+1} \rangle 
-\partial_u q_k (u_k, m) \tilde{u}.
$$



## Hessian action

### Incremental equations 
We consider the Hessian Lagrangian, defined as

$$ 
\begin{align}
L^H &= 
\sum_{n=1}^{N} r(u_n, u_{n-1}, m, \hat{p}_n, t_n) \\
& + 
\sum_{n=1}^{N} \partial_u q_n (u_n, m) \hat{u}_n + \partial_1 r(u_n, u_{n-1}, m, p_n, t_n) \hat{u}_n
+ \partial_2 r(u_{n+1}, u_n, m, p_{n+1}, t_{n+1}) \hat{u}_n  \\
& + \sum_{n=1}^{N} \partial_m r(u_n, u_{n-1}, m, p_n, t_n) \hat{m}
\end{align}
$$

The incremental equations can be obtained through derivatives of this Lagrangian. The incremental forward problem is straightforwardly derived as $\partial_p L^{H} = 0$.

### Incremental adjoint
The incremental adjoint equation, given by 

$$ 
\partial_u L^H = 0 
$$ 

is slightly more involved. Computing this derivative with respect to each snapshot $u_{k}$, we have 

$$ 
\begin{align}
\partial_u L^H \tilde{u}_k = & 
\partial_1 r(u_k, u_{k-1}, m, \hat{p}_k, t_k)\tilde{u}_k 
+ 
\partial_2 r(u_{k+1}, u_{k}, m, \hat{p}_{k+1}, t_{k+1})\tilde{u}_k 
\\
& + 
\partial_u^2 q_k (u_k, m) \hat{u}_k \tilde{u}_k \\
& + \partial_1 \partial_1 r(u_k, u_{k-1}, m, p_k, t_k) \hat{u}_k \tilde{u}_k \\
& + \partial_2 \partial_1 r(u_{k}, u_{k-1}, m, p_{k}, t_k) \hat{u}_{k-1} \tilde{u}_k \\
& + \partial_2 \partial_2 r(u_{k+1}, u_k, m, p_{k+1}, t_{k+1}) \hat{u}_k \tilde{u}_k \\
& + \partial_1 \partial_2 r(u_{k+1}, u_{k}, m, p_{k+1}, t_{k+1}) \hat{u}_{k+1} \tilde{u}_{k} \\
& 
+ \partial_1 \partial_m r(u_k, u_{k-1}, m, p_k, t_k) \hat{m} \tilde{u}_n\\
& + \partial_2 \partial_m r(u_{k+1}, u_{k}, m, p_{k+1}, t_{k+1}) \hat{m} \tilde{u}_n
\end{align}
$$

In `hippylib`, only the first line, i.e.,

$$
\partial_1 r(u_k, u_{k-1}, m, \hat{p}_k, t_k)\tilde{u}_k 
+ 
\partial_2 r(u_{k+1}, u_{k}, m, \hat{p}_{k+1}, t_{k+1})\tilde{u}_k 
$$

is handled within the `solveIncrementalAdj` call. 

The remaining parts of the adjoint RHS are given through the `rhs` argument, where

- The observation term $$ \partial_u^2 q_k(u_k, m) \hat{u}_k \tilde{u}_k $$ is assembled through calls to the `Misfit` class. 

- the second derivatives with respect to $u$, $$ \begin{align} \partial_1 \partial_1 r(u_k, u_{k-1}, m, p_k, t_k) \hat{u}_k \tilde{u}_k + \partial_2 \partial_1 r(u_{k}, u_{k-1}, m, p_{k}, t_k) \hat{u}_{k-1} \tilde{u}_k \\ \partial_2 \partial_2 r(u_{k+1}, u_k, m, p_{k+1}, t_{k+1}) \hat{u}_k \tilde{u}_k + \partial_1 \partial_2 r(u_{k+1}, u_{k}, m, p_{k+1}, t_{k+1}) \hat{u}_{k+1} \tilde{u}_{k} \end{align} $$ are assembled through calls to `applyWuu`.

- The mixed second derivatives, $$ \partial_1 \partial_m r(u_k, u_{k-1}, m, p_k, t_k) \hat{m} \tilde{u}_k + \partial_2 \partial_m r(u_{k+1}, u_{k}, m, p_{k+1}, t_{k+1}) \hat{m} \tilde{u}_n $$ are assembled through calls to `applyWum`.

Thus, the implementation for the incremental adjoint is essentially identical to the adjoint equation within the PDE problem class, and rely on the implementations of the Hessian blocks of $r$ to supply the RHS.

### Hessian blocks
We now consider the Hessian blocks involved in assembling the RHS.

Recall that the PDE residual is  

$$ 
\sum_{n=1}^{N} r(u_n, u_{n-1}, m, p_n, t_n) = 0
$$ 

For `applyWum`, we have

$$
[W_{um}\hat{m}]_k
= 
\partial_1 \partial_m r(u_k, u_{k-1}, m, p_k, t_k) \hat{m} \tilde{u}_k
+ 
\partial_2 \partial_m r(u_{k+1}, u_{k}, m, p_{k+1}, t_{k+1}) \hat{m} \tilde{u}_k.
$$

We also have its transpose, `applyWmu`, which is 

$$
[W_{mu}\hat{u}_k] \tilde{m}
= 
\partial_1 \partial_m r(u_k, u_{k-1}, m, p_k, t_k) \tilde{m} \hat{u}_k
+ 
\partial_2 \partial_m r(u_{k+1}, u_{k}, m, p_{k+1}, t_{k+1}) \tilde{m} \hat{u}_k.
$$

The `applyWuu` block is a little complicated, since we need to consider both the input and ouptut time indices. In particular, it is block tri-diagonal, where 

$$
[W_{uu}\hat{u}_{k}]_{k}
= 
\partial_1 \partial_1 r(u_k, u_{k-1}, m, p_k, t_k) \hat{u}_k \tilde{u}_{k}
+ 
\partial_2 \partial_2 r(u_{k+1}, u_{k}, m, p_{k+1}, t_{k+1}) \hat{u}_k \tilde{u}_{k},
$$

$$
[W_{uu}\hat{u}_{k}]_{k-1}
= 
\partial_2 \partial_1 r(u_k, u_{k-1}, m, p_k, t_k) \hat{u}_k \tilde{u}_{k-1}
$$

and 

$$
[W_{uu}\hat{u}_{k}]_{k+1}
= 
\partial_1 \partial_2 r(u_{k+1}, u_{k}, m, p_{k+1}, t_{k+1}) \hat{u}_k \tilde{u}_{k+1},
$$


#### Simplifications for implicit Euler

As with the adjoint equation, simplifications can be made when using implicit Euler. For any form with $\partial_2 r(u_{k+1}, u_{k}, m, p_{k+1}, t_{k+1})$, one can again plug in any $u_{k+1}, u_{k}$ due to linearity in the second argument (and $u_k$ never has mixed nonlinearities with $u_{k+1}$ or $m$). This often saves the need to retrieve $u_{k+1}$ in addition to $u_{k}, u_{k-1}$ within a time step. Again, the current `hippylib` implementation makes these simplifications.