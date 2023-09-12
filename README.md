# Regularization Tools

Implementation of Ridge and Tikhonov regularizations based on SVD (Singular Value Decomposition) and GSVD (Generalized Singular Value Decomposition). Notably, the solver admits 2d measurements and works for a range of $\lambda$ hyperparameters at once.

Let $A \in \mathbb{R}^{m\times n}$, $Y \in \mathbb{R}^{m\times N}$, $L \in \mathbb{R}^{n\times n}$ solve $X \in \mathbb{R}^{n\times N}$ such as:
$$\| Y-AX\|_2^2 + \lambda^2 \|L X\|_2^2$$
be minimum.

If $L = I_n$ use ``Regularizer.rigde`` method class, otherwise ``Regularizer.tikhonov``.

## Usage

```python
from regularization_tools import Ridge, Tikhonov
import numpy as np 

# matrix definition
m, n = 50, 100
A = np.random.rand(m, n)

# measurements
N = 15
Y = np.random.rand(m, N)

# Ridge model
model = Ridge(A)

# or Tikhonov model (require L matrix additionally)
# model = Tikhonov(A, L)

# hyperparameter definition 
p = 10
model.set_lambdas(1e-2, 1e2, p) # 1e-2 and 1e2 times the lims of filter factores.

# solver
X = model.solve(Y)

# X is a tensor with shape (p, n, N)
```