from src.regularization_tools import Ridge
import numpy as np 
from geometric_plotter import Plotter

rng = np.random.default_rng(seed=3)

n = 50
p = 100

x = rng.random(size=(n, 1))
N0 = np.zeros((n, 1))
N1 = rng.random(size=(n, 1))

for N in [N0, N1]:
    p0 = Plotter(_2d=True, figsize=(5,5))
    for k, style in zip([1, 2, 3], ['-k', '--k', '-.k']):
        A = Ridge.random_matrix_by_cond_number(n, k=k, seed=0)
        # print(f'cond = {np.linalg.cond(A) : .0g}')
        B = A@x 
        noisy_B = B + 1e-3*N
        model = Ridge(A)
        model.set_lambdas(1e-4, 1e3, p)
        xs = model.solve(noisy_B)

        err = np.linalg.norm(xs-x[np.newaxis, :, :], axis=(1,2))
        err /= np.linalg.norm(x)

        p0.axs.loglog(model.lambdas, err, style)
        opt_ix = err.argmin()
        p0.axs.axvline(model.lambdas[opt_ix], color='gray', linewidth=.25)

    p0.axs.set_xlabel('(log-log)')
    p0.axs.set_xlabel('$\lambda$')
    p0.axs.set_ylabel('$\\frac{\|x^* - x \|}{\|x\|}$')
p0.show()