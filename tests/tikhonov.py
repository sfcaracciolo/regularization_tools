from src.regularization_tools import Tikhonov
import numpy as np 
import sklearn.linear_model as skl

for _ in range(100):

    m, n = 50, 100
    p = 10
    N = 15

    A, L = np.random.rand(m, n), np.eye(n)
    B = np.random.rand(m, N)

    model = Tikhonov(A, L)
    model.set_lambdas(1e-2, 1e2, p)
    X = model.solve(B)
    skX = np.empty_like(X)
    for i, l in enumerate(model.lambdas):
        skmodel = skl.Ridge(l**2, fit_intercept=False, solver='svd')
        skX[i, ...] = skmodel.fit(A, B).coef_.T

    assert np.allclose(X, skX, rtol=1e-4)

print('OK')
