from matplotlib import pyplot as plt
from src.regularization_tools import Tikhonov
import numpy as np 
import sklearn.linear_model as skl

for _ in range(100):

    p, n = 50, 100
    _p = 5
    N = 15

    A, L = np.random.rand(p, n), np.eye(n)
    Y = np.random.rand(p, N)

    model = Tikhonov(A, L)
    lambdas = model.lambdaspace(1e-3, 1e-1, _p)
    X = model.solve(Y, lambdas)

    skX = np.empty_like(X)
    for i, l in enumerate(lambdas):
        skmodel = skl.Ridge(l**2, fit_intercept=False, solver='svd')
        skX[i, ...] = skmodel.fit(A, Y).coef_.T
    assert np.allclose(skX, X, rtol=1e-2, atol=1e-6)

print('OK')
