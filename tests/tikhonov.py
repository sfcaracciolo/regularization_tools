from src.regularization_tools import *
import numpy as np 
from sklearn.linear_model import Ridge

for _ in range(100):

    m, n = 50, 100
    p = 10
    N = 15

    A = np.random.rand(m, n)
    L = np.eye(n)
    B = np.random.rand(m, N)

    model = Regularizer.tikhonov(A, L)
    lambdas = model.lambda_logspace(*model.lambda_range, p)
    model.compute_filter_factors(lambdas)
    X = model.solve(B)

    skX = np.empty_like(X)
    for i, l in enumerate(lambdas):
        skmodel = Ridge(l**2, fit_intercept=False, solver='svd')
        skX[i, ...] = skmodel.fit(A, B).coef_.T

    assert np.allclose(X, skX)

print('OK')
