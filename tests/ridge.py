from src.regularization_tools import *
import numpy as np 
from sklearn.linear_model import Ridge

for _ in range(100):
    
    m, n = 50, 100
    p = 10
    N = 15

    A = np.random.rand(m, n)
    B = np.random.rand(m, N)

    model = Regularizer.ridge(A)
    lambdas = model.lambda_logspace(*model.lambda_range, p)
    model.compute_filter_factors(lambdas)
    X = model.solve(B)

    skX = np.empty_like(X)
    for i, l in enumerate(lambdas):
        skmodel = Ridge(l**2, fit_intercept=False, solver='svd')#, tol=1e-4)
        skX[i, ...] = skmodel.fit(A, B).coef_.T

    assert np.allclose(X, skX, rtol=1e-5, atol=1e-8)

print('OK')
