from src.regularization_tools import Ridge
import numpy as np 
import sklearn.linear_model as skl

for _ in range(100):
    
    m, n = 50, 100
    p = 10
    N = 15

    A = np.random.rand(m, n)
    B = np.random.rand(m, N)

    model = Ridge(A)
    lambdas = model.lambdaspace(1e-4, 1e4, p)
    X = model.solve(B, lambdas)

    skX = np.empty_like(X)
    for i, l in enumerate(lambdas):
        skmodel = skl.Ridge(l**2, fit_intercept=False, solver='svd')
        skX[i, ...] = skmodel.fit(A, B).coef_.T

    assert np.allclose(X, skX)

print('OK')
