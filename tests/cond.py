from src.regularization_tools import Ridge
import numpy as np 

for _ in range(100):
    
    n = np.random.randint(10, 100)
    k = np.random.randint(1, 12)
    M = Ridge.random_matrix_by_cond_number(n, k)
    assert np.allclose(np.linalg.cond(M), 10**k)

print('OK')
