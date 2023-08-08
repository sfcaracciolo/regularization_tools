from typing import Callable
import numpy as np
from gsvd import gsvd

class Regularizer:
    def __init__(self, filter_factors: Callable, U: np.ndarray, V: np.ndarray, lambda_range: tuple = None) -> None:
        self.filter_factors = filter_factors
        self.U = U
        self.V = V
        self.lambda_range = lambda_range

    def solve(self, B: np.ndarray):
        """
        A [m, n], n > m, r = rank(A)
        B [m, N], N = # samples
        
        f [p, r], p = # lambdas.
        U [m, r]
        V [n, r]
        """
        p, _ = self.f.shape
        _, N = B.shape
        n, _ = self.V.shape
        m, _ = self.U.shape
        X = np.zeros((p, n, N), dtype=np.float64)
        T = np.zeros((p, N), dtype=np.float64)
        R = self.U.T @ B # r x N
        for i in range(m):
            vi = self.V[:, i] # [n]
            fi = self.f[:, i] # [p]
            ri = R[i, :] # [N]
            np.outer(fi, ri, out=T)
            X += T[:, np.newaxis, :] * vi[np.newaxis, :, np.newaxis] # [p, 1, N] * [1, n, 1] = [p, n, N]
        return X

    @classmethod
    def ridge(cls, A: np.ndarray):
        """ Zero Order Tikhonov
        s = r x 1
        U = m x r
        """
        U, s, Vt = np.linalg.svd(A, full_matrices=False, compute_uv=True)
        V = Vt.T # n x r
        # broad_s = s[np.newaxis, :] # 1 x r
        filter_factors = Regularizer.rigde_factors(s)
        return cls(filter_factors, U, V, lambda_range=Regularizer.rigde_range(s))
    
    @staticmethod
    def rigde_factors(s: np.ndarray):
        return lambda x: s/(s**2 + x**2)
    
    @staticmethod
    def rigde_range(s: np.ndarray):
        s[s==0.] = np.nan
        return np.nanmin(s), np.nanmax(s)

    @classmethod
    def tikhonov(cls, A: np.ndarray, L: np.ndarray):
        """ High Order Tikhonov"""
        m, n = A.shape
        (U, _), (D_A, D_L), X = gsvd(A, L)
        s_a = np.zeros(n, dtype=np.float32)
        s_a[:m] = D_A.diagonal()
        s_l = D_L.diagonal()
        filter_factors = Regularizer.tikhonov_factors(s_a, s_l)
        return cls(filter_factors, U, X, lambda_range=Regularizer.tikhonov_range(s_a, s_l))

    @staticmethod
    def tikhonov_factors(sa, sb):
        return lambda x: sa/(sa**2 + (x*sb)**2)

    @staticmethod
    def tikhonov_range(sa: np.ndarray, sb: np.ndarray):
        return Regularizer.rigde_range(sa/sb)

    def compute_filter_factors(self, lambdas: np.ndarray):
        p = lambdas.size
        _, r = self.V.shape
        self.f = np.empty((p, r), dtype=np.float32) # filter factors
        broad_l = lambdas[:, np.newaxis] # p x 1
        self.f[:] = self.filter_factors(broad_l)

    def lambda_logspace(self, l_min, l_max, num: int):
        return np.logspace(np.log10(l_max), np.log10(l_min), num, endpoint=True)
