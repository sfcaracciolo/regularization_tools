import numpy as np
from gsvd import gsvd
from abc import ABC, abstractmethod

import scipy as sp

class AbstractRegularizer(ABC):

    def auto_lambdas(self, e_min: float, e_max: float, num: int):
        inf, sup = [a*b for a, b in zip((e_min, e_max), self.lims)]
        self.set_lambdas(inf, sup, num)

    def set_lambdas(self, l_min: float, l_max: float, num: int):
        self.lambdas = np.logspace(np.log10(l_min), np.log10(l_max), num, endpoint=True)
        self.compute_filter_factors()
    
    def compute_filter_factors(self):
        p = self.lambdas.size
        _, r = self.V.shape
        self.f = np.empty((p, r), dtype=np.float32) # filter factors
        self.f[:] = self.filter_factors(self.lambdas[:, np.newaxis]) # p x 1
        return self.f

    def solve(self, Y: np.ndarray):
        """
        A [m, n], n > m, r = rank(A)
        Y [m, N], N = # samples
        
        f [p, r], p = # lambdas.
        U [m, r]
        V [n, r]
        """
        if Y.ndim == 1:
            Y = Y[:,np.newaxis]

        self.Y = Y
        p, _ = self.f.shape
        _, N = Y.shape
        n, _ = self.V.shape
        m, _ = self.U.shape
        self.X = np.zeros((p, n, N), dtype=np.float64)
        T = np.zeros((p, N), dtype=np.float64)
        R = self.U.T @ Y # r x N
        for i in range(m):
            vi = self.V[:, i] # [n]
            fi = self.f[:, i] # [p]
            ri = R[i, :] # [N]
            np.outer(fi, ri, out=T)
            self.X += T[:, np.newaxis, :] * vi[np.newaxis, :, np.newaxis] # [p, 1, N] * [1, n, 1] = [p, n, N]
        return self.X
    
    def metric(self, M: np.ndarray):
        return np.linalg.norm(M, ord='fro', axis=(1, 2))

    @staticmethod
    def random_matrix_by_cond_number(N: int, k: float, seed: int = None):
        """ Generator of a correlation random matrix (N, N) with condition number equal to 10^k"""
        f = lambda x, a, b: a/x**b
        n = np.arange(1, N+1)
        b = k / np.log10(N)
        r = np.sum(f(n, 1, b)) # if b > 1 sp.special.zeta(b, 1) - sp.special.zeta(b, N+1) works
        a = N/r
        X = sp.stats.random_correlation(f(n, a, b), seed)
        return X.rvs()

    @abstractmethod
    def compute_penalizations(self) -> np.ndarray:
        raise NotImplemented
    
    @abstractmethod
    def compute_residuals(self) -> np.ndarray:
        raise NotImplemented
    
    @abstractmethod
    def factor_lims(self, s: np.ndarray) -> tuple[float, float]:
        raise NotImplemented
    
    @abstractmethod
    def filter_factors(self, lambdas: np.ndarray) -> np.ndarray:
        raise NotImplemented

class Ridge(AbstractRegularizer):

    def __init__(self, A: np.ndarray) -> None:
        """ Zero Order Tikhonov
        s = r x 1
        U = m x r
        """
        self.A = A
        self.U, self.s, Vt = np.linalg.svd(A, full_matrices=False, compute_uv=True)
        self.V = Vt.T # n x r
        self.lims = self.factor_lims(self.s)
    
    def filter_factors(self, lambdas: np.ndarray):
        return self.s/(self.s**2 + lambdas**2)
    
    def factor_lims(self, s: np.ndarray):
        return np.min(s), np.max(s)

    def compute_penalizations(self) -> np.ndarray:
        self.P = self.metric(self.X)
        return self.P
    
    def compute_residuals(self) -> np.ndarray:
        E = np.tensordot(self.A, self.X, axes=(1,1)) - self.Y[:, np.newaxis, :]
        E = np.swapaxes(E, 1, 0)
        self.R = self.metric(E)
        return self.R

class Tikhonov(AbstractRegularizer):

    def __init__(self, A: np.ndarray, B: np.ndarray) -> None:
        """ High Order Tikhonov"""
        self.A, self.B = A, B
        m, n = A.shape
        (self.U, _), (D_A, D_B), X = gsvd(A, B)
        self.sa = np.zeros(n, dtype=np.float32)
        self.sa[:m] = D_A.diagonal()
        self.sb = D_B.diagonal()
        self.V = X
        self.lims = self.factor_lims(self.sa/self.sb)
    
    def filter_factors(self, lambdas: np.ndarray):
        return self.sa/(self.sa**2 + (lambdas*self.sb)**2)

    def factor_lims(self, s: np.ndarray):
        s[s==0.] = np.nan
        return np.nanmin(s), np.nanmax(s)

    def compute_penalizations(self) -> np.ndarray:
        M = np.tensordot(self.B, self.X, axes=(1, 1))
        M = np.swapaxes(M, 1, 0)
        self.P = self.metric(M)
        return self.P

    def compute_residuals(self) -> np.ndarray:
        E = np.tensordot(self.A, self.X, axes=(1,1)) - self.Y[:, np.newaxis, :]
        E = np.swapaxes(E, 1, 0)
        self.R = self.metric(E)
        return self.R
    