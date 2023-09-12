import numpy as np
from gsvd import gsvd
from abc import ABC, abstractmethod

class Regularizer(ABC):
    
    def set_lambdas(self, e_min: float, e_max: float, num: int):
        inf, sup = [np.log10(a*b) for a, b in zip((e_min, e_max), self.lims)]
        self.lambdas = np.logspace(sup, inf, num, endpoint=True)
        self.f = self.compute_filter_factors()
    
    def compute_filter_factors(self):
        p = self.lambdas.size
        _, r = self.V.shape
        f = np.empty((p, r), dtype=np.float32) # filter factors
        broad_l = self.lambdas[:, np.newaxis] # p x 1
        f[:] = self.filter_factors(broad_l)
        return f

    def solve(self, B: np.ndarray):
        """
        A [m, n], n > m, r = rank(A)
        B [m, N], N = # samples
        
        f [p, r], p = # lambdas.
        U [m, r]
        V [n, r]
        """
        if B.ndim == 1:
            B = B[:,np.newaxis]

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
    
    @abstractmethod
    def factor_lims(self, s: np.ndarray) -> tuple[float, float]:
        raise NotImplemented
    
    @abstractmethod
    def filter_factors(self, lambdas: np.ndarray) -> np.ndarray:
        raise NotImplemented

class Ridge(Regularizer):

    def __init__(self, A: np.ndarray) -> None:
        """ Zero Order Tikhonov
        s = r x 1
        U = m x r
        """
        self.U, self.s, Vt = np.linalg.svd(A, full_matrices=False, compute_uv=True)
        self.V = Vt.T # n x r
        self.lims = self.factor_lims(self.s)
    
    def filter_factors(self, lambdas: np.ndarray):
        return self.s/(self.s**2 + lambdas**2)
    
    def factor_lims(self, s: np.ndarray):
        return np.min(s), np.max(s)

class Tikhonov(Regularizer):

    def __init__(self, A: np.ndarray, B: np.ndarray) -> None:
        """ High Order Tikhonov"""
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
    

