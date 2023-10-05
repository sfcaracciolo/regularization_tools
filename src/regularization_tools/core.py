import numpy as np
from gsvd import gsvd
from abc import ABC, abstractmethod
import scipy as sp

class AbstractRegularizer(ABC):
    
    @staticmethod
    def ill_cond_matrix(N: int, k: float, seed: int = None):
        """ Generator of a correlation random matrix (N, N) with condition number equal to 10^k"""
        f = lambda x, a, b: a/x**b
        n = np.arange(1, N+1)
        b = k / np.log10(N)
        r = np.sum(f(n, 1, b)) # if b > 1 sp.special.zeta(b, 1) - sp.special.zeta(b, N+1) works
        a = N/r
        X = sp.stats.random_correlation(f(n, a, b), seed)
        return X.rvs()

    @staticmethod
    def lambdaspace(start: float, end: float, num: int = 100):
        return np.logspace(np.log10(start), np.log10(end), num, endpoint=True)

    @abstractmethod
    def penalization(self, *args) -> np.ndarray:
        raise NotImplemented
    
    @abstractmethod
    def residual(self, *args) -> np.ndarray:
        raise NotImplemented
    
    @abstractmethod
    def solve(self, *args) -> np.ndarray:
        raise NotImplemented
    
class Ridge(AbstractRegularizer):

    def __init__(self, A: np.ndarray) -> None:
        """ Zero Order Tikhonov
        A : m x n
        r = rank(A)
        S : r x r
        U : m x r
        V :  n x r
        """
        self.A = A
        self.U, s, Vt = np.linalg.svd(A, full_matrices=False, compute_uv=True)
        self.V = Vt.T # n x r
        self.S = sp.sparse.dia_matrix((s, 0), shape=(s.size, s.size), dtype=np.float64)
    
    def solve(self, Y: np.ndarray, lambdas: np.ndarray):
        """
        Y : m x N 
        N = # samples
        """
        if Y.ndim == 1:
            Y = Y[:,np.newaxis]

        p = lambdas.size
        _, N = Y.shape
        n, r = self.V.shape
        m, _ = self.U.shape

        s2 = self.S.diagonal()**2
        Z = self.S.T @ self.U.T @ Y
        X = np.empty((p, n, N), dtype=np.float64)
        for i in range(p):
            ff = 1./(s2 + lambdas[i]**2)
            invDl = sp.sparse.dia_matrix((ff, 0), shape=(r, r), dtype=np.float64)
            X[i] = self.V @ invDl @ Z
        return X
    
    def penalization(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(X, ord='fro', axis=(1, 2))
    
    def residual(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        E = np.tensordot(self.A, X, axes=(1,1)) - Y[:, np.newaxis, :]
        return np.linalg.norm(E, ord='fro', axis=(0, 2))

class Tikhonov(AbstractRegularizer):

    def __init__(self, A: np.ndarray, B: np.ndarray) -> None:
        """ High Order Tikhonov
        A : p x n
        B : m1 x n
        r = rank([A, B]') must be full (r=n)
        U_1 : p x p
        X :  n x n
        """
        self.A, self.B = A, B
        (self.U_1, _), (self.D_A, self.D_B), self.X = gsvd(A, B, tol=1e-16)

    def solve(self, Y: np.ndarray, lambdas: np.ndarray):
        """
        Y : p x N 
        N = # samples
        """
        if Y.ndim == 1:
            Y = Y[:,np.newaxis]

        _p = lambdas.size
        _, N = Y.shape
        n, _= self.X.shape
        
        a2 = (self.D_A.T @ self.D_A).diagonal()
        a = np.sqrt(a2)
        b2 = (self.D_B.T @ self.D_B).diagonal()
        Z = self.D_A.T @ self.U_1.T @ Y
        X = np.empty((_p, n, N), dtype=np.float64)
        for i in range(_p):
            ff = a/(a2 + b2 * lambdas[i]**2)
            invDl = sp.sparse.dia_matrix((ff, 0), shape=(n, n), dtype=np.float64)
            X[i] = self.X @ invDl @ Z
        return X

    def penalization(self, X: np.ndarray) -> np.ndarray:
        E = np.tensordot(self.B, X, axes=(1, 1))
        return np.linalg.norm(E, ord='fro', axis=(0, 2))
    
    def residual(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        E = np.tensordot(self.A, X, axes=(1,1)) - Y[:, np.newaxis, :]
        return np.linalg.norm(E, ord='fro', axis=(0, 2))
    