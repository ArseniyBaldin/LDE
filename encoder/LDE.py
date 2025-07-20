import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from utilities.enc_utils import generate_gaussian


class LDE:

    def __init__(self, eps, center, dim, func='const'):
        self.func = func
        self.eps = eps
        self.gamma = ((1 + self.eps) / (1 - self.eps)) ** 0.5
        self.center = center
        self.dim = dim

    def I_in_der(self, g):
        if abs(g) >= 1:
            raise ValueError("ABS(g) >= 1: значение g выходит за допустимый интервал (-1, 1)")
        result = 1 / (1 - g ** 2) ** 0.5
        return result

    def I_out_der(self, g):
        result = self.gamma / (g ** self.dim) * (1 - (self.gamma * (1 - g)) ** 2) ** (self.dim / 2 - 1)
        return result

    def I_in(self, x):
        integrand = lambda g: self.I_in_der(g) * g ** (self.dim - 1)
        result, error = quad(integrand, 0, x)
        return result

    def I_out(self, x):
        integrand = lambda g: self.I_out_der(g) * g ** (self.dim - 1)
        result, error = quad(integrand, self.eps, x)
        return result

    def I_in_inv(self, y, tol=1e-6):
        f = lambda x: self.I_in(x) - y
        result = root_scalar(f, bracket=[1e-8, self.eps - 1e-8], method='bisect', xtol=tol)
        if not result.converged:
            raise RuntimeError("I_in_inv: решение не найдено")
        return result.root

    def I_out_inv(self, y, tol=1e-6):
        f = lambda x: self.I_out(x) - y
        result = root_scalar(f, bracket=[self.eps + 1e-8, 1 - 1e-8], method='bisect', xtol=tol)
        if not result.converged:
            raise RuntimeError("I_out_inv: решение не найдено")
        return result.root

    def forward(self, data):
        data = data - self.center
        data /= np.max(np.linalg.norm(data, axis=1))
