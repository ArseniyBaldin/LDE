import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from utilities.data_utils import generate_gaussian, generate_ball, estimate_local_scaling
from utilities.visual_utils import set_axes_equal, make_sphere, animate_transition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class LDE:

    def __init__(self, eps, dim, center=None, func=lambda x: 1):

        self.func = func
        self.eps = eps
        if self.eps != 1:
            self.gamma = ((1 + self.eps) / (1 - self.eps)) ** 0.5
        else:
            self.gamma = None
        self.center = center
        self.dim = dim

    def I_in_der(self, g):
        if abs(g) >= 1:
            raise ValueError("g выходит за допустимый интервал (-1, 1)")
        return 1 / np.sqrt(1 - g ** 2)

    def I_in(self, x):
        integrand = lambda g: self.I_in_der(g) * g ** (self.dim - 1)
        result, _ = quad(integrand, 0, x, limit=500)
        return result

    def I_in_inv(self, y, tol=1e-8):

        f = lambda x: self.I_in(x) - y
        result = root_scalar(f, bracket=[0, self.eps], method='bisect', xtol=tol)
        if not result.converged:
            raise RuntimeError("I_in_inv: решение не найдено")
        return result.root

    def func_normalized(self, x):
        integrand = lambda t: self.func(t) * t ** (self.dim - 1)
        integral, _ = quad(integrand, 0, self.eps, limit=500)
        if integral == 0:
            raise ValueError("Нормирующий интеграл равен нулю")
        return (self.func(x) / integral) * self.I_in(self.eps)

    def g(self, X):

        def g_point(x):
            if 0 <= x <= 1:
                integrand = lambda t: self.func_normalized(t) * t ** (self.dim - 1)
                integral, _ = quad(integrand, 0, x, limit=500)
                return self.I_in_inv(integral)

            else:
                print(x)
                raise ValueError("X лежит вне отрезка [0, 1]")

        X = np.array(X).reshape(-1)
        result = np.zeros_like(X, dtype=float)

        for i, x in enumerate(X):
            result[i] = g_point(x)

        return result

    import numpy as np

    def forward(self, _data):
        data = np.copy(_data) - self.center
        data /= np.max(np.linalg.norm(data, axis=1)) / EPS

        norms_old = np.linalg.norm(data, axis=1).reshape(-1, 1)
        norms_new = self.g(norms_old).reshape(-1, 1)

        data = data * (norms_new / norms_old)

        # Inner points projection
        z_inner = np.sqrt(1 - np.sum(data ** 2, axis=1, keepdims=True))
        data = np.hstack([data, z_inner])

        return data

    # Численное вычисление коэффициентов растяжения через якобиан

    def estimate_stretching(self, data, eps=1e-3):
        """
         Вычисляет матрицу Грамма (J^T * J) и корень из её детерминанта
         для отображения f(data).

         Parameters:
             f (function): Функция, отображение f(data).
             data (numpy.ndarray): Входной массив данных размерности (n, d),
                                   где n — количество точек, d — размерность данных.

         Returns:
             tuple: (матрица Грамма, корень из детерминанта матрицы Грамма)
         """
        f = self.forward
        # Размерность входных данных
        n, d = data.shape

        # Вычисление значения функции f(data)
        output = f(data)
        m = output.shape[1]  # Размерность выходного пространства

        # Вычисление Якобиана J для каждого элемента в data
        J = np.zeros((n, m, d))  # Трёхмерный массив для Якобианов

        for i in range(n):
            for j in range(d):
                # Численная аппроксимация частных производных (финитные разности)
                epsilon = 1e-5
                data_perturbed = data.copy()
                data_perturbed[i, j] += epsilon
                J[i, :, j] = (f(data_perturbed)[i] - f(data)[i]) / epsilon

        # Вычисление J^T * J для каждого элемента
        gram_matrices = np.zeros((n, d, d))
        gram_determinants_sqrt = np.zeros(n)

        for i in range(n):
            JTJ = J[i].T @ J[i]  # Матрица Грамма
            gram_matrices[i] = JTJ
            gram_determinants_sqrt[i] = np.sqrt(np.linalg.det(JTJ))

        return gram_determinants_sqrt


EPS = 1

gaus_func = lambda x: 2.7 ** (- 30 * (x-0.5) ** 2)
const_func = lambda x: 1

encoder = LDE(EPS, 2, func=const_func, center=[0, 0])


