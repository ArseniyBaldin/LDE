import numpy as np


def generate_gaussian(mean=(0, 0), sigma=(1, 1), angle_deg=0, n_points=1000):
    """
    Генерирует двумерное гауссовское облако с заданными параметрами.

    :param mean: tuple (mu_x, mu_y)
    :param sigma_x: стандартное отклонение по главной оси x
    :param sigma_y: стандартное отклонение по главной оси y
    :param angle_deg: угол поворота в градусах
    :param n_points: количество точек
    :return: массив shape (n_points, 2)
    """
    dim = np.array(mean).shape[-1]
    #2D нормальное распределение
    points = np.random.randn(n_points, dim) * sigma

    # Поворот
    if dim == 2:
        theta = np.deg2rad(angle_deg)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        points = points @ rotation_matrix.T

    # Сдвиг
    points += np.array(mean)

    return points


import numpy as np


def generate_ball(n_points=100, dim=2):
    """
    Генерация равномерных точек внутри N-мерного единичного шара.

    :param n_points: число точек
    :param dim: размерность пространства
    :return: массив shape (n_points, dim)
    """
    # 1. Сэмплируем направления на сфере (нормируем гауссовские векторы)
    directions = np.random.randn(n_points, dim)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # 2. Сэмплируем радиусы с плотностью ~ r^{dim - 1}
    radii = np.random.rand(n_points) ** (1 / dim)

    # 3. Растягиваем направления по радиусам
    points = directions * radii[:, np.newaxis]

    return points

def split_points(points, eps):
    norms = np.linalg.norm(points[:, :-1], axis=1)
    inner_points = points[norms <= eps]
    outer_points = points[norms > eps]
    return inner_points, outer_points

def estimate_local_scaling(f, data, eps=1e-3):
    """
    Численно оценивает sqrt(det(JᵀJ)) для каждой точки.
    """
    n, d = data.shape
    f_data = f(data)
    m = f_data.shape[1]

    J = np.zeros((n, m, d))

    for i in range(n):
        for j in range(d):
            data_plus = data.copy()
            data_minus = data.copy()
            data_plus[i, j] += eps
            data_minus[i, j] -= eps
            grad = (f(data_plus)[i] - f(data_minus)[i]) / (2 * eps)
            J[i, :, j] = grad

    det_sqrt = np.zeros(n)
    for i in range(n):
        JTJ = J[i].T @ J[i]
        det = np.linalg.det(JTJ)
        det_sqrt[i] = np.sqrt(max(det, 0))  # защита от отрицательных из-за численного шума

    return det_sqrt
