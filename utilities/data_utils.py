import numpy as np
from utilities.visual_utils import plot_clusters

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

def generate_ball(mean=(0, 0), radius=1.0, n_points=1000):
    """
    Генерация равномерных точек внутри N-мерного шара.

    :param center: tuple, координаты центра шара (определяет размерность)
    :param radius: float, радиус шара
    :param n_points: int, число точек
    :return: np.ndarray shape (n_points, dim)
    """
    mean = np.asarray(mean)
    dim = mean.shape[-1]

    # 1. Направления на сфере (нормализованные нормальные векторы)
    directions = np.random.randn(n_points, dim)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # 2. Радиусы с плотностью ~ r^{dim-1}
    radii = np.random.rand(n_points) ** (1 / dim)

    # 3. Умножаем на радиус и добавляем смещение
    points = directions * (radii[:, np.newaxis] * radius) + mean

    return points

def generate_clusters(configs):
    """
    Генерирует точки и метки для нескольких кластеров по списку конфигураций.

    :param configs: список словарей, каждый из которых содержит:
        - "type": "gaussian" или "ball"
        - остальные параметры зависят от типа (см. generate_gaussian / generate_ball)

    :return: tuple (X, y)
        - X: np.ndarray, shape (total_points, dim)
        - y: np.ndarray, shape (total_points,)
    """
    all_points = []
    all_labels = []

    for label, config in enumerate(configs):
        cluster_type = config.pop("type").lower()

        if cluster_type == "gaussian":
            points = generate_gaussian(**config)
        elif cluster_type == "ball":
            points = generate_ball(**config)
        else:
            raise ValueError(f"Unknown cluster type: {cluster_type}")

        all_points.append(points)
        all_labels.append(np.full(len(points), label))

    X = np.vstack(all_points)
    y = np.concatenate(all_labels)
    return X, y

import numpy as np
import torch

def pad_to_statevector_dim(X):
    """
    Дополняет каждый вектор нулями до длины 2^k, где k минимально такое, что 2^k > D.
    Возвращает массив формы [N, 2^k].

    :param X: np.ndarray или torch.Tensor, shape [N, D]
    :return: того же типа, shape [N, 2^k]
    """
    is_numpy = isinstance(X, np.ndarray)

    # Приводим к torch для удобства
    if is_numpy:
        X = torch.from_numpy(X)

    N, D = X.shape
    k = int(np.ceil(np.log2(D + 1)))
    target_dim = 2 ** k - 1

    pad_size = target_dim - D
    padded = torch.nn.functional.pad(X, (0, pad_size), mode="constant", value=0.0)

    return padded.numpy() if is_numpy else padded

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

if __name__ == "__main__":
    configs = [
        {"type": "gaussian", "mean": (0, 0), "sigma": (1, 1), "angle_deg": 0, "n_points": 500},
        {"type": "gaussian", "mean": (5, 5), "sigma": (0.5, 2), "angle_deg": 45, "n_points": 500},
        {"type": "ball", "mean": (2, -3), "radius": 1.0, "n_points": 400},
        {"type": "ball", "mean": (-4, 4), "radius": 0.5, "n_points": 300},
    ]

    X, y = generate_clusters(configs)
    plot_clusters(X, y)
