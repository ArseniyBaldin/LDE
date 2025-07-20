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
