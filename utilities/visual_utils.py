import numpy as np

import matplotlib.pyplot as plt

def plot_clusters(X, y, figsize=(10, 10), title="Cluster Visualization"):
    """
    Визуализирует двумерные кластеры с разными цветами и маркерами.

    :param X: np.ndarray shape (n_samples, 2), координаты точек
    :param y: np.ndarray shape (n_samples,), метки классов
    :param figsize: tuple, размер графика
    :param title: str, заголовок
    """
    assert X.shape[1] == 2, "Функция поддерживает только 2D-визуализацию"

    # Уникальные метки и палитра
    classes = np.unique(y)
    colors = plt.cm.get_cmap("tab10", len(classes))
    markers = ['o', 's', 'P', 'X', 'D', '^', 'v', '<', '>', '*', 'H', '8', 'p', '+', 'x', '1', '2', '3', '4', '|', '_']

    # Красивые Unicode символы (масти, сердечки и т.п.)
    unicode_markers = ['$\heartsuit$', '$\spadesuit$', '$\clubsuit$', '$\diamondsuit$', '$\bigstar$', '$\bullet$', '$\checkmark$']
    full_markers = markers + unicode_markers

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    for i, cls in enumerate(classes):
        cls_points = X[y == cls]
        marker = full_markers[i % len(full_markers)]
        ax.scatter(cls_points[:, 0], cls_points[:, 1],
                   label=f"Class {cls}",
                   color=colors(i),
                   marker=marker,
                   edgecolors='k',
                   s=70,
                   alpha=0.85)

    # Оформление
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("X₁", fontsize=12)
    ax.set_ylabel("X₂", fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Центр координат и оси
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)

    ax.legend()
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.show()

def set_axes_equal(ax):
    '''Устанавливает одинаковый масштаб по осям x, y, z'''
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])

    for center, set_lim in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        set_lim(center - radius, center + radius)

def make_sphere(ax):
    # Нарисуем сферу
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)

    # Обрезаем нижнюю полусферу (если нужно)
    z_mask = z_sphere >= 0
    x_sphere = np.where(z_mask, x_sphere, np.nan)
    y_sphere = np.where(z_mask, y_sphere, np.nan)
    z_sphere = np.where(z_mask, z_sphere, np.nan)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.2, linewidth=0)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_transition(points_before, points_after, filename='transition.gif', steps=60):
    """
    Анимация перехода от points_before (shape N×d) к points_after (shape N×(d+1)).
    Строится плавный переход с добавлением новой координаты.
    """
    assert points_before.shape[0] == points_after.shape[0], "Points count mismatch"
    assert points_after.shape[1] == points_before.shape[1] + 1, "Output must have one extra dimension"

    N = points_before.shape[0]
    D = points_after.shape[1]

    # Дополняем points_before до той же размерности (добавляем 0-координату)
    points_before_ext = np.hstack([points_before, np.zeros((N, 1))])

    # Интерполяционная функция
    def interpolate(alpha):
        return (1 - alpha) * points_before_ext + alpha * points_after

    # Создание фигуры
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter([], [], [], c='blue')

    # Границы
    all_points = np.vstack([points_before_ext, points_after])
    ax.set_xlim([all_points[:, 0].min(), all_points[:, 0].max()])
    ax.set_ylim([all_points[:, 1].min(), all_points[:, 1].max()])
    ax.set_zlim([all_points[:, 2].min(), all_points[:, 2].max()])

    def update(frame):
        alpha = frame / steps
        interpolated = interpolate(alpha)
        scatter._offsets3d = (interpolated[:, 0], interpolated[:, 1], interpolated[:, 2])
        ax.set_title(f"Step {frame}/{steps}")
        return scatter,

    ani = FuncAnimation(fig, update, frames=steps + 1, interval=50, blit=False)

    # === Сохранение GIF ===
    ani.save(filename, writer=PillowWriter(fps=20))
    plt.close()

