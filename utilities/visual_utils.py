import numpy as np

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

