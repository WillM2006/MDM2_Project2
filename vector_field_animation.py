import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.animation import FuncAnimation

num_points = 50

width = (-5, 5)
height = (-5, 5)


x, z = np.meshgrid(np.linspace(width[0], width[1], 50),
                   np.linspace(height[0], height[1], 50))


def vector_field(x, z, t, dis=0.05):
    u = np.sin(x) * np.cos(z) * np.exp(-2 * dis * t)
    w = -np.cos(x) * np.sin(z) * np.exp(-2 * dis * t)
    return u, w


def rand_points(width, height, num):
    x = np.random.uniform(width[0], width[1], size=num)
    y = np.random.uniform(height[0], height[1], size=num)
    return np.column_stack((x, y))

def rand_points(width, height, num):
    x = np.random.uniform(width[0], width[1], size=num)
    y = np.random.uniform(height[0], height[1], size=num)
    return np.column_stack((x, y))


# simple Euler step moving particles with local velocity
def advect_points(points, t, dt=0.01):
    u_p, w_p = vector_field(points[:, 0], points[:, 1], t)
    points[:, 0] += u_p * dt
    points[:, 1] += w_p * dt
    return points


# Initial conditions
points = rand_points(width, height, num_points)


fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(width)
ax.set_ylim(height)

# initial background vector field
u_grid, w_grid = vector_field(x, z, 0)
quiv = ax.quiver(x, z, u_grid, w_grid)

# scatter for particles
scat = ax.scatter(points[:, 0], points[:, 1], c='r', s=15)

# line object for triangulation edges (we store as a single Line2D with NaNs between segments)
tri_lines, = ax.plot([], [], 'k-', lw=0.6)


def update(frame):
    global points

    t = frame * 0.01
    dt = 0.01

    # advect particles using velocity at particle positions
    points = advect_points(points, t, dt=dt)

    # recompute triangulation and prepare segment lists
    tri = Delaunay(points)
    segments_x = []
    segments_y = []
    for simplex in tri.simplices:
        for i in range(3):
            p1 = points[simplex[i]]
            p2 = points[simplex[(i + 1) % 3]]
            segments_x.extend([p1[0], p2[0], np.nan])
            segments_y.extend([p1[1], p2[1], np.nan])

    # update artists instead of recreating them
    tri_lines.set_data(segments_x, segments_y)
    scat.set_offsets(points)

    # update background vector field (on grid)
    u_grid, w_grid = vector_field(x, z, t)
    quiv.set_UVC(u_grid, w_grid)

    return scat, tri_lines, quiv


ani = FuncAnimation(fig=fig, func=update, frames=400, interval=30)

plt.show()