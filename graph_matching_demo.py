#!/bin/env python

import numpy as np
import scipy
import numba


@numba.jit
def initial_points(count: int) -> np.ndarray:
    return np.random.uniform(-3, 3, size=(2, count))


@numba.jit
def velocity_field(points: np.ndarray, t: float, v: float) -> np.ndarray:
    sines = np.sin(points)
    coses = np.cos(points)
    e = np.exp(-2 * v * t)

    velocities = np.zeros(points.shape)
    velocities[0, :] = sines[0, :] * coses[1, :] * e
    velocities[1, :] = -coses[0, :] * sines[1, :] * e
    return velocities


@numba.jit
def advect_points(points: np.ndarray, t: float, v: float, dt: float):
    return points + velocity_field(points, t, v) * dt


@numba.jit
def binary_affinity_measure(epsilon, x, y, i1, i2, j1, j2) -> float:
    xnorm = np.sqrt(np.sum(np.square(x[:, i1] - x[:, i2])))
    ynorm = np.sqrt(np.sum(np.square(y[:, j1] - y[:, j2])))
    measure = xnorm - ynorm
    return np.exp(-((measure / epsilon) ** 2))


@numba.jit
def affinity_matrix(epsilon, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    (_, n1) = x.shape
    (_, n2) = y.shape
    n = n1 * n2
    a = np.zeros((n, n))

    for i1 in range(n1):
        for i2 in range(n1):
            for j1 in range(n2):
                for j2 in range(n2):
                    ihat = i1 * n2 + j1
                    jhat = i2 * n2 + j2
                    a[ihat, jhat] = binary_affinity_measure(
                        epsilon, x, y, i1, i2, j1, j2
                    )

    return a


@numba.jit
def normalize(p: np.ndarray):
    return p / np.sqrt(np.sum(np.square(p)))


@numba.jit
def discretize(point_count, p: np.ndarray) -> np.ndarray:
    p = p.reshape((point_count, point_count))

    discrete = np.zeros(p.shape, dtype=np.int8)

    for j in range(point_count):
        i = np.argmax(p[:, j])
        discrete[i, j] = 1

    return discrete


@numba.jit
def probabalistic_graph_matching(
    point_count: int, affinity: np.ndarray, iterations: int, threshold: float
):
    (n, _) = affinity.shape

    p = np.repeat(1 / point_count, n)

    for _ in range(iterations):
        p_new = normalize(affinity @ p)
        p_frac = p_new / p
        for j in range(n):
            affinity[:, j] *= p_frac
        cost = np.sqrt(np.sum(np.square(p_new - p))) / n
        p = p_new
        if cost < threshold:
            break

    return discretize(point_count, p)


def generate_frames():
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.animation import FuncAnimation

    # Parameters
    t = 0.0
    dt = 0.05
    v = 0.4
    epsilon = 0.5
    match_iterations = 100
    match_threshold = 1e-3
    point_count = 100

    # Initial value
    points = initial_points(point_count)

    for index in range(100):
        new_points = advect_points(points, t, v, dt)

        affinity = affinity_matrix(epsilon, points, new_points)
        assignments = probabalistic_graph_matching(
            point_count, affinity, match_iterations, match_threshold
        )

        figure, axes = plt.subplots()
        axes.set_xlim(-3, 3)
        axes.set_ylim(-3, 3)
        axes.scatter(points[0, :], points[1, :], c="red", label="Old points")
        axes.scatter(new_points[0, :], new_points[1, :], c="blue", label="New points")
        for i in range(point_count):
            for j in range(point_count):
                if assignments[i, j] != 0:
                    xs = [points[0, i], new_points[0, j]]
                    ys = [points[1, i], new_points[1, j]]
                    axes.plot(xs, ys, c="grey")
                    break  # only 1 assignment per row
        axes.legend()
        figure.savefig(f"{index}.png")
        plt.close(figure)
        print(f"saved frame {index+1}/100")

        points = new_points

    # # Set up figure
    # figure, axes = plt.subplots()
    # scatter = axes.scatter(points[0, :], points[1, :])

    # # update function for animation & data
    # def update_figure(_: Figure):
    #     global points, t, axes
    #     t += dt
    #     new_points = advect_points(points, t, v, dt)
    #     np.random.default_rng().shuffle(new_points, axis=1)

    #     a = affinity_matrix(epsilon, points, new_points)
    #     matches = probabalistic_graph_matching(
    #         point_count, a, match_iterations, match_threshold
    #     )

    #     for i in range(point_count):
    #         for j in range(point_count):
    #             if matches[i, j] != 0:
    #                 xs = [points[0, i], new_points[0, j]]
    #                 ys = [points[1, i], new_points[1, j]]
    #                 axes.plot(xs, ys, c="grey")

    #     scatter.set_offsets(points.T)

    #     points = new_points

    #     return []

    # # run animation
    # _animation = FuncAnimation(
    #     figure, update_figure, interval=animation_interval_ms, save_count=40
    # )
    # plt.show()
    # plt.savefig(figure)


generate_frames()
