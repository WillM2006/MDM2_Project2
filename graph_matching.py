#!/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def rand_points(width, height, num):
    x = np.random.uniform(width[0], width[1], size=num)
    y = np.random.uniform(height[0], height[1], size=num)
    return np.column_stack((x, y))

def vector_field(x, z, t, dis=0.05):
    u = np.sin(x) * np.cos(z) * np.exp(-2 * dis * t)
    w = -np.cos(x) * np.sin(z) * np.exp(-2 * dis * t)
    return u, w

# simple Euler step moving particles with local velocity
def advect_points(points, t, dt=0.01):
    u_p, w_p = vector_field(points[:, 0], points[:, 1], t)
    points[:, 0] += u_p * dt
    points[:, 1] += w_p * dt

# No normalization, since it requires no-outlier condition
def normalize(x):
    size = np.sum(x)
    return x / size

def discretize(p):
    d = np.zeros(p.shape, dtype=np.int8)
    print(p)
    for i in range(p.shape[0]):
        d[i] = 1 if p[i] > 0.0001 else 0
    return d

def graph_match_iteration(affinity: np.ndarray, p: np.ndarray):
    q = affinity @ p
    p_new = normalize(q)

    p_frac = p_new / p
    affinity_new = np.zeros(affinity.shape)
    for j in range(affinity.shape[1]):
        affinity_new[:, j] = affinity[:, j] * p_frac

    cost = np.sqrt(np.sum(np.square(p_new - p))) / p.shape[0]
    return (affinity_new, p_new, cost)

def probabalistic_graph_matching(affinity: np.ndarray, iterations: int, threshold: float):
    (n, n_other) = affinity.shape
    assert n == n_other

    p = np.random.random(n)

    for _ in range(iterations):
        affinity, p, cost = graph_match_iteration(affinity, p)
        print(f"{cost=}")
        if cost < threshold:
            break

    return discretize(p)

def affinity_matrix(new, old, epsilon):
    (n_old, _) = old.shape
    (n_new, _) = new.shape

    n = n_new * n_old

    affinities = np.zeros((n, n))

    for i in range(n_old):
        for iprime in range(n_new):
            for j in range(n_old):
                for jprime in range(n_new):
                    d_old_x = old[i, 0] - old[j, 0]
                    d_old_y = old[i, 1] - old[j, 1]
                    d_new_x = new[iprime, 0] - new[jprime, 0]
                    d_new_y = new[iprime, 1] - new[jprime, 1]

                    d_old = np.sqrt(d_old_x**2 + d_old_y**2)
                    d_new = np.sqrt(d_new_x**2 + d_new_y**2)

                    numerator = d_old - d_new
                    power = -(numerator / epsilon) ** 2

                    affinity = np.exp(power)

                    ai = (i - 1) * n_new + iprime
                    aj = (j-1) * n_new + jprime
                    affinities[ai, aj] = affinity

    return affinities

points = rand_points([-1, 1], [-1 ,1], 10)
dt = 0.01

for i in range(100):
    # generate next frame
    new_points = np.copy(points)
    advect_points(new_points, i * dt, dt)

    # compute affinity matrix
    print("generating affinity matrix...")
    epsilon = 100
    affinity = affinity_matrix(points, new_points, epsilon)
    print(affinity)

    # perform graph matching
    iterations = 100
    threshold = 0.00001
    print("matching graph...")
    output = probabalistic_graph_matching(affinity, iterations, threshold)
    print(np.reshape(output, (10, 10)))
    break

    # render figure
    plt.scatter(points[:, 0], points[:, 1])
    plt.savefig(f"figs/{i}.png")

    # set new frame as old frame
    points = new_points
