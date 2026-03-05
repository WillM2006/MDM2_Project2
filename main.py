#!/bin/env python

import numpy as np
import ot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def random_positions(count: int, scale: float) -> np.ndarray:
    return (np.random.random((count, 2)) - 0.5) * scale


def mock_velocity_field(position, t, nu) -> np.ndarray:
    # Eq. 1 and 2 from project brief.
    e = np.exp(-2.0 * nu * t)

    velocity = np.zeros(position.shape)

    velocity[:, 0] = np.sin(position[:, 0]) * np.cos(position[:, 1]) * e
    velocity[:, 1] = -np.cos(position[:, 0]) * np.sin(position[:, 1]) * e
    return velocity


def integrate_position(positions, t, nu):
    velocity = mock_velocity_field(positions, t, nu)
    positions += velocity * t

print("posistions...")

count = 500
first = random_positions(count, 1)

second = np.copy(first)
integrate_position(second, 0.2, 1.0)

np.random.shuffle(first)
np.random.shuffle(second)

print("cost matrix...")

costs = np.zeros((count, count))
for i in range(count):
    for j in range(count):
        dx = first[i, 0] - second[j, 0]
        dy = first[i, 1] - second[j, 1]
        distance = np.abs((dx * dx) + (dy * dy))
        costs[i, j] = distance

print("assignments...")

plan = ot.solve(costs).plan

print("figure...")

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Particle assignment between frames")

ax.scatter(first[:, 0], first[:, 1], c="r", label="Previous")
ax.scatter(second[:, 0], second[:, 1], c="b", label="Next")

for i in range(count):
    for j in range(count):
        if plan[i, j] != 0:
            xs = [first[i, 0], second[j, 0]]
            ys = [first[i, 1], second[j, 1]]
            ax.add_line(Line2D(xs, ys, color="grey"))

ax.legend()

fig.savefig("figure.png", dpi=100)
