#!/bin/env python

import graph_matching
import scipy.spatial
import scipy.interpolate
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt


def calculate_particle_assignments(
    old: np.ndarray, new: np.ndarray, epsilon: float, threshold: float
) -> np.ndarray:
    affinity = graph_matching.SpectralGraphMatchingInfo(old, new, epsilon)

    while graph_matching.iterate(affinity) > threshold:
        pass

    assignment = graph_matching.solution(affinity)

    return assignment

def get_simplex(simplices: np.ndarray, x: float, y: float) -> np.ndarray:
    """Return the simplex (a, b, c) containing the position (x, y)."""

    return simplices[0] # TODO


def run(infile, extent: float):
    DIMENSIONS = 2

    corners = np.array(
        [
            [-extent / 2, -extent / 2],
            [-extent / 2, extent / 2],
            [extent / 2, -extent / 2],
            [extent / 2, extent / 2],
        ]
    )

    # Construct CSV reader to get coordinates from input file
    csv_reader = csv.reader(infile)

    # Read the first line of the input so the main loop has a 'previous' to compare against
    first_line = next(csv_reader)
    previous_positions = np.array(first_line, np.float32).reshape(DIMENSIONS, -1).T
    previous_positions = np.concat([corners, previous_positions])

    # We also need our initial triangulation
    triangulation = scipy.spatial.Delaunay(previous_positions)

    figure, axis = plt.subplots()

    simplices = triangulation.simplices

    # plot simplices
    for simplex in simplices:
        xs = previous_positions[simplex[[0, 1, 2, 0]], 0]
        ys = previous_positions[simplex[[0, 1, 2, 0]], 1]
        axis.plot(xs, ys, c="grey", zorder=1)

    axis.scatter(previous_positions[:, 0], previous_positions[:, 1])

    figure.savefig("figures/0.png")
    plt.close(figure)

    gridpoints = np.linspace(-extent / 2, extent / 2)

    # Iterate all the remaining lines in the CSV
    for index, line in enumerate(csv_reader, start=1):
        # Get positions from the CSV input
        positions = np.array(line, np.float32).reshape(DIMENSIONS, -1).T
        positions = np.concat([corners, positions])

        # Determine assignments
        EPSILON = 1.0
        COST_THRESHOLD = 1e-4
        assignments = calculate_particle_assignments(
            previous_positions, positions, EPSILON, COST_THRESHOLD
        )

        # plot simplices
        figure, axis = plt.subplots()

        simplices = assignments[simplices]

        for indices in simplices:
            xs = positions[indices[[0, 1, 2, 0]], 0]
            ys = positions[indices[[0, 1, 2, 0]], 1]
            axis.plot(xs, ys, c="grey", zorder=1)

        velocities = np.zeros(positions.shape)
        for i in range(previous_positions.shape[0]):
            j = assignments[i]
            velocities[j, :] = positions[j, :] - previous_positions[i, :]

        axis.quiver(positions[:, 0], positions[:, 1], velocities[:, 0], velocities[:, 1], scale=0.5, zorder=-1)

        # interpolated_velocities_x = np.ones((50, 50))
        # interpolated_velocities_y = np.ones((50, 50))

        # for i in range(50):
        #     for j in range(50):
        #         x = gridpoints[i]
        #         y = gridpoints[j]

        #         # identify simplex
        #         (a, b, c) = todo()

        #         # barycentric interpolation
        #         value = scipy.interpolate.barycentric_interpolate(xcoords, ycoords, )

        #         interpolated_velocities_x[i, j] = vx
        #         interpolated_velocities_y[i, j] = vy

        # axis.quiver(
        #     gridpoints, gridpoints, interpolated_velocities_x, interpolated_velocities_y
        # )


        axis.scatter(positions[:, 0], positions[:, 1])
        figure.savefig(f"figures/{index}.png")
        plt.close(figure)

        positions, previous_positions = previous_positions, positions


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Particle pairing example")
    parser.add_argument("--extent", required=True, type=float)
    arguments = parser.parse_args()
    run(sys.stdin, arguments.extent)
