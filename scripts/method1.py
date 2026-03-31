#!/bin/env python

import graph_matching
import scipy.spatial
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt


def calculate_particle_assignments(
    old: np.ndarray, new: np.ndarray, epsilon: float, threshold: float
) -> np.ndarray:
    affinity = graph_matching.SpectralGraphMatchingInfo(old.T, new.T, epsilon)

    while graph_matching.iterate(affinity) > threshold:
        pass

    assignment = graph_matching.solution(affinity)

    return assignment


def run(infile):
    DIMENSIONS = 2

    # Construct CSV reader to get coordinates from input file
    csv_reader = csv.reader(infile)

    # Read the first line of the input so the main loop has a 'previous' to compare against
    first_line = next(csv_reader)
    previous_positions = np.array(first_line, np.float32).reshape(DIMENSIONS, -1)

    # We also need our initial triangulation
    triangulation = scipy.spatial.Delaunay(previous_positions.T)

    figure, axis = plt.subplots()

    simplices = triangulation.simplices

    # plot simplices
    for simplex in simplices:
        xs = previous_positions[0, simplex]
        ys = previous_positions[1, simplex]
        axis.plot(xs, ys, c="grey", zorder=1)

    axis.scatter(previous_positions[0, :], previous_positions[1, :])

    figure.savefig("figures/0.png")
    plt.close(figure)

    # Iterate all the remaining lines in the CSV
    for index, line in enumerate(csv_reader, start=1):
        # Get positions from the CSV input
        positions = np.array(line, np.float32).reshape(DIMENSIONS, -1)

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
            xs = positions[0, indices]
            ys = positions[1, indices]
            axis.plot(xs, ys, c="grey", zorder=1)

        axis.scatter(positions[0, :], positions[1, :])
        figure.savefig(f"figures/{index}.png")
        plt.close(figure)

        positions, previous_positions = previous_positions, positions


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Particle pairing example")
    arguments = parser.parse_args()
    run(sys.stdin)
