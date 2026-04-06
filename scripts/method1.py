#!/bin/env python

import csv
import graph_matching
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.spatial
import sys

DIMENSIONS = 2
"""Constant number of dimensions, since this code is written for only a 2D slice of a 3D flow."""


def generate_corner_points(extent: float) -> np.ndarray:
    """Generate fixed corner points used to define the domain boundary during processing."""

    return np.array(
        [
            [-extent / 2, -extent / 2],
            [-extent / 2, extent / 2],
            [extent / 2, -extent / 2],
            [extent / 2, extent / 2],
        ]
    )


def read_points(line, corners: np.ndarray) -> np.ndarray:
    """Produce an array containing points extracted from a CSV reader iterator.
    Fixed corner points for the domain boundary are also added."""

    points = np.array(line, np.float32).reshape(DIMENSIONS, -1).T
    return np.concat([corners, points])


def assign_particles(
    old: np.ndarray, new: np.ndarray, epsilon: float, threshold: float
) -> np.ndarray:
    """Generate particle pairing between frames using the spectral graph matching method."""

    info = graph_matching.SpectralGraphMatchingInfo(old, new, epsilon)

    while graph_matching.iterate(info) > threshold:
        pass

    return graph_matching.solution(info)


def plot_frame(
    name,
    vertices: np.ndarray,
    simplices: np.ndarray,
    velocities: np.ndarray,
):
    """Plot a frame for visualization of the algorithm."""

    QUIVER_SCALE = 0.5
    LOOP_INDICES = [0, 1, 2, 0]

    # Create figure
    figure, axes = plt.subplots()

    # Plot simplices
    for simplex in simplices:
        indices = simplex[LOOP_INDICES]
        axes.plot(vertices[indices, 0], vertices[indices, 1], c="grey", zorder=1)

    # Plot vertex positions
    axes.scatter(vertices[:, 0], vertices[:, 1])

    # Plot velocities of each vertex
    axes.quiver(
        vertices[:, 0],
        vertices[:, 1],
        velocities[:, 0],
        velocities[:, 1],
        scale=QUIVER_SCALE,
        zorder=-1,
    )

    # Save and close the figure, creating the `figures/` directory if not existing
    figures_dir = Path("figures")
    if not figures_dir.exists():
        figures_dir.mkdir()
    figure.savefig(figures_dir / f"{name}.svg")
    plt.close(figure)


def iterate(
    index: int, simplices: np.ndarray, previous: np.ndarray, points: np.ndarray
):
    # Determine assignments
    EPSILON = 1.0
    COST_THRESHOLD = 1e-4
    assignments = assign_particles(previous, points, EPSILON, COST_THRESHOLD)

    # Update simplexes too newly reassigned particle indices
    simplices = assignments[simplices]

    # Caculate velocities at each vertex (backwards finite difference)
    velocities = np.zeros(points.shape)
    for i in range(previous.shape[0]):
        j = assignments[i]
        velocities[j, :] = points[j, :] - previous[i, :]

    # Save figure for visualization
    plot_frame(index, points, simplices, velocities)

    return simplices


def run(infile, extent: float):
    # Corner points used to fix the boundaries of the domain
    corners = generate_corner_points(extent)

    # Create iterator over lines of input from an external CSV file. See the readme for how this
    # is structured
    csv_reader = csv.reader(infile)

    # Assemble a set of points to process containing the next line of data from the CSV as well as
    # the corner points
    previous_positions = read_points(next(csv_reader), corners)

    # Calculate triangulation of the first frame
    triangulation = scipy.spatial.Delaunay(previous_positions)
    simplices = triangulation.simplices

    # Iterate over the remaining frames, generating data for each
    for index, line in enumerate(csv_reader, start=1):
        positions = read_points(line, corners)
        simplices = iterate(index, simplices, previous_positions, positions)
        positions, previous_positions = previous_positions, positions


if __name__ == "__main__":
    from argparse import ArgumentParser

    # Construct CLI argument parser
    parser = ArgumentParser(description="Particle pairing example")
    parser.add_argument("--extent", required=True, type=float)

    # Parse CLI argument
    arguments = parser.parse_args()

    # Pass stdin and arguments to program
    run(sys.stdin, arguments.extent)
