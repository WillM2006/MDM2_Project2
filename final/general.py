#!/bin/env python
#
# Shared code used between all three methods.
#

import csv
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from typing import Callable
from numba import jit
from pathlib import Path

DIMENSIONS = 2


def _read_points(line, boundary_points) -> np.ndarray:
    """Produce an array containing points extracted from a CSV reader iterator.
    Fixed corner points for the domain boundary are also added."""

    points = np.array(line, np.float32).reshape(DIMENSIONS, -1).T
    return np.concat([points, boundary_points])


@jit(nopython=True)
def _calculate_initial_affinity_matrix(
    old: np.ndarray, new: np.ndarray, epsilon: float
) -> np.ndarray:
    """Calculate the initial affinity matrix for assigning particles, using position data only"""

    (n_old, _) = old.shape
    (n_new, _) = new.shape

    costs = np.zeros((n_old, n_new))

    for y in range(n_old):
        for x in range(n_new):
            sqr_distance = np.sum(np.square(old[y, :] - new[x, :]))
            costs[y, x] = sqr_distance

    # convert from costs to affinities
    e2 = epsilon**2
    return np.exp(-costs / e2)


def _assign_particles(affinities: np.ndarray) -> np.ndarray:
    """Determine the solution to the linear sum assignment problemn given a matrix of affinities (scores)"""

    _, columns = scipy.optimize.linear_sum_assignment(affinities, maximize=True)
    return columns


def _count_differences(assignments: np.ndarray, new_assignments: np.ndarray) -> int:
    """Calculate the number of differences between two arrays of the same length"""

    (length,) = assignments.shape
    assert new_assignments.shape[0] == length

    differences = 0

    for i in range(length):
        if assignments[i] != new_assignments[i]:
            differences += 1

    return differences


def _run_frame_pair(
    previous_positions: np.ndarray,
    next_positions: np.ndarray,
    epsilon: float,
    evaluate_assignments: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Using the given `evaluate_assignments` function, determine an assignment of particles between two frames."""

    MAX_ITERATIONS = 10

    # determine initial affinities of assignments (numpy matrix)
    affinity_matrix = _calculate_initial_affinity_matrix(
        previous_positions, next_positions, epsilon
    )

    # use this to calculate initial particle assignments
    assignments = _assign_particles(affinity_matrix)

    # apply penalties until assignments stop changing
    changed_assignment_count = assignments.shape[0]
    iterations = 0
    while changed_assignment_count > 0 and iterations < MAX_ITERATIONS:
        iterations += 1

        # evaluate the assignments and get a penalty matrix
        penalties = evaluate_assignments(
            assignments, previous_positions, next_positions
        )

        # update the new affinity matrix with the calculated penalties
        affinity_matrix *= penalties

        # calculate new assignments from the updated affinities
        new_assignments = _assign_particles(affinity_matrix)

        # check how many assignments were changed
        changed_assignment_count = _count_differences(assignments, new_assignments)

        # end of assignment iteration, new assignments become old assignments
        assignments = new_assignments

    return assignments


def _calculate_velocities(
    old_positions: np.ndarray, new_positions: np.ndarray, assignments: np.ndarray
) -> np.ndarray:
    length = old_positions.shape[0]
    assert length == new_positions.shape[0]  # TODO: support differently sized vectors
    velocities = np.zeros((length, 2))

    for i in range(length):
        j = assignments[i]
        velocities[j, :] = new_positions[j, :] - old_positions[i, :]

    return velocities


def _serialize_interpolated_velocities(
    gridpoints: np.ndarray, velocities: np.ndarray, index: int, csv_writer
):
    """Write interpolated velocities into the output CSV"""

    xs, ys = np.meshgrid(gridpoints, gridpoints)
    us = velocities[:, :, 0].ravel()
    vs = velocities[:, :, 1].ravel()

    csv_writer.writerow(xs.ravel())
    csv_writer.writerow(ys.ravel())
    csv_writer.writerow(us.ravel())
    csv_writer.writerow(vs.ravel())


def _generate_boundary_points(extent: float, count: int) -> np.ndarray:
    """Generate fixed corner and edge points on the domain boundary. These points must always have 0 velocity (no-slip)."""

    extent = extent + 0.5  # add space around the boundary

    points = np.linspace(-extent / 2, extent / 2, num=count + 1, endpoint=False)
    points_flipped = np.linspace(extent / 2, -extent / 2, num=count + 1, endpoint=False)
    lows = np.repeat(-extent / 2, count + 1)
    highs = np.repeat(extent / 2, count + 1)

    top = np.column_stack((points, highs))
    right = np.column_stack((highs, points_flipped))
    bottom = np.column_stack((points_flipped, lows))
    left = np.column_stack((lows, points))

    return np.concatenate((bottom, top, left, right))


def _plot_frame(
    name,
    vertices: np.ndarray,
    velocities: np.ndarray,
    gridpoints: np.ndarray,
    interpolated_velocities: np.ndarray,
    frames_dir: Path,
    extent: float,
):
    """Plot a frame for visualization of the algorithm."""

    # Create figure
    figure, axes = plt.subplots()
    axes.set_aspect(1.0)
    axes.set_xlim(-extent / 2, extent / 2)
    axes.set_ylim(-extent / 2, extent / 2)

    # Plot vertex positions
    axes.scatter(vertices[:, 0], vertices[:, 1], s=18, zorder=2)

    # Plot velocities of each vertex
    axes.quiver(
        vertices[:, 0],
        vertices[:, 1],
        velocities[:, 0],
        velocities[:, 1],
        scale=2,
        zorder=3,
    )

    # Plot interpolated velocities
    axes.quiver(
        gridpoints,
        gridpoints,
        interpolated_velocities[:, :, 0],
        interpolated_velocities[:, :, 1],
        scale=4,
        zorder=3,
    )

    if not frames_dir.exists():
        frames_dir.mkdir()
    figure.savefig(frames_dir / f"{name}.svg")
    plt.close(figure)


def run(
    infile,
    outfile,
    epsilon: float,
    extent: float,
    sample_count: int,
    boundary_point_count: int,
    frames_dir: Path,
    evaluate_assignments: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    interpolate_velocities: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
):
    """
    Reads position data from the CSV file at `file`.
    Iteratively evaluates each frame pair's assignments using `evaluate_assignments` function.
    Calculates interpolated velocities using `interpolate_velocities` function.
    """

    # generate points to define the boundary of the domain
    boundary_points = _generate_boundary_points(extent, boundary_point_count)

    # set up csv reader and writer (data input and output)
    frames = csv.reader(infile)
    csv_writer = csv.writer(outfile)

    # load the first 'current' frame from the csv
    current_positions = _read_points(next(frames), boundary_points)

    gridpoints = np.linspace(-extent / 2, extent / 2, sample_count)

    # iterate through pairs of frames (current_frame, next_frame)
    for index, next_frame_data in enumerate(frames, start=1):
        # convert the string data into a numpy array
        next_positions = _read_points(next_frame_data, boundary_points)

        # determine the optimal assignments between frame pairs
        assignments = _run_frame_pair(
            current_positions, next_positions, epsilon, evaluate_assignments
        )

        # calculate velocities
        velocities = _calculate_velocities(
            current_positions, next_positions, assignments
        )

        # use callback to determine velocities interpolated between particle positions
        interpolated_velocities = interpolate_velocities(
            gridpoints, next_positions, velocities
        )

        # save the interpolated velocities to the output CSV file
        _serialize_interpolated_velocities(
            gridpoints, interpolated_velocities, index, csv_writer
        )

        # additionally, serialize a visualisation for debugging/presentation
        _plot_frame(
            index,
            next_positions,
            velocities,
            gridpoints,
            interpolated_velocities,
            frames_dir,
            extent,
        )

        # end of frame iteration, next frame become current frame
        current_positions, next_positions = next_positions, current_positions
