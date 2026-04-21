#!/bin/env python

import csv
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from numba import jit
from typing import Optional

DIMENSIONS = 2
"""Constant number of dimensions, since this code is written for only a 2D slice of a 3D flow."""


def generate_corner_points(extent: float, edgepoints: int) -> np.ndarray:
    """Generate fixed corner and edge points on the domain boundary. These points must always have 0 velocity (no-slip)."""

    points = np.linspace(-extent / 2, extent / 2, num=edgepoints + 1, endpoint=False)
    points_flipped = np.linspace(
        extent / 2, -extent / 2, num=edgepoints + 1, endpoint=False
    )
    lows = np.repeat(-extent / 2, edgepoints + 1)
    highs = np.repeat(extent / 2, edgepoints + 1)

    top = np.column_stack((points, highs))
    right = np.column_stack((highs, points_flipped))
    bottom = np.column_stack((points_flipped, lows))
    left = np.column_stack((lows, points))

    return np.concatenate((bottom, top, left, right))


def read_points(line, corners: np.ndarray) -> np.ndarray:
    """Produce an array containing points extracted from a CSV reader iterator.
    Fixed corner points for the domain boundary are also added."""

    points = np.array(line, np.float32).reshape(DIMENSIONS, -1).T
    return np.concat([corners, points])


@jit(nopython=True)
def generate_assignment_cost_matrix(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    (n_old, _) = old.shape
    (n_new, _) = new.shape

    costs = np.zeros((n_old, n_new))

    for y in range(n_old):
        for x in range(n_new):
            costs[y, x] = np.sqrt(np.sum(np.square(old[y, :] - new[x, :])))

    return costs


def assign_particles(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Generate particle pairing between frames using the spectral graph matching method."""

    cost_matrix = generate_assignment_cost_matrix(old, new)
    _, columns = linear_sum_assignment(cost_matrix)

    return columns


@jit(nopython=True)
def triangle_contains_point(vertices: np.ndarray, position: np.ndarray) -> bool:
    """Returns `True` if the triangle defined by `vertices` contains `position`. `False` otherwise."""

    # precalculate relevant vectors
    ab = vertices[1, :] - vertices[0, :]
    bc = vertices[2, :] - vertices[1, :]
    ca = vertices[0, :] - vertices[2, :]
    ap = vertices[1, :] - position
    bp = vertices[2, :] - position
    cp = vertices[0, :] - position

    # cross product, third (non-plane) axis only
    # if all signs are the same, then the point is inside the triangle
    # so we only have to determine the sign (True if positive, False otherwise).
    a_sign = ab[0] * ap[1] > ab[1] * ap[0]
    b_sign = bc[0] * bp[1] > bc[1] * bp[0]
    c_sign = ca[0] * cp[1] > ca[1] * cp[0]

    # true iff all cross-product signs are the same
    return (a_sign == b_sign) and (b_sign == c_sign) and (c_sign == a_sign)


@jit(nopython=True)
def find_containing_simplex(
    position: np.ndarray, positions: np.ndarray, simplices: np.ndarray
) -> Optional[int]:
    """Given a `position`, find a simplex in `simplices` (as triples of indicies into `positions`) which contains it."""

    # TODO: construct spatial partitioning tree to speed this up (O(n) to O(log n)).

    for index, simplex in enumerate(simplices):
        vertices = positions[simplex]
        if triangle_contains_point(vertices, position):
            return index

    return None


def barycentric_interpolate(
    vertices: np.ndarray, values: np.ndarray, p: np.ndarray
) -> np.ndarray:
    """Given 3 `vertices` of a triangle each with associated `values`, find the value at `p` using
    barycentric interpolation."""

    interpolate = LinearNDInterpolator(vertices, values)
    return interpolate(p)


def save_frame(
    output: Path, xs: np.ndarray, ys: np.ndarray, us: np.ndarray, vs: np.ndarray
):
    with open(output) as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "u", "v"])

        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                u = us[xi, yi]
                v = vs[xi, yi]
                writer.writerow([x, y, u, v])


def main(infile, outdir: Path, extent: float, edgepoints: int, gridsize: int):
    # if the output directory doesn't exist, create it
    outdir.mkdir(exist_ok=True)

    # Points of a grid at which velocities will be interpolated
    gridpoints = np.linspace(-extent / 2, extent / 2, gridsize)
    interpolated_velocities = np.zeros((gridsize, gridsize, DIMENSIONS))

    # Corner points used to fix the boundaries of the domain
    corners = generate_corner_points(extent, edgepoints)

    # Create iterator over lines of input from an external CSV file. See the readme for how this
    # is structured
    csv_reader = csv.reader(infile)

    # Assemble a set of points to process, containing the first line of data from the CSV as well
    # as the corner points
    previous_positions = read_points(next(csv_reader), corners)

    # Iterate over the remaining frames, generating data for each
    for index, line in enumerate(csv_reader, start=1):
        positions = read_points(line, corners)

        # Determine assignments
        assignments = assign_particles(previous_positions, positions)

        # Update simplices to newly reassigned particle indices
        # generate fresh triangulation of previous frame
        triangulation = Delaunay(previous_positions)
        old_simplices = triangulation.simplices
        new_simplices = assignments[old_simplices]

        # Caculate velocities at each vertex (backwards finite difference)
        velocities = np.zeros(positions.shape)
        for i in range(previous_positions.shape[0]):
            j = assignments[i]
            velocities[j, :] = positions[j, :] - previous_positions[i, :]

        for y in range(gridsize):
            for x in range(gridsize):
                position = np.array([gridpoints[x], gridpoints[y]])

                simplex_id = find_containing_simplex(position, positions, new_simplices)
                assert simplex_id is not None

                simplex = new_simplices[simplex_id]
                vertices = positions[simplex]
                vels = velocities[simplex]

                interpolated_velocities[y, x, :] = barycentric_interpolate(
                    vertices, vels, position
                )

        # Save figure for visualization
        output_frame_name = outdir / f"{index}.csv"
        save_frame(
            output_frame_name,
            gridpoints,
            gridpoints,
            interpolated_velocities[:, :, 0],
            interpolated_velocities[:, :, 1],
        )

        # swap previous and current positions for next iteration
        positions, previous_positions = previous_positions, positions


if __name__ == "__main__":
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser(
        description="Extract velocity information from a CSV of particle positions"
    )
    parser.add_argument("--infile", type=Path, required=True, help="The file to read CSV data from")
    parser.add_argument(
        "--extent",
        required=True,
        type=float,
        help="maximum extent of data from the origin in either axis",
    )
    parser.add_argument(
        "--edgepoints",
        required=True,
        type=int,
        help="the number of fixed points placed around the boundary",
    )
    parser.add_argument(
        "--gridsize",
        required=True,
        type=int,
        help="the density of points to sample velocity at along both axes",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="The directory to write extracted velocity information to",
    )

    arguments = parser.parse_args()
    infile = open(arguments.infile, "r")

    main(
        infile,
        arguments.outdir,
        arguments.extent,
        arguments.edgepoints,
        arguments.gridsize,
    )
