#!/bin/env python

import general
import numpy as np
import scipy.interpolate
from numba import jit
from typing import Optional


@jit(nopython=True)
def evaluate_assignments(
    assignments: np.ndarray, previous_positions: np.ndarray, next_positions: np.ndarray
) -> np.ndarray:
    p = previous_positions.shape[0]
    n = next_positions.shape[0]

    penalties = np.ones((p, n))

    return penalties


def barycentric_interpolate(
    vertices: np.ndarray, values: np.ndarray, point: np.ndarray
) -> np.ndarray:
    """Given 3 `vertices` of a triangle each with associated `values`, find the value at `p` using
    barycentric interpolation."""

    interpolate = scipy.interpolate.LinearNDInterpolator(vertices, values)
    return interpolate(point)


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

    # cross product, third (off-plane) axis only
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


def triangulate(positions: np.ndarray) -> np.ndarray:
    delaunay = scipy.spatial.Delaunay(positions)
    return delaunay.simplices


def interpolate_velocities(
    gridpoints: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> np.ndarray:
    grid_size = gridpoints.shape[0]
    dimensions = 2

    simplices = triangulate(positions)

    interpolated_velocities = np.zeros((grid_size, grid_size, dimensions))

    for y in range(grid_size):
        for x in range(grid_size):
            position = np.array([gridpoints[x], gridpoints[y]])

            simplex_id = find_containing_simplex(position, positions, simplices)
            assert simplex_id is not None

            simplex = simplices[simplex_id]
            vertices = positions[simplex]
            vels = velocities[simplex]

            interpolated_velocities[y, x, :] = barycentric_interpolate(
                vertices, vels, position
            )

    return interpolated_velocities


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description="Velocity interpolation (Delaunay)")
    parser.add_argument(
        "--infile", type=Path, required=True, help="path to read input data from"
    )
    parser.add_argument(
        "--outfile", type=Path, required=True, help="path to write output data to"
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="directory to save visualisations to",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="initial position assignment parameter",
    )
    parser.add_argument(
        "--extent", type=float, required=True, help="size of the domain"
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        required=True,
        help="number of samples to predict velocities at",
    )
    parser.add_argument(
        "--edgepoints",
        type=int,
        required=True,
        help="number of points along each boundary edge",
    )

    arguments = parser.parse_args()
    infile = open(arguments.infile, "r")
    outfile = open(arguments.outfile, "w")

    general.run(
        infile,
        outfile,
        arguments.epsilon,
        arguments.extent,
        arguments.sample_count,
        arguments.edgepoints,
        arguments.frames_dir,
        evaluate_assignments,
        interpolate_velocities,
    )


# def assign_particles(old: np.ndarray, new: np.ndarray) -> np.ndarray:
#     """Generate particle pairing between frames using the spectral graph matching method."""

#     USE_SPECTRAL_METHOD = False

#     if USE_SPECTRAL_METHOD:
#         EPSILON = 1.0
#         COST_THRESHOLD = 1e-4

#         info = graph_matching.SpectralGraphMatchingInfo(old, new, EPSILON)

#         while graph_matching.iterate(info) > COST_THRESHOLD:
#             pass

#         return graph_matching.solution(info)
#     else:
#         cost_matrix = generate_assignment_cost_matrix(old, new)
#         _, columns = scipy.optimize.linear_sum_assignment(cost_matrix)
#         return columns

# def calculate_velocities(
#     old_positions: np.ndarray, new_positions: np.ndarray, assignments: np.ndarray
# ) -> np.ndarray:
#     length = old_positions.shape[0]
#     assert length == new_positions.shape[0]  # TODO: support non-equal size vector
#     velocities = np.zeros(length)

#     for i in range(length):
#         j = assignments[i]
#         velocities[j, :] = new_positions[j, :] - old_positions[i, :]

#     return velocities


# def generate_corner_points(extent: float, edgepoints: int) -> np.ndarray:
#     """Generate fixed corner and edge points on the domain boundary. These points must always have 0 velocity (no-slip)."""

#     points = np.linspace(-extent / 2, extent / 2, num=edgepoints + 1, endpoint=False)
#     points_flipped = np.linspace(
#         extent / 2, -extent / 2, num=edgepoints + 1, endpoint=False
#     )
#     lows = np.repeat(-extent / 2, edgepoints + 1)
#     highs = np.repeat(extent / 2, edgepoints + 1)

#     top = np.column_stack((points, highs))
#     right = np.column_stack((highs, points_flipped))
#     bottom = np.column_stack((points_flipped, lows))
#     left = np.column_stack((lows, points))

#     return np.concatenate((bottom, top, left, right))




# def run(infile, outfile, frames_dir: Path, extent: float, edgepoints: int):
#     # Corner points used to fix the boundaries of the domain
#     corners = generate_corner_points(extent, edgepoints)

#     # Create iterator over lines of input from an external CSV file. See the readme for how this
#     # is structured
#     csv_reader = csv.reader(infile)

#     # Assemble a set of points to process containing the first line of data from the CSV as well as
#     # the corner points
#     previous_positions = read_points(next(csv_reader), corners)

#     # Iterate over the remaining frames, generating data for each
#     for index, line in enumerate(csv_reader, start=1):
#         positions = read_points(line, corners)

#         # Determine assignments
#         assignments = assign_particles(previous_positions, positions)

#         # Update simplices to newly reassigned particle indices
#         # generate fresh triangulation of previous frame
#         triangulation = scipy.spatial.Delaunay(previous_positions)
#         simplices = triangulation.simplices
#         simplices = assignments[simplices]

#         # Caculate velocities at each vertex (backwards finite difference)
#         velocities = np.zeros(positions.shape)
#         for i in range(previous_positions.shape[0]):
#             j = assignments[i]
#             velocities[j, :] = positions[j, :] - previous_positions[i, :]

#         # Save figure for visualization
#         plot_frame(
#             index,
#             positions,
#             simplices,
#             velocities,
#             gridpoints,
#             interpolated_velocities,
#             frames_dir,
#         )

#         # swap previous and current positions for next iteration
#         positions, previous_positions = previous_positions, positions


# only needed in triangular methods
# Corner points used to fix the boundaries of the domain
# corners = generate_corner_points(extent, edgepoints)
