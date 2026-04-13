import numpy as np
import csv
import scipy
import sys

DIMENSIONS = 2


def read_points(line, corners: np.ndarray) -> np.ndarray:
    """Produce an array containing points extracted from a CSV reader iterator.
    Fixed corner points for the domain boundary are also added."""

    points = np.array(line, np.float32).reshape(DIMENSIONS, -1).T
    return np.concat([corners, points])


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


def calculate_initial_affinity_matrix(
    old: np.ndarray, new: np.ndarray, epsilon: float
) -> np.ndarray:
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


def triangulate(positions: np.ndarray) -> np.ndarray:
    triangulation = scipy.spatial.Delaunay(positions)
    return triangulation.simplices


def assign_particles(cost_matrix: np.ndarray) -> np.ndarray:
    _, columns = scipy.optimize.linear_sum_assignment(cost_matrix)
    return columns


def calculate_velocities(
    old_positions: np.ndarray, new_positions: np.ndarray, assignments: np.ndarray
) -> np.ndarray:
    length = old_positions.shape[0]
    assert length == new_positions.shape[0]  # TODO: support non-equal size vector
    velocities = np.zeros(length)

    for i in range(length):
        j = assignments[i]
        velocities[j, :] = new_positions[j, :] - old_positions[i, :]

    return velocities


def calculate_mass_error(simplices, positions, velocities):
    pass


def calculate_momentum_error(simplices, positions, velocities):
    pass


def calculate_energy_error(simplices, positions, velocities):
    pass


def run(infile, extent: float, edgepoints: int):
    # todo: CLI arguments
    # tunable constants used to alter cost matrices during iteration
    MASS_CORRECTION_STRENGTH = 0.1
    MOMENTUM_CORRECTION_STRENGTH = 0.1
    ENERGY_CORRECTION_STRENGTH = 0.1
    ERROR_THRESHOLD = 0.001
    EPSILON = 1.0

    # only needed in triangular methods
    # Corner points used to fix the boundaries of the domain
    corners = generate_corner_points(extent, edgepoints)

    # get a CSV reader over stdin
    frames = csv.reader(infile)

    # get the first two frames to prime the iteration
    old_frame = read_points(next(frames), corners)
    current_frame = read_points(next(frames), corners)

    # iterate over all frames (current_frame) which have a predecessor (old_frame) and successor (new_frame)
    for new_frame_raw in frames:
        new_frame = read_points(new_frame_raw, corners)

        # calculate a triangulation on the current frame
        simplices = triangulate(current_frame)

        # generate initial affinity matrices
        backwards_affinities = calculate_initial_affinity_matrix(
            old_frame,
            current_frame,
            EPSILON,
        )
        forwards_affinities = calculate_initial_affinity_matrix(
            current_frame,
            new_frame,
            EPSILON,
        )

        # iterare until we converge on a result
        error = np.inf
        while error > ERROR_THRESHOLD:
            # calculate assignments from affinity matrices
            backwards_assignments = assign_particles(backwards_affinities)
            forwards_assignments = assign_particles(forwards_affinities)

            # calculate particle velocities
            old_velocities = calculate_velocities(
                old_frame, current_frame, backwards_assignments
            )
            new_velocities = calculate_velocities(
                current_frame, new_frame, forwards_assignments
            )

            # calculate errors derived from physical information (backwards)
            backwards_mass_error = calculate_mass_error(
                simplices, backwards_assignments, old_velocities
            )
            backwards_momentum_error = calculate_momentum_error(
                simplices, backwards_assignments, old_velocities
            )
            backwards_energy_error = calculate_energy_error(
                simplices, backwards_assignments, old_velocities
            )
            backwards_error = (
                MASS_CORRECTION_STRENGTH * backwards_mass_error
                + MOMENTUM_CORRECTION_STRENGTH * backwards_momentum_error
                + ENERGY_CORRECTION_STRENGTH * backwards_energy_error
            )

            # calculate errors derived from physical information (forwards)
            forwards_mass_error = calculate_mass_error(
                simplices, forwards_assignments, new_velocities
            )
            forwards_momentum_error = calculate_momentum_error(
                simplices, forwards_assignments, new_velocities
            )
            forwards_energy_error = calculate_energy_error(
                simplices, forwards_assignments, new_velocities
            )
            forwards_error = (
                MASS_CORRECTION_STRENGTH * forwards_mass_error
                + MOMENTUM_CORRECTION_STRENGTH * forwards_momentum_error
                + ENERGY_CORRECTION_STRENGTH * forwards_energy_error
            )

            # apply calculated errors (backwards)
            backwards_affinities -= backwards_error
            forwards_affinities -= forwards_error

            # recalculate total error
            error = np.sum(np.square(backwards_error)) + np.sum(
                np.square(forwards_error)
            )

        # reorganise frames for next iteration
        old_frame = current_frame
        current_frame = new_frame
