#!/bin/env python

import scipy.optimize
import scipy.spatial
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt


def calculate_cost_matrix(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    (_, n_old) = old.shape
    (_, n_new) = new.shape

    # TODO: Vectorized form
    costs = np.zeros((n_new, n_old))
    for i in range(n_old):
        for j in range(n_new):
            dx = new[0, j] - old[0, i]
            dy = new[1, j] - old[1, i]
            costs[i, j] = (dx * dx) + (dy * dy)

    return costs

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
    for index, line in enumerate(csv_reader):
        # Get positions from the CSV input
        positions = np.array(line, np.float32).reshape(DIMENSIONS, -1)

        # Determine assignment between previous and current frames
        cost_matrix = calculate_cost_matrix(previous_positions, positions)
        _, assignment = scipy.optimize.linear_sum_assignment(
            cost_matrix, maximize=False
        )

        # plot simplices
        figure, axis = plt.subplots()

        simplices = assignment[simplices]

        for indices in simplices:
            xs = positions[0, indices]
            ys = positions[1, indices]
            axis.plot(xs, ys, c="grey", zorder=1)

        axis.scatter(positions[0, :], positions[1, :])
        figure.savefig(f"figures/{index+1}.png")
        print(index+1)
        plt.close(figure)


        positions, previous_positions = previous_positions, positions

    # # Preallocate arrays and objects
    # rng = np.random.default_rng()
    # positions = np.zeros((DIMENSIONS, count))
    # next_positions = np.zeros((DIMENSIONS, count))
    # position_sines = np.zeros((DIMENSIONS, count))
    # position_cosines = np.zeros((DIMENSIONS, count))
    # velocities = np.zeros((DIMENSIONS, count))
    # cost_matrix = np.zeros((count, count))
    # time = 0

    # while True:
    #     # Initialize point positions (fake sample data)
    #     generate_random_points(rng, extent, count, positions)

    #     # Calculate next frame via simulation
    #     np.sin(positions, out=position_sines)
    #     np.cos(positions, out=position_cosines)
    #     sample_velocities(position_sines, position_cosines, velocities, nu, time)
    #     advect_points(positions, velocities, next_positions, timestep)
    #     shuffle_positions(rng, next_positions)  # avoid trivial assignments

    #     # Determine assignment between this and next frame
    #     calculate_cost_matrix(count, positions, next_positions, cost_matrix)
    #     _, assignment = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)

    #     # Set up for next frame
    #     next_positions, positions = positions, next_positions
    #     time += timestep

    #     print(assignment)
    #     break  # TODO


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Particle pairing example")
    arguments = parser.parse_args()
    run(sys.stdin)


# #!/bin/env python

# import numpy as np
# import matplotlib.pyplot as plt


# def generate_random_points(extent: float, count: int):
#     """Randomly generate `count` positions (2D) with all values in `[-extent/2, extent/2]`."""
#     return np.random.uniform(-extent / 2, extent / 2, size=(2, count))


# def sample_velocities(
#     position_sines: np.ndarray,
#     position_cosines: np.ndarray,
#     velocities: np.ndarray,
#     nu: float,
#     time: float,
# ):
#     """Write the sample velocity field values at `positions` to `velocities`"""
#     np.multiply(position_sines[0, :], position_cosines[1, :], out=velocities[0, :])
#     np.multiply(position_cosines[0, :], position_sines[1, :], out=velocities[1, :])
#     velocities[1, :] *= -1
#     # velocities *= np.exp(-2 * nu * time)


# def advect_points(
#     positions: np.ndarray,
#     velocities: np.ndarray,
#     positions_out: np.ndarray,
#     dt: float,
# ):
#     """Integrate 2D positions according to an sample velocity field."""
#     # positions_out = positions + (velocities * dt) written in-place to avoid allocation
#     np.multiply(velocities, dt, out=positions_out)
#     np.add(positions, positions_out, out=positions_out)


# def shuffle_positions(rng: np.random.Generator, positions: np.ndarray):
#     """Shuffle positions to avoid trivial particle matchings."""
#     rng.shuffle(positions, 1)


# def normalize(p: np.ndarray):
#     return p / np.sqrt(np.sum(np.square(p)))


# # TODO: as CLI arguments
# # Constants
# EXTENT = 6
# POINT_COUNT = 100
# TIMESTEP = 0.125
# ITERATIONS = 100
# NU = 0.5

# # Simulation data
# rng = np.random.default_rng()
# positions = generate_random_points(EXTENT, POINT_COUNT)
# position_sines = np.zeros(positions.shape)
# position_cosines = np.zeros(positions.shape)
# velocities = np.zeros(positions.shape)
# new_positions = np.zeros(positions.shape)
# time = 0

# # Plotting data
# figure, axes = plt.subplots()
# axes.set_title("Particles")
# axes.set_xlim(-EXTENT / 2, EXTENT / 2)
# axes.set_ylim(-EXTENT / 2, EXTENT / 2)
# scatter_old = axes.scatter(positions[0, :], positions[1, :])
# scatter_new = axes.scatter(positions[0, :], positions[1, :])

# for iteration in range(ITERATIONS):
#     # Simulate particle positions
#     np.sin(positions, out=position_sines)
#     np.cos(positions, out=position_cosines)
#     sample_velocities(position_sines, position_cosines, velocities, NU, time)
#     advect_points(positions, velocities, new_positions, TIMESTEP)
#     shuffle_positions(rng, new_positions)

#     # TODO: Matching

#     # Plot iteration figure
#     scatter_old.set_offsets(positions.T)
#     scatter_new.set_offsets(new_positions.T)
#     figure.savefig(f"figures/{iteration+1}.png")

#     # Update values for next iteration
#     new_positions, positions = positions, new_positions
#     time += TIMESTEP


# # def discretize(p):
# #     d = np.zeros(p.shape, dtype=np.int8)
# #     print(p)
# #     for i in range(p.shape[0]):
# #         d[i] = 1 if p[i] > 0.0001 else 0
# #     return d

# # def graph_match_iteration(affinity: np.ndarray, p: np.ndarray):
# #     q = affinity @ p
# #     p_new = normalize(q)

# #     p_frac = p_new / p
# #     affinity_new = np.zeros(affinity.shape)
# #     for j in range(affinity.shape[1]):
# #         affinity_new[:, j] = affinity[:, j] * p_frac

# #     cost = np.sqrt(np.sum(np.square(p_new - p))) / p.shape[0]
# #     return (affinity_new, p_new, cost)

# # def probabalistic_graph_matching(affinity: np.ndarray, iterations: int, threshold: float):
# #     (n, n_other) = affinity.shape
# #     assert n == n_other

# #     p = np.random.random(n)

# #     for _ in range(iterations):
# #         affinity, p, cost = graph_match_iteration(affinity, p)
# #         print(f"{cost=}")
# #         if cost < threshold:
# #             break

# #     return discretize(p)

# # def affinity_matrix(new, old, epsilon):
# #     (n_old, _) = old.shape
# #     (n_new, _) = new.shape

# #     n = n_new * n_old

# #     affinities = np.zeros((n, n))

# #     for i in range(n_old):
# #         for iprime in range(n_new):
# #             for j in range(n_old):
# #                 for jprime in range(n_new):
# #                     d_old_x = old[i, 0] - old[j, 0]
# #                     d_old_y = old[i, 1] - old[j, 1]
# #                     d_new_x = new[iprime, 0] - new[jprime, 0]
# #                     d_new_y = new[iprime, 1] - new[jprime, 1]

# #                     d_old = np.sqrt(d_old_x**2 + d_old_y**2)
# #                     d_new = np.sqrt(d_new_x**2 + d_new_y**2)

# #                     numerator = d_old - d_new
# #                     power = -(numerator / epsilon) ** 2

# #                     affinity = np.exp(power)

# #                     ai = (i - 1) * n_new + iprime
# #                     aj = (j-1) * n_new + jprime
# #                     affinities[ai, aj] = affinity

# #     return affinities

# # points = rand_points([-1, 1], [-1 ,1], 10)
# # dt = 0.01

# # for i in range(100):
# #     # generate next frame
# #     new_points = np.copy(points)
# #     advect_points(new_points, i * dt, dt)

# #     # compute affinity matrix
# #     print("generating affinity matrix...")
# #     epsilon = 100
# #     affinity = affinity_matrix(points, new_points, epsilon)
# #     print(affinity)

# #     # perform graph matching
# #     iterations = 100
# #     threshold = 0.00001
# #     print("matching graph...")
# #     output = probabalistic_graph_matching(affinity, iterations, threshold)
# #     print(np.reshape(output, (10, 10)))
# #     break

# #     # render figure
# #     plt.scatter(points[:, 0], points[:, 1])
# #     plt.savefig(f"figs/{i}.png")

# #     # set new frame as old frame
# #     points = new_points
