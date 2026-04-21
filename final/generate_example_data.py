#!/bin/env python
#
# Script for generating sample particle data for testing.
#

import sys
import csv
import numpy as np


def generate_random_points(
    rng: np.random.Generator, extent: float, count: int, out: np.ndarray
):
    """Generate `(2, count)` values in the interval `[-extent/2, extent/2]`"""
    rng.random(size=(2, count), dtype=np.float32, out=out)
    out -= np.float32(0.5)
    out *= np.float32(extent)


def sample_velocities(
    position_sines: np.ndarray,
    position_cosines: np.ndarray,
    velocities: np.ndarray,
    nu: float,
    time: float,
):
    """Write the sample velocity field values at `positions` to `velocities`"""
    np.multiply(position_sines[0, :], position_cosines[1, :], out=velocities[0, :])
    np.multiply(position_cosines[0, :], position_sines[1, :], out=velocities[1, :])
    velocities[1, :] *= -1
    velocities *= np.exp(-2 * nu * time)


def advect_points(
    positions: np.ndarray,
    velocities: np.ndarray,
    positions_out: np.ndarray,
    dt: float,
):
    """Integrate 2D positions according to an sample velocity field."""
    # simple euler integration, since this doesn't need to be very physically accurate
    # positions_out = positions + (velocities * dt) written in-place to avoid allocation
    np.multiply(velocities, dt, out=positions_out)
    np.add(positions, positions_out, out=positions_out)


def shuffle_positions(rng: np.random.Generator, positions: np.ndarray):
    """Shuffle positions to avoid trivial particle matchings."""
    rng.shuffle(positions, 1)


def generate(
    seed: int,
    count: int,
    frames: int,
    extent: float,
    timestep: float,
    dissipation: float,
):
    # Allocate objects
    rng = np.random.default_rng(seed)
    time = 0.0
    csv_writer = csv.writer(sys.stdout)

    # Allocate arrays
    positions = np.zeros((DIMENSIONS, count), dtype=np.float32)
    velocities = np.zeros((DIMENSIONS, count), dtype=np.float32)
    position_sines = np.zeros((DIMENSIONS, count), dtype=np.float32)
    position_cosines = np.zeros((DIMENSIONS, count), dtype=np.float32)
    next_positions = np.zeros((DIMENSIONS, count), dtype=np.float32)

    # Randomly generate initial positions
    generate_random_points(rng, extent, count, positions)

    # Write the initial positions to stdout
    csv_writer.writerow(positions.ravel())

    # Start simulating
    for _ in range(frames):
        # Calculate velocities and the next positions
        np.sin(positions, out=position_sines)
        np.cos(positions, out=position_cosines)
        sample_velocities(
            position_sines, position_cosines, velocities, dissipation, time
        )
        advect_points(positions, velocities, next_positions, timestep)
        shuffle_positions(rng, next_positions)  # avoid trivial assignments

        # Prepare for next iteration
        time += timestep
        positions, next_positions = next_positions, positions

        # Write the new positions to stdout
        csv_writer.writerow(positions.ravel())
        sys.stdout.flush() # make sure results are written immediately


if __name__ == "__main__":
    from argparse import ArgumentParser

    DIMENSIONS = 2  # 2D data only

    # Define CLI parser
    parser = ArgumentParser(description="Generate particle position timeseries")
    parser.add_argument("--count", required=True, help="number of particles", type=int)
    parser.add_argument(
        "--frames", required=True, help="number of frames to simulate", type=int
    )
    parser.add_argument(
        "--extent",
        help="extent of initial particle positions",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--seed", help="value to seed RNG with (random by default)", type=int
    )
    parser.add_argument(
        "--timestep",
        default=0.1,
        help="duration between frames (0.1 by default)",
        type=float,
    )
    parser.add_argument(
        "--dissipation",
        default=1.0,
        help="how quickly the velocity dissipates in the simulated flow (1.0 by default)",
        type=float,
    )

    # Parse CLI arguments
    arguments = parser.parse_args()

    # Generate data
    generate(
        arguments.seed,
        arguments.count,
        arguments.frames,
        arguments.extent,
        arguments.timestep,
        arguments.dissipation,
    )
