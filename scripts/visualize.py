#!/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation
import csv


def run(filein, outpath: str, fps: int, extent: float):
    # CSV reader for getting inputs
    csv_reader = csv.reader(filein)

    input_line = next(csv_reader)
    positions = np.array(input_line, dtype=np.float32).reshape(2, -1)

    figure, axes = plt.subplots()
    axes.set_xlim(-extent / 2, extent / 2)
    axes.set_ylim(-extent / 2, extent / 2)
    scatter = axes.scatter(positions[0, :], positions[1, :])

    def animate(_):
        input_line = next(csv_reader)
        positions = np.array(input_line, dtype=np.float32).reshape(2, -1)
        scatter.set_offsets(positions.T)
        return []

    try:
        animation = matplotlib.animation.FuncAnimation(
            figure, animate, cache_frame_data=False
        )
        gif_writer = matplotlib.animation.PillowWriter(fps)
        animation.save(outpath, writer=gif_writer)
    except StopIteration:
        pass


if __name__ == "__main__":
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser(description="visualize particle positions")
    parser.add_argument(
        "--extent", default=1.0, type=float, help="extent of visualized space"
    )
    parser.add_argument(
        "--fps",
        default=30,
        type=float,
        help="time delta between simulated frames",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="path to write output to",
    )

    arguments = parser.parse_args()

    run(sys.stdin, arguments.output, arguments.fps, arguments.extent)
