#!/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from openpiv import tools, pyprocess
import csv


def run(directory: Path, visdir: Path, outfile, count: int):
    csv_writer = csv.writer(outfile)

    if not visdir.exists():
        visdir.mkdir(exist_ok=True)

    images = []
    for i in range(1, count + 1):
        path = directory / f"{i}.png"
        image = tools.imread(path).astype(np.int8)
        images.append(image)

    window_size = 32
    overlap = 16
    for i in range(0, count - 1):
        u, v, _ = pyprocess.extended_search_area_piv(
            images[i], images[i + 1], window_size, overlap
        )

        figure, axis = plt.subplots()
        axis.imshow(images[i])

        x, y = pyprocess.get_coordinates(
            image_size=images[i].shape,
            search_area_size=window_size,
            overlap=overlap,
        )

        # write four rows to represent the gridpoints and velocities of this frame
        csv_writer.writerow(x.ravel())
        csv_writer.writerow(y.ravel())
        csv_writer.writerow(u.ravel())
        csv_writer.writerow(v.ravel())

        axis.quiver(x, y, u, -v, scale=30)
        figure.savefig(visdir / f"{i+1}.png")
        plt.close(figure)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Example of particle image velocimetry via OpenPIV"
    )
    parser.add_argument(
        "--indir",
        required=True,
        type=Path,
        help="the directory to read images from",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="the directory to write images from",
    )
    parser.add_argument(
        "--outfile",
        required=True,
        type=Path,
        help="file to write sampled gridpoints to",
    )
    parser.add_argument(
        "--count", required=True, type=int, help="how many images to read"
    )
    arguments = parser.parse_args()

    outfile = open(
        arguments.outfile, "w"
    )  # fine to not use `with` since we're only using one `open` per execution
    run(arguments.indir, arguments.outdir, outfile, arguments.count)
