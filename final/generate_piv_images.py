#!/bin/env python
#
# Generate fake "input" images from generated data, used to evaluate PIV methods.
#

import sys
import numpy as np
import scipy.ndimage
from PIL import Image
import csv
from pathlib import Path


def generate_image(size: int, extent: float, positions: np.ndarray, blobsize: float):
    shape = (size, size)
    image = np.zeros(shape, dtype=np.uint8)

    for [x, y] in positions:
        # convert into pixel-coordinates before writing to image
        cx = int((x / extent + 0.5) * size)
        cy = int((y / extent + 0.5) * size)
        if cx < size and cy < size:
            image[cx, cy] = 255

    filtered = scipy.ndimage.gaussian_filter(image, blobsize)
    return filtered


def run(infile, outdir: Path, size: int, extent: float, blobsize: float):
    csv_reader = csv.reader(infile)

    if not outdir.exists():
        outdir.mkdir(exist_ok=True)

    for index, line in enumerate(csv_reader):
        points = np.array(line, np.float32).reshape(2, -1).T

        image = generate_image(size, extent, points, blobsize)

        filename = outdir / f"{index+1}.png"
        pil_image = Image.fromarray(image)
        pil_image.save(filename)


if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define CLI parser
    parser = ArgumentParser(description="Generate images from particles positions")
    parser.add_argument(
        "--size",
        help="size of the generated images",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--extent",
        help="extent of particle positions",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--blobsize",
        required=True,
        help="how much to blur generated images",
        type=float,
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="directory to write output images into",
        type=Path,
    )

    # Parse CLI arguments
    arguments = parser.parse_args()

    # Generate data
    run(
        sys.stdin,
        arguments.outdir,
        arguments.size,
        arguments.extent,
        arguments.blobsize,
    )
