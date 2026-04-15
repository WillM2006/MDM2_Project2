#!/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from openpiv import tools, pyprocess, validation, filters, scaling
from PIL import Image


def run(directory: Path, count: int):
    print("reading images")
    images = []
    for i in range(1, count + 1):
        path = directory / f"{i}.png"
        image = tools.imread(path).astype(np.int8)
        images.append(image)

    window_size = 32
    overlap = 16
    print("working")
    for i in range(0, count - 1):
        u, v, signal_noise_ratio = pyprocess.extended_search_area_piv(
            images[i], images[i + 1], window_size, overlap
        )

        figure, axis = plt.subplots()
        axis.imshow(images[i])
        x = np.linspace(0, images[i].shape[0], num=u.shape[0])
        y = np.linspace(0, images[i].shape[1], num=u.shape[0])
        axis.quiver(x, y, u, v)
        figure.savefig(f"piv_figures/{i+1}.png")
        plt.close(figure)



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Example of particle image velocimetry via OpenPIV"
    )
    parser.add_argument(
        "--directory",
        required=True,
        type=Path,
        help="the directory to read images from",
    )
    parser.add_argument(
        "--count", required=True, type=int, help="how many images to read"
    )
    arguments = parser.parse_args()

    run(arguments.directory, arguments.count)
