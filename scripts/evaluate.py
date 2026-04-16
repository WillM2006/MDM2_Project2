#!/bin/env python
#
# script for evaluating the performance of velocity extraction methods
#

import sys
import csv
from pathlib import Path

# TODO: this is unfinished

def run(infile):
    reader = csv.reader(infile)

    while True:
        xs = next(reader)
        ys = next(reader)
        us = next(reader)
        vs = next(reader)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="evaluate the performance of velocity extraction methods")
    parser.add_argument("--infile", type=Path, help="the CSV file to read from")

    arguments = parser.parse_args()
    infile = arguments.infile
    if infile == None:
        infile = sys.stdin

    run(infile)
