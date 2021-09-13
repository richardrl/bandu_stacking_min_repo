"""
From https://github.com/igilitschenski/deep_bingham
Generates the lookup table for the Bingham normalization constant.
"""
from __future__ import print_function

import numpy as np
import time
import utils_bingham


def generate_bd_lookup_table():
    coords = np.linspace(-50, 0, 4)
    duration = time.time()
    utils_bingham.build_bd_lookup_table(
        "uniform", {"coords": coords, "bounds": (-50, 0), "num_points": 4},
        "precomputed/lookup_-50_0_4.dill")

    duration = time.time() - duration

    print('lookup table function took %0.3f ms' % (duration * 1000.0))


if __name__ == "__main__":
    generate_bd_lookup_table()