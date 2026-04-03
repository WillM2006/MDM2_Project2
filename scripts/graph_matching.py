#!/bin/env python

import numpy as np
from numba import jit


class SpectralGraphMatchingInfo:
    def __init__(self, old: np.ndarray, new: np.ndarray, epsilon: float):
        (n1, two) = old.shape
        assert two == 2
        self.n1 = n1

        (n2, two) = old.shape
        assert two == 2
        self.n2 = n2

        self.affinity = _affinity_matrix(old, new, epsilon)

        self.p = np.repeat(1 / n2, n1 * n1)
        self.p_new = self.p.copy()


def iterate(data: SpectralGraphMatchingInfo) -> float:
    """Perform a single iteration of spectral graph matching, mutating structures in-place.
    Returns the cost the current configuration."""

    data.p_new = data.affinity @ data.p
    data.p_new = _normalize(data.p_new)

    p_frac = data.p_new / data.p
    for j in range(data.affinity.shape[1]):
        data.affinity[:, j] = data.affinity[:, j] * p_frac

    data.p, data.p_new = data.p_new, data.p

    cost = np.sqrt(np.sum(np.square(data.p_new - data.p))) / data.p.shape[0]

    return cost


def solution(data: SpectralGraphMatchingInfo) -> np.ndarray:
    """Generate and return a dicretized assignment matrix."""

    return _solution(data.p, data.n1, data.n2)


@jit(nopython=True, parallel=True)
def _solution(p: np.ndarray, n1: int, n2: int) -> np.ndarray:
    p = p.reshape((n1, n2))

    discrete = np.zeros(n1, dtype=np.int32)
    for i in range(n1):
        discrete[i] = np.argmax(p[i, :])

    return discrete


@jit(nopython=True, parallel=True)
def _affinity_matrix(old: np.ndarray, new: np.ndarray, epsilon: float) -> np.ndarray:
    (n_old, _) = old.shape
    (n_new, _) = new.shape

    n = n_new * n_old

    affinities = np.zeros((n, n))

    for i in range(n_old):
        for iprime in range(n_new):
            for j in range(n_old):
                for jprime in range(n_new):
                    d_old_x = old[i, 0] - old[j, 0]
                    d_old_y = old[i, 1] - old[j, 1]
                    d_new_x = new[iprime, 0] - new[jprime, 0]
                    d_new_y = new[iprime, 1] - new[jprime, 1]

                    d_old = np.sqrt(d_old_x**2 + d_old_y**2)
                    d_new = np.sqrt(d_new_x**2 + d_new_y**2)

                    affinity = d_old - d_new
                    ai = i * n_new + iprime
                    aj = j * n_new + jprime
                    affinities[ai, aj] = affinity

    return np.exp(-np.square(affinities / epsilon))


@jit(nopython=True, parallel=True)
def _normalize(p: np.ndarray):
    return p / np.sqrt(np.sum(np.square(p)))
