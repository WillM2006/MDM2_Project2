#!/bin/env python

import general
import numpy as np
from scipy.optimize import least_squares


def place_centres(max_spacing=2, domain_min=(-3, -3), domain_max=(3, 3), buffer=True):
    """
    Place RBF centres on a uniform grid across the domain.
    Optionally, include a buffer layer of centres on all 4 sides
    """
    x_min = domain_min[0]
    x_max = domain_max[0]
    y_min = domain_min[1]
    y_max = domain_max[1]

    # number of centres per axis
    n_x = int(np.ceil((x_max - x_min) / max_spacing) + 1)
    n_y = int(np.ceil((y_max - y_min) / max_spacing) + 1)

    spacing_x = (x_max - x_min) / (n_x - 1)
    spacing_y = (y_max - y_min) / (n_y - 1)

    xs = x_min + spacing_x * np.arange(n_x)
    ys = y_min + spacing_y * np.arange(n_y)

    if buffer:
        x_0 = [xs[0] - spacing_x]
        x_f = [xs[-1] + spacing_x]
        xs = np.concatenate((x_0, xs, x_f))

        y_0 = [ys[0] - 2 * spacing_y]
        y_f = [ys[-1] + spacing_y]
        ys = np.concatenate((y_0, ys, y_f))

    X, Y = np.meshgrid(xs, ys)
    centres = np.column_stack((X.ravel(), Y.ravel()))
    return centres


def build_phi(particle_positions, centres, sigmas):
    num = np.sum((particle_positions[:, None, :] - centres[None, :, :]) ** 2, axis=-1)
    denom = 2 * sigmas[None, :] ** 2
    phi = np.exp(-num / denom)
    return phi


"""
def best_widths_and_weights(particle_positions, velocities, centres):
    N = len(centres)
    params0 = np.concatenate([0.5*np.ones(N), np.ones(N)])  # [sigma, b]

    def get_residuals(target_velocity):
        def residuals(params):
            sigma = params[0:N]
            b     = params[N:2*N]
            return build_phi(particle_positions, centres, sigma) @ b - target_velocity
        return residuals

    result_u = least_squares(get_residuals(velocities[:, 0]), params0)
    result_w = least_squares(get_residuals(velocities[:, 1]), params0)

    sigma_u, b_u = result_u.x[0:N], result_u.x[N:2*N]
    sigma_w, b_w = result_w.x[0:N], result_w.x[N:2*N]

    best_widths  = np.vstack([sigma_u, sigma_w])
    best_weights = np.vstack([b_u, b_w])

    return best_widths, best_weights
"""


def best_widths_and_weights(particle_positions, velocities, centres):
    N = len(centres)
    params0 = np.concatenate([0.5 * np.ones(N), np.zeros(N)])

    # precompute squared distances once — same for u and w
    diff = particle_positions[:, None, :] - centres[None, :, :]
    r2 = np.sum(diff**2, axis=-1)  # (M, N)

    def make_fns(target_velocity):
        def residuals(params):
            sigma = params[0:N]
            b = params[N : 2 * N]
            phi = np.exp(-r2 / (2 * sigma[None, :] ** 2))
            return phi @ b - target_velocity

        def jacobian(params):
            sigma = params[0:N]
            b = params[N : 2 * N]
            phi = np.exp(-r2 / (2 * sigma[None, :] ** 2))
            dr_dsigma = b[None, :] * phi * r2 / sigma[None, :] ** 3
            dr_db = phi
            return np.hstack([dr_dsigma, dr_db])  # (M, 2N)

        return residuals, jacobian

    res_u, jac_u = make_fns(velocities[:, 0])
    res_w, jac_w = make_fns(velocities[:, 1])

    result_u = least_squares(res_u, params0, jac=jac_u, ftol=1e-4, xtol=1e-4)
    result_w = least_squares(res_w, params0, jac=jac_w, ftol=1e-4, xtol=1e-4)

    sigma_u, b_u = result_u.x[0:N], result_u.x[N : 2 * N]
    sigma_w, b_w = result_w.x[0:N], result_w.x[N : 2 * N]

    best_widths = np.vstack([sigma_u, sigma_w])
    best_weights = np.vstack([b_u, b_w])

    return best_widths, best_weights


def evaluate(eval_positions, centres, widths, weights):
    phi_u = build_phi(eval_positions, centres, widths[0])
    phi_w = build_phi(eval_positions, centres, widths[1])

    u = phi_u @ weights[0]
    w = phi_w @ weights[1]

    velocity = np.column_stack([u, w])

    return velocity


def mass_score(positions, centres, widths, weights):
    sigma_u = widths[0]
    b_u = weights[0]
    sigma_w = widths[1]
    b_w = weights[1]

    x_minus_mu = positions[:, None, :] - centres[None, :, :]

    phi_u = build_phi(positions, centres, sigma_u)
    phi_w = build_phi(positions, centres, sigma_w)

    du_dx = np.sum(b_u * (-x_minus_mu[:, :, 0]) / sigma_u**2 * phi_u, axis=1)
    dw_dz = np.sum(b_w * (-x_minus_mu[:, :, 1]) / sigma_w**2 * phi_w, axis=1)

    mass_score = np.abs(du_dx + dw_dz)

    return mass_score


def momentum_score(positions, centres, widths, weights, nu):
    sigma_u = widths[0]
    sigma_w = widths[1]
    b_u = weights[0]
    b_w = weights[1]

    x_minus_u = positions[:, None, :] - centres[None, :, :]

    phi_u = build_phi(positions, centres, sigma_u)  # (M, N)
    phi_w = build_phi(positions, centres, sigma_w)

    dx = x_minus_u[:, :, 0]
    dz = x_minus_u[:, :, 1]

    su = sigma_u[None, :] ** 2  # (1, N)
    sw = sigma_w[None, :] ** 2

    u = np.sum(b_u * phi_u, axis=1)
    w = np.sum(b_w * phi_w, axis=1)

    d2u_dxdz = np.sum(b_u * (dx * dz / su**2) * phi_u, axis=1)
    d2u_dz2 = np.sum(b_u * (dz**2 / su**2 - 1 / su) * phi_u, axis=1)
    d2w_dx2 = np.sum(b_w * (dx**2 / sw**2 - 1 / sw) * phi_w, axis=1)
    d2w_dxdz = np.sum(b_w * (dx * dz / sw**2) * phi_w, axis=1)

    d3u_dx2dz = np.sum(b_u * (dz / su**2 - dx**2 * dz / su**3) * phi_u, axis=1)
    d3u_dz3 = np.sum(b_u * (3 * dz / su**2 - dz**3 / su**3) * phi_u, axis=1)
    d3w_dx3 = np.sum(b_w * (3 * dx / sw**2 - dx**3 / sw**3) * phi_w, axis=1)
    d3w_dxdz2 = np.sum(b_w * (dx / sw**2 - dx * dz**2 / sw**3) * phi_w, axis=1)

    domega_dx = d2w_dx2 - d2u_dxdz
    domega_dz = d2w_dxdz - d2u_dz2

    nabla_sq_omega = (d3w_dx3 - d3u_dx2dz) + (d3w_dxdz2 - d3u_dz3)

    residual = u * domega_dx + w * domega_dz - nu * nabla_sq_omega

    momentum_score = np.abs(residual)

    return momentum_score


def combined_score(
    mass_score=None, momentum_score=None, energy_score=None, weights=None
):
    """
    Combined score is weighted sum of scores
    Each score is normalised - i.e. each weight = 1/initial score
    Then the combined score is then averaged - so thart combined score has same meaning even if energy score is not available for example
    """

    available_scores = {}
    if mass_score is not None:
        available_scores["mass"] = mass_score
    if momentum_score is not None:
        available_scores["momentum"] = momentum_score
    if energy_score is not None:
        available_scores["energy"] = energy_score

    # first iteration: weights = 1 / initial score, so each term starts at 1
    if weights is None:
        weights = {}
        for i in available_scores:
            weights[i] = 1.0 / np.mean(available_scores[i])

    comb_score = np.zeros_like(list(available_scores.values())[0])
    for i in available_scores:
        comb_score += weights[i] * available_scores[i]

    n = len(available_scores)
    avg_combined_score = comb_score / n

    return avg_combined_score, weights


def evaluate_assignments() -> np.ndarray:
    pass


def interpolate_velocities() -> np.ndarray:
    pass


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description="Velocity interpolation (Delaunay)")
    parser.add_argument(
        "--infile", type=Path, required=True, help="path to read input data from"
    )
    parser.add_argument(
        "--outfile", type=Path, required=True, help="path to write output data to"
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="directory to save visualisations to",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="initial position assignment parameter",
    )
    parser.add_argument(
        "--extent", type=float, required=True, help="size of the domain"
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        required=True,
        help="number of samples to predict velocities at",
    )
    parser.add_argument(
        "--edgepoints",
        type=int,
        required=True,
        help="number of points along each boundary edge",
    )

    arguments = parser.parse_args()
    infile = open(arguments.infile, "r")
    outfile = open(arguments.outfile, "w")

    general.run(
        infile,
        outfile,
        arguments.epsilon,
        arguments.extent,
        arguments.sample_count,
        arguments.edgepoints,
        arguments.frames_dir,
        evaluate_assignments,
        interpolate_velocities,
    )
