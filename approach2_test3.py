# approach2_test3
# !/bin/env python

"""
python generate_example_data.py --count 50 --frames 60 --extent 6 --seed 12345 --dissipation 0.5 \
  | python approach2_test3.py --extent 6.5 --edgepoints 7

Conservation score formulas
All three scores are evaluated *per particle* so they can be mapped back
directly onto the (n_particles × n_particles) affinity matrix.

Mass (incompressibility):
    S_mass_p  = |∇·u(x_p)|
    where u is the piecewise-linear velocity field on the Delaunay mesh.
    For triangle T with barycentric coordinates, the divergence is constant:
        ∇·u = Σ_k  u_k · ∇λ_k
    and is evaluated at each particle from the triangles that share it (area-weighted mean).

Momentum (cell-face flux):
    For each Voronoi cell V_p (dual of particle p):
        S_mom_p = |Σ_{faces} (u_face · n̂) * |face|| / |V_p|
    This is the net flux of momentum (i.e. velocity) through the cell boundary.
    For an incompressible flow this should be zero.

Energy (local KE smoothness):
    S_energy_p = |v_p - v̄_p|² / (|v̄_p|² + δ)
    where v̄_p is the area-weighted mean velocity of the k nearest neighbours.
"""

import csv
import sys
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.spatial
import scipy.optimize
import scipy.interpolate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Constants
DIMENSIONS = 2
K_NEIGHBOURS = 6          # neighbours used for the energy score
ENERGY_EPSILON = 1e-8     # avoid divide-by-zero in energy score


# Boundary / corner points
def generate_corner_points(extent: float, edgepoints: int) -> np.ndarray:
    """Generate fixed no-slip points around the domain boundary."""
    pts = np.linspace(-extent / 2, extent / 2, num=edgepoints + 1, endpoint=False)
    pts_flip = pts[::-1]
    lo = np.full(edgepoints + 1, -extent / 2)
    hi = np.full(edgepoints + 1,  extent / 2)

    top    = np.column_stack((pts,      hi))
    right  = np.column_stack((hi,       pts_flip))
    bottom = np.column_stack((pts_flip, lo))
    left   = np.column_stack((lo,       pts))

    return np.concatenate((bottom, top, left, right))


def read_points(line, corners: np.ndarray) -> np.ndarray:
    """Parse one CSV line into (N, 2) positions, prepending the fixed corners."""
    pts = np.array(line, np.float32).reshape(DIMENSIONS, -1).T
    return np.concatenate([corners, pts], axis=0)


# Affinity matrix
def initial_affinity_matrix(old: np.ndarray, new: np.ndarray,
                             epsilon: float) -> np.ndarray:
    """
    A[i,j] = exp(-||old_i - new_j||^2 / eps^2)

    This is a Gaussian kernel: large when the two points are close,
    small when they are far apart.
    """
    # Use broadcasting to compute all pairwise squared distances at once.
    diff = old[:, np.newaxis, :] - new[np.newaxis, :, :]   # (N_old, N_new, 2)
    sq_dist = np.sum(diff ** 2, axis=-1)                    # (N_old, N_new)
    return np.exp(-sq_dist / epsilon ** 2)


def assign_from_affinity(affinity: np.ndarray) -> np.ndarray:
    """
    Convert an affinity (similarity) matrix to a cost matrix and solve
    the linear sum assignment (Hungarian algorithm).

    Returns `columns`: columns[i] = j means particle i is matched to j.
    """
    cost = -affinity          # maximise affinity  ≡  minimise negative affinity
    _, columns = scipy.optimize.linear_sum_assignment(cost)
    return columns


# Velocity from assignment
def compute_velocities(old: np.ndarray, new: np.ndarray,
                       assignment: np.ndarray) -> np.ndarray:
    """
    v[j] = new[j] - old[i]   where assignment[i] = j.

    Returns an (N, 2) array of velocity vectors indexed in the *new* frame.
    """
    N = new.shape[0]
    velocities = np.zeros((N, 2), dtype=np.float64)
    for i, j in enumerate(assignment):
        velocities[j, :] = new[j, :] - old[i, :]
    return velocities


# Delaunay triangulation + Voronoi dual
def triangulate(positions: np.ndarray) -> scipy.spatial.Delaunay:
    return scipy.spatial.Delaunay(positions)


def voronoi_areas(positions: np.ndarray,
                  tri: scipy.spatial.Delaunay) -> np.ndarray:
    """
    Compute the approximate Voronoi cell area for each point via the
    dual of the Delaunay triangulation.

    Strategy: each triangle contributes 1/3 of its area to each of its
    three vertices (a robust approximation that works for all triangulations).
    """
    N = positions.shape[0]
    areas = np.zeros(N, dtype=np.float64)
    for simplex in tri.simplices:
        i, j, k = simplex
        a, b, c = positions[i], positions[j], positions[k]
        # Signed area via cross product (take absolute value)
        triangle_area = abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])) * 0.5
        areas[i] += triangle_area / 3.0
        areas[j] += triangle_area / 3.0
        areas[k] += triangle_area / 3.0
    # Guard against degenerate zero-area cells
    areas = np.where(areas < 1e-15, 1e-15, areas)
    return areas


# Conservation scores (all per-particle)
def barycentric_gradient(vertices: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Return the 3 barycentric gradient vectors ∇λ_k  (each is a 2-vector)
    and the signed area of the triangle.

    Given triangle vertices A, B, C (each a 2-vector),
    the gradients are the rows of A^{-1} (the last row dropped since
    λ3 = 1 - λ1 - λ2, but we return all three for convenience).
    """
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    det = (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    if abs(det) < 1e-15:
        return np.zeros((3, 2)), 0.0

    area = det / 2.0
    inv_det = 1.0 / det

    grad = np.array([
        [(y2 - y3) * inv_det,  (x3 - x2) * inv_det],
        [(y3 - y1) * inv_det,  (x1 - x3) * inv_det],
        [(y1 - y2) * inv_det,  (x2 - x1) * inv_det],
    ])
    return grad, area


def score_mass(positions: np.ndarray, velocities: np.ndarray,
               tri: scipy.spatial.Delaunay,
               voronoi_areas_arr: np.ndarray) -> np.ndarray:
    """
    Per-particle mass-conservation score:
        S_mass_p = |∇·u|_p

    ∇·u is constant within each Delaunay triangle (piecewise-linear field).
    We accumulate an area-weighted average over all triangles sharing vertex p.

    For incompressible flow: ∇·u = 0 everywhere → score should be near 0.
    """
    N = positions.shape[0]
    divergence_sum = np.zeros(N, dtype=np.float64)
    weight_sum     = np.zeros(N, dtype=np.float64)

    for simplex in tri.simplices:
        i, j, k = simplex
        verts = positions[np.array([i, j, k])]
        grads, area = barycentric_gradient(verts)
        if abs(area) < 1e-15:
            continue

        # ∇·u = Σ_m  u_m · ∇λ_m     (dot product per vertex, summed)
        div = (np.dot(velocities[i], grads[0]) +
               np.dot(velocities[j], grads[1]) +
               np.dot(velocities[k], grads[2]))

        w = abs(area)
        for idx in (i, j, k):
            divergence_sum[idx] += div * w
            weight_sum[idx]     += w

    # Avoid zero-weight nodes (isolated boundary points)
    mask = weight_sum > 1e-15
    score = np.zeros(N)
    score[mask] = np.abs(divergence_sum[mask] / weight_sum[mask])
    return score


def score_momentum(positions: np.ndarray, velocities: np.ndarray,
                   tri: scipy.spatial.Delaunay,
                   voronoi_areas_arr: np.ndarray) -> np.ndarray:
    """
    Per-particle momentum-conservation score (net flux through Voronoi cell):
        S_mom_p = |Σ_{faces} (u_face · n̂) * |face|| / |V_p|

    Dual-mesh interpretation:
      Each Delaunay edge (shared by two triangles) corresponds to a Voronoi
      face.  The Voronoi face runs between the circumcentres of the two
      triangles.  The velocity on the shared edge is approximated as the
      average of the two endpoint velocities.

    For incompressible flow (∇·u = 0) this flux integral should vanish
    (Gauss theorem).  Non-zero values flag physically implausible assignments.
    """
    N = positions.shape[0]
    flux_accum = np.zeros(N, dtype=np.float64)

    # Build edge → triangle adjacency from the Delaunay structure
    # tri.neighbors[t, k] = neighbour triangle opposite vertex k of triangle t.
    simplices  = tri.simplices        # (T, 3)
    neighbors  = tri.neighbors        # (T, 3)  -1 if no neighbour

    for t_idx, simplex in enumerate(simplices):
        verts_t = positions[simplex]
        cc_t    = circumcentre(verts_t)
        if cc_t is None:
            continue

        for local_k in range(3):
            t_nbr = neighbors[t_idx, local_k]
            if t_nbr == -1 or t_nbr < t_idx:
                continue   # boundary edge or already processed

            cc_nbr = circumcentre(positions[simplices[t_nbr]])
            if cc_nbr is None:
                continue

            # Voronoi edge vector and its length
            voronoi_edge = cc_nbr - cc_t
            face_len     = np.linalg.norm(voronoi_edge)
            if face_len < 1e-15:
                continue

            # Normal to the Voronoi edge (perpendicular, pointing "outward"
            # relative to the Delaunay edge between the two shared vertices).
            normal = np.array([-voronoi_edge[1], voronoi_edge[0]]) / face_len

            # Shared Delaunay edge: vertices are those NOT at local position k
            shared = [simplex[m] for m in range(3) if m != local_k]
            p, q   = shared[0], shared[1]
            u_face = 0.5 * (velocities[p] + velocities[q])
            flux   = np.dot(u_face, normal) * face_len

            # Accumulate into both adjacent particles
            flux_accum[p] += abs(flux)
            flux_accum[q] += abs(flux)

    # Normalise by Voronoi cell area
    score = flux_accum / voronoi_areas_arr
    return score


def circumcentre(verts: np.ndarray) -> Optional[np.ndarray]:
    """Circumcentre of a triangle.  Returns None if degenerate."""
    ax, ay = verts[0]
    bx, by = verts[1]
    cx, cy = verts[2]
    D = 2.0 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(D) < 1e-15:
        return None
    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / D
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / D
    return np.array([ux, uy])


def score_energy(positions: np.ndarray, velocities: np.ndarray,
                 voronoi_areas_arr: np.ndarray,
                 k: int = K_NEIGHBOURS) -> np.ndarray:
    """
    Per-particle energy-conservation score:
        S_energy_p = |v_p - v̄_p|² / (|v̄_p|² + ε)

    v̄_p is the area-weighted mean velocity of the k nearest neighbours
    of particle p.  A particle whose velocity departs significantly from
    its neighbours is likely mis-matched.
    """
    N = positions.shape[0]
    k_actual = min(k, N - 1)
    if k_actual < 1:
        return np.zeros(N)

    tree = scipy.spatial.KDTree(positions)
    _, idx = tree.query(positions, k=k_actual + 1)   # includes self at index 0

    score = np.zeros(N, dtype=np.float64)
    for p in range(N):
        nbrs     = idx[p, 1:]                          # exclude self
        weights  = voronoi_areas_arr[nbrs]
        w_sum    = weights.sum()
        if w_sum < 1e-15:
            continue
        v_mean   = (velocities[nbrs] * weights[:, np.newaxis]).sum(axis=0) / w_sum
        v_diff   = velocities[p] - v_mean
        score[p] = np.dot(v_diff, v_diff) / (np.dot(v_mean, v_mean) + ENERGY_EPSILON)

    return score


# Score → affinity correction
def normalise_score(score: np.ndarray) -> np.ndarray:
    """Normalise a per-particle score vector to [0, 1]."""
    s_max = score.max()
    if s_max < 1e-15:
        return score
    return score / s_max


def affinity_correction(affinity: np.ndarray,
                        score_mass_arr: np.ndarray,
                        score_mom_arr: np.ndarray,
                        score_energy_arr: np.ndarray,
                        w_mass: float,
                        w_mom: float,
                        w_energy: float,
                        correction_strength: float = 0.1) -> np.ndarray:
    """
    Reduce the affinity of particle assignments that violate conservation laws.

    For each row i (old particle), the plausibility of matching it to column j
    (new particle) is penalised by the conservation scores at particle j.

    The total per-particle penalty is:
        penalty_j = w_mass * S_mass_j + w_mom * S_mom_j + w_energy * S_energy_j

    The affinity correction is:
        A[i, j]  →  A[i, j] * exp(-correction_strength * penalty_j)

    Using a multiplicative (soft) correction avoids driving any affinity
    to exactly zero and keeps the matrix well-conditioned.
    """
    # Normalise each score independently to [0, 1]
    nm = normalise_score(score_mass_arr)
    nmo = normalise_score(score_mom_arr)
    ne = normalise_score(score_energy_arr)

    penalty = w_mass * nm + w_mom * nmo + w_energy * ne  # shape (N_new,)

    # Broadcast: apply column-wise penalty (each column = a new-frame particle)
    correction = np.exp(-correction_strength * penalty[np.newaxis, :])
    return affinity * correction


# Barycentric velocity interpolation
def interpolate_velocity_field(positions: np.ndarray,
                               velocities: np.ndarray,
                               tri: scipy.spatial.Delaunay,
                               gridpoints: np.ndarray) -> np.ndarray:
    """
    Interpolate per-particle velocities onto a regular grid using the
    piecewise-linear (barycentric) interpolant defined by the Delaunay mesh.

    Returns an (G, G, 2) array.
    """
    G = len(gridpoints)
    interpolated = np.zeros((G, G, 2))

    interp_u = scipy.interpolate.LinearNDInterpolator(positions, velocities[:, 0])
    interp_v = scipy.interpolate.LinearNDInterpolator(positions, velocities[:, 1])

    xx, yy = np.meshgrid(gridpoints, gridpoints)
    pts_grid = np.column_stack([xx.ravel(), yy.ravel()])
    u_vals = interp_u(pts_grid).reshape(G, G)
    v_vals = interp_v(pts_grid).reshape(G, G)

    interpolated[:, :, 0] = u_vals
    interpolated[:, :, 1] = v_vals
    return interpolated


# Plotting
def plot_frame(name, positions, tri, velocities, gridpoints,
               interp_vel, voronoi_areas_arr, score, frames_dir: Path):
    LOOP = [0, 1, 2, 0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: mesh + velocities ---
    ax = axes[0]
    ax.set_title("Delaunay mesh + velocity field")
    for simplex in tri.simplices:
        idx = simplex[LOOP]
        ax.plot(positions[idx, 0], positions[idx, 1], c="lightgrey", lw=0.5, zorder=0)
    ax.scatter(positions[:, 0], positions[:, 1], s=12, zorder=2)
    ax.quiver(positions[:, 0], positions[:, 1],
              velocities[:, 0], velocities[:, 1],
              color="steelblue", scale=1.0, width=0.003, zorder=1)
    # Grid velocities
    ax.quiver(gridpoints, gridpoints,
              interp_vel[:, :, 0], interp_vel[:, :, 1],
              color="tomato", alpha=0.5, scale=4.0, width=0.002, zorder=-1)
    ax.set_aspect("equal")

    # --- Right: conservation score heatmap ---
    ax2 = axes[1]
    ax2.set_title("Conservation score (per particle)")
    sc = ax2.scatter(positions[:, 0], positions[:, 1],
                     c=score, cmap="hot_r", s=20, zorder=2)
    plt.colorbar(sc, ax=ax2, label="score")
    ax2.set_aspect("equal")

    frames_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(frames_dir / f"{name}.svg")
    plt.close(fig)


# Main processing loop
def run(infile, frames_dir: Optional[Path],
        extent: float, edgepoints: int,
        epsilon: float = 1.0,
        correction_strength: float = 0.15,
        max_iter: int = 20,
        convergence_tol: float = 1e-4,
        gridsize: int = 40):
    """
    Full pipeline: read frames, match particles, score conservation laws,
    refine affinity matrix, repeat, then interpolate velocity field.
    """
    corners = generate_corner_points(extent, edgepoints)
    csv_reader = csv.reader(infile)

    # Prime the iteration with the first frame
    prev_positions = read_points(next(csv_reader), corners)

    for frame_index, line in enumerate(csv_reader, start=1):
        curr_positions = read_points(line, corners)
        N_old = prev_positions.shape[0]
        N_new = curr_positions.shape[0]

        print(f"[Frame {frame_index}]  {N_old} → {N_new} particles", file=sys.stderr)

        # ---- Initial affinity matrix ----------------------------------------
        affinity = initial_affinity_matrix(prev_positions, curr_positions, epsilon)

        # ---- Triangulation on current frame (fixed throughout refinement) ----
        tri = triangulate(curr_positions)
        voronoi_areas_arr = voronoi_areas(curr_positions, tri)

        # ---- Determine initial weights using first-iteration scores ----------
        assignment    = assign_from_affinity(affinity)
        velocities    = compute_velocities(prev_positions, curr_positions, assignment)

        s_mass   = score_mass(curr_positions, velocities, tri, voronoi_areas_arr)
        s_mom    = score_momentum(curr_positions, velocities, tri, voronoi_areas_arr)
        s_energy = score_energy(curr_positions, velocities, voronoi_areas_arr)

        # Weights are reciprocals of initial scores (so all start equal at 1)
        # (guard against zero initial scores)
        w_mass   = 1.0 / (s_mass.mean()   + 1e-12)
        w_mom    = 1.0 / (s_mom.mean()    + 1e-12)
        w_energy = 1.0 / (s_energy.mean() + 1e-12)

        # ---- Iterative refinement -------------------------------------------
        prev_assignment = None
        for iteration in range(max_iter):
            # Apply conservation-law corrections to affinity
            affinity_corrected = affinity_correction(
                affinity, s_mass, s_mom, s_energy,
                w_mass, w_mom, w_energy,
                correction_strength=correction_strength
            )

            # Solve assignment
            new_assignment = assign_from_affinity(affinity_corrected)

            # Check convergence: did the assignment change?
            if prev_assignment is not None:
                n_changed = np.sum(new_assignment != prev_assignment)
                frac_changed = n_changed / len(new_assignment)
                print(f"  iter {iteration+1:3d}: {n_changed} assignments changed "
                      f"({100*frac_changed:.1f}%)", file=sys.stderr)
                if frac_changed <= convergence_tol:
                    print(f"  Converged after {iteration+1} iterations.", file=sys.stderr)
                    break
            prev_assignment = new_assignment

            # Recompute velocities and scores from corrected assignment
            assignment = new_assignment
            velocities = compute_velocities(prev_positions, curr_positions, assignment)
            s_mass   = score_mass(curr_positions, velocities, tri, voronoi_areas_arr)
            s_mom    = score_momentum(curr_positions, velocities, tri, voronoi_areas_arr)
            s_energy = score_energy(curr_positions, velocities, voronoi_areas_arr)

            # Update affinity matrix using corrections (in-place on the *original*
            # affinities so corrections accumulate multiplicatively)
            affinity = affinity_correction(
                affinity, s_mass, s_mom, s_energy,
                w_mass, w_mom, w_energy,
                correction_strength=correction_strength
            )

        # ---- Final velocities from converged assignment ---------------------
        velocities = compute_velocities(prev_positions, curr_positions, assignment)

        # ---- Combined plausibility score (lower = better) ------------------
        combined_score = (w_mass   * normalise_score(s_mass) +
                          w_mom    * normalise_score(s_mom) +
                          w_energy * normalise_score(s_energy))
        print(f"  Final combined score: {combined_score.mean():.5f} "
              f"(mean over all particles)", file=sys.stderr)

        # ---- Velocity interpolation onto grid -------------------------------
        gridpoints = np.linspace(-extent / 2, extent / 2, gridsize)
        interp_vel = interpolate_velocity_field(
            curr_positions, velocities, tri, gridpoints
        )

        # ---- Optional visualisation -----------------------------------------
        if frames_dir is not None:
            plot_frame(
                frame_index,
                curr_positions, tri, velocities,
                gridpoints, interp_vel,
                voronoi_areas_arr, combined_score,
                frames_dir,
            )

        # Advance frame
        prev_positions = curr_positions


# CLI entry point
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description=(
            "Method 2: Voronoi dual-mesh particle matching with conservation-law "
            "score feedback.  Reads CSV frames from stdin."
        )
    )
    parser.add_argument("--extent", required=True,
                        help="axis-aligned domain size", type=float)
    parser.add_argument("--edgepoints", required=True,
                        help="fixed no-slip points per boundary edge", type=int)
    parser.add_argument("--frames-dir", default=None,
                        help="directory to write SVG frame visualisations into",
                        type=Path)
    parser.add_argument("--epsilon", default=1.0,
                        help="Gaussian kernel width for initial affinity (default 1.0)",
                        type=float)
    parser.add_argument("--correction-strength", default=0.15,
                        help="how aggressively conservation scores correct affinities "
                             "(default 0.15)",
                        type=float)
    parser.add_argument("--max-iter", default=20,
                        help="maximum refinement iterations per frame (default 20)",
                        type=int)
    parser.add_argument("--gridsize", default=40,
                        help="resolution of interpolated velocity grid (default 40)",
                        type=int)

    args = parser.parse_args()

    run(
        sys.stdin,
        args.frames_dir,
        args.extent,
        args.edgepoints,
        epsilon=args.epsilon,
        correction_strength=args.correction_strength,
        max_iter=args.max_iter,
        gridsize=args.gridsize,
    )