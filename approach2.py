# approach2
# Dual-mesh (Voronoi) approach for particle-velocity inference.

import numpy as np
import scipy.spatial
import scipy.optimize
import scipy.interpolate
from typing import Optional, Tuple
 

# Constants
DIMENSIONS = 2
K_NEIGHBOURS = 6          # number of neighbours used for the energy score
ENERGY_EPSILON = 1e-8     # avoiding divide-by-zero in energy score


# Boundary / corner points
def generate_corner_points(extent: float, edgepoints: int) -> np.ndarray:
    # Generate fixed no-slip points around the domain boundary
    # These points carry zero velocity and anchor the triangulation
    # at the edges of the domain, preventing triangles there
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
    # Parse one CSV row of flattened (x, z) coordinates into an (N, 2) array
    # and prepend the fixed boundary corner points
    pts = np.array(line, np.float32).reshape(DIMENSIONS, -1).T
    return np.concatenate([corners, pts], axis=0)



# Affinity matrix and assignment (Hungarian algorithm)
def initial_affinity_matrix(old: np.ndarray, new: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Build a Gaussian affinity matrix between two frames of particle positions
    A[i,j] = exp(-||old_i - new_j||^2 / eps^2)
    Large when particles are close, small when far apart
    """
    # Using broadcasting to compute all pairwise squared distances at once
    diff = old[:, np.newaxis, :] - new[np.newaxis, :, :]   # (N_old, N_new, 2)
    sq_dist = np.sum(diff ** 2, axis=-1)                    # (N_old, N_new)
    return np.exp(-sq_dist / epsilon ** 2)

def assign_from_affinity(affinity: np.ndarray) -> np.ndarray:
    """
    Solving the linear sum assignment problem (Hungarian algorithm) on the
    affinity matrix to find the optimal one-to-one particle matching.
    Returns `assignment` where assignment[i] = j means old particle i
    is matched to new particle j.
    """
    cost = -affinity          # maximise affinity  ≡  minimise negative affinity
    _, columns = scipy.optimize.linear_sum_assignment(cost)
    return columns



# Velocities
def compute_velocities(old: np.ndarray, new: np.ndarray, 
                       assignment: np.ndarray) -> np.ndarray:
    # computing per particle velocities from a frame pair and an assignment vector
    # v[j] = new[j] - old[i]   where assignment[i] = j
    # Returns an (N, 2) array of velocity vectors indexed in the new frame
    
    N = new.shape[0]
    velocities = np.zeros((N, 2), dtype=np.float64)
    for i, j in enumerate(assignment):
        velocities[j, :] = new[j, :] - old[i, :]
    return velocities



# Delaunay triangulation and Voronoi dual areas
def triangulate(positions: np.ndarray) -> scipy.spatial.Delaunay:
    return scipy.spatial.Delaunay(positions)
# Building a Delaunay triangulation of the given particle positions

def voronoi_areas(positions: np.ndarray,
                  tri: scipy.spatial.Delaunay) -> np.ndarray:
    """
    Computing the approximate Voronoi cell area for each point by using
    the dual of the Delaunay triangulation

    Each triangle contributes one third of its area to each of its
    three vertices. This is the difference from Approach 1, each particle
    now owns a region of space, giving a physically meaningful cell mass
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



# Helper geometry (defining them before the score functions that call them)
def barycentric_gradient(vertices: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Returning the 3 barycentric gradient vectors ∇λ_k  (each is a 2-vector)
    and the signed area of the triangle, used to compute ∇·u analytically
    on the piecewise-linear Delaunay mesh. 
    """
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    det = (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    if abs(det) < 1e-15:
        return np.zeros((3, 2)), 0.0

    inv_det = 1.0 / det

    grad = np.array([
        [(y2 - y3) * inv_det,  (x3 - x2) * inv_det],
        [(y3 - y1) * inv_det,  (x1 - x3) * inv_det],
        [(y1 - y2) * inv_det,  (x2 - x1) * inv_det],
    ])
    return grad, det / 2.0


def circumcentre(verts: np.ndarray) -> Optional[np.ndarray]:
    # Circumcentre of a triangle, Returns None if degenerate
    ax, ay = verts[0]
    bx, by = verts[1]
    cx, cy = verts[2]
    D = 2.0 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(D) < 1e-15:
        return None
    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / D
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / D
    return np.array([ux, uy])
 


# Conservation scores (all per-particle)
def score_mass(positions: np.ndarray, velocities: np.ndarray,
               tri: scipy.spatial.Delaunay,
               voronoi_areas_arr: np.ndarray) -> np.ndarray:
    """
    Per-particle mass-conservation score:
        S_mass_p = |∇·u|_p

    ∇·u is constant within each Delaunay triangle (piecewise-linear field).
    Accumulating an area-weighted average over all triangles sharing vertex p.

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
    # tri.neighbors[t, k] = neighbour triangle opposite vertex k of triangle t
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



# Affinity correction and combined score
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


def combined_score(s_mass=None, s_mom=None, s_energy=None, weights=None):
    """
    Weighted combination of per-particle conservation scores, matching the
    interface of Approach 3's combined_score for easy cross-comparison.
 
    On the first call (weights=None) each score is normalised by its own
    mean so all three terms start contributing equally. The normalisation
    weights are returned so they can be passed back on subsequent calls.
    """
    available = {}
    if s_mass   is not None: available['mass']     = s_mass
    if s_mom    is not None: available['momentum'] = s_mom
    if s_energy is not None: available['energy']   = s_energy
 
    if weights is None:
        weights = {k: 1.0 / (np.mean(v) + 1e-12) for k, v in available.items()}
 
    score = np.zeros_like(next(iter(available.values())))
    for k, v in available.items():
        score += weights[k] * v
 
    return score / len(available), weights
 
 

# Velocity interpolation onto a regular grid 
def interpolate_velocity_field(positions: np.ndarray,
                               velocities: np.ndarray,
                               gridpoints: np.ndarray) -> np.ndarray:
    """
    Interpolate per-particle velocities onto a regular grid using a
    piecewise-linear (barycentric) interpolant.
    Returns an (G, G, 2) array of velocity vectors.
    """
    G  = len(gridpoints)
    xx, yy    = np.meshgrid(gridpoints, gridpoints)
    pts_grid  = np.column_stack([xx.ravel(), yy.ravel()])
 
    interp_u = scipy.interpolate.LinearNDInterpolator(positions, velocities[:, 0])
    interp_v = scipy.interpolate.LinearNDInterpolator(positions, velocities[:, 1])
 
    result = np.zeros((G, G, 2))
    result[:, :, 0] = interp_u(pts_grid).reshape(G, G)
    result[:, :, 1] = interp_v(pts_grid).reshape(G, G)
    return result