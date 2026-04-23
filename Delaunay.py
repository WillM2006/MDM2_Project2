

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from shapely import points
from shapely.geometry import Polygon

from createPoints import generate_points
from createPoints import vector_field

# create points
width = (-5, 5)
height = (-5, 5)
num_points = 200
num_frames = 25
dt = 0.01
RESOLUTION = 50
df = generate_points(width, height, num_points, num_frames, dt=dt)

def get_points(df, frameidx):
    x = df.loc[frameidx, "x"]
    y = df.loc[frameidx, "y"]
    return np.column_stack((x, y))

def C_IJ(df, frame1idx, frame2idx):
    ''' Calculate the cross-correlation coefficient between all the triangles in 2 frames (higher values are better)'''

    def inRadius(point1, point2, R):
        r = np.sqrt((point1[0]-point2[0])**2
                    +(point1[1]-point2[1])**2)
        if r <= R:
            return True
        else:
            return False
        
    def translate_triangle(tri_pts, centroid1):
        # translate first
        A, B, C = tri_pts - centroid1

        def angle(p, q, r):
            v1 = q - p
            v2 = r - p
            return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        angles = [
            angle(A, B, C),
            angle(B, A, C),
            angle(C, A, B)
        ]

        # find vertex opposite largest angle
        idx = np.argmax(angles)
        large_vertex = tri_pts[idx]
        other_vertexs = [tri_pts[(idx+1)%3], tri_pts[(idx+2)%3]]

        v1 = other_vertexs[0] - large_vertex
        v2 = other_vertexs[1] - large_vertex

        v1n = v1 / np.linalg.norm(v1)
        v2n = v2 / np.linalg.norm(v2)

        direction = v1n + v2n

        theta = np.arctan2(direction[1], direction[0])

        R = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta),  np.cos(-theta)]
        ])

        translated = (tri_pts - large_vertex) @ R.T

        return translated


    def tri_area(tri_pts):
        a, b, c = tri_pts
        return 0.5 * abs(np.linalg.det([b - a, c - a]))

    def tri_overlap_area(tri_pts1, tri_pts2, centroid1, centroid2):
        translated_pts1 = translate_triangle(tri_pts1, centroid1)
        translated_pts2 = translate_triangle(tri_pts2, centroid2)
        poly1 = Polygon(translated_pts1)
        poly2 = Polygon(translated_pts2)
        return poly1.intersection(poly2).area

    points1 = get_points(df, frame1idx)
    points2 = get_points(df, frame2idx)

    if len(points1) < 3 or len(points2) < 3:
        raise ValueError("Not enough points for Delaunay triangulation")

    tri1 = Delaunay(points1)
    tri2 = Delaunay(points2)

    centroids1 = np.mean(tri1.points[tri1.simplices], axis=1)
    centroids2 = np.mean(tri2.points[tri2.simplices], axis=1)

    C = np.zeros((len(tri1.simplices), len(tri2.simplices)))

    for i, tri_i in enumerate(tri1.simplices):
        centroid_i = centroids1[i]

        for j, tri_j in enumerate(tri2.simplices):
            centroid_j = centroids2[j]

            if inRadius(centroid_i, centroid_j, R):
                tri_i_pts = tri1.points[tri_i]
                tri_j_pts = tri2.points[tri_j]
                C[i, j] = tri_overlap_area(tri_i_pts, tri_j_pts, centroid_i, centroid_j) / np.sqrt(tri_area(tri_i_pts) * tri_area(tri_j_pts))

    return C, tri1, tri2, centroids1, centroids2

def pair(C, centroids1, centroids2, max_dist=0.02):
    pairs = []

    candidates = []

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            score = C[i, j]
            if score <= 0:
                continue

            d = np.linalg.norm(centroids2[j] - centroids1[i])

            if max_dist is not None and d > max_dist:
                continue

            candidates.append((score, i, j))

    candidates.sort(reverse=True, key=lambda x: x[0])

    used_i = set()
    used_j = set()

    for score, i, j in candidates:
        if i in used_i or j in used_j:
            continue

        pairs.append((i, j))
        used_i.add(i)
        used_j.add(j)

    return pairs

def calc_velocity(df, frame1idx, frame2idx):
    C, tri1, tri2, centroids1, centroids2 = C_IJ(df, frame1idx, frame2idx)
    pairs = pair(C, centroids1, centroids2)

    velocities = []
    for i, j in pairs:
        dr = centroids2[j] - centroids1[i]
        v = dr / dt
        pos = centroids1[i]
        velocities.append((pos, v))
    return velocities

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

def interpolate_velocity_field(df, frame0idx, frame1idx):
    velocities = calc_velocity(df, frame0idx, frame1idx)
    
    points = np.array([p for p, v in velocities])
    values = np.array([v for p, v in velocities])

    if len(points) < 3:
        raise ValueError("Not enough points for interpolation")


    interp_u = LinearNDInterpolator(points, values[:, 0])
    interp_v = LinearNDInterpolator(points, values[:, 1])


    fallback_u = NearestNDInterpolator(points, values[:, 0])
    fallback_v = NearestNDInterpolator(points, values[:, 1])

    def velocity(x, y):
        u = interp_u(x, y)
        v = interp_v(x, y)
        
        if np.isnan(u) or np.isnan(v):
            u = fallback_u(x, y)
            v = fallback_v(x, y)

        return np.array([u, v])

    return velocity, points



class Main:

    def __init__(self, df, frame0idx, frame1idx):
        self.df = df
        self.frame0 = frame0idx
        self.frame1 = frame1idx

    def plot_triangle_matches(self):
        points1 = get_points(self.df, self.frame0)
        points2 = get_points(self.df, self.frame1)

        C, tri1, tri2, centroids1, centroids2 = C_IJ(self.df, self.frame0, self.frame1)
        pair12 = pair(C, centroids1, centroids2)

        plt.figure(figsize=(6, 6))
        plt.scatter(points1[:, 0], points1[:, 1], label="Frame 1", s=20)
        plt.scatter(points2[:, 0], points2[:, 1], label="Frame 2", s=20)
        plt.scatter(centroids1[:, 0], centroids1[:, 1], marker='x')
        plt.scatter(centroids2[:, 0], centroids2[:, 1], marker='x')

        for i, j in pair12:
            p1 = centroids1[i]
            p2 = centroids2[j]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.5)

        plt.legend()
        plt.gca().set_aspect('equal')
        plt.title("Triangle Centroid Matching")
        plt.xlim(width)
        plt.ylim(height)
        plt.show()

    def plot_interpolated_field(self, res=RESOLUTION):
        velocity, _ = interpolate_velocity_field(self.df, self.frame0, self.frame1)

        x, z = np.meshgrid(np.linspace(width[0], width[1], res),
                            np.linspace(height[0], height[1], res))

        u = np.zeros_like(x)
        w = np.zeros_like(z)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                v = velocity(x[i, j], z[i, j])
                u[i, j] = v[0]
                w[i, j] = v[1]

        plt.figure(figsize=(12, 12))
        plt.quiver(x, z, u, w)

        u_true, w_true = vector_field(x, z, 0)
        plt.quiver(x, z, u_true, w_true, color='r', alpha=0.5)
        plt.show()

    def test_pairing(self):
        C, tri1, tri2, centroids1, centroids2 = C_IJ(self.df, 0, 1)
        pair12 = pair(C, centroids1, centroids2)

        def distance(pairs, c1, c2):
            return [np.linalg.norm(c1[p1] - c2[p2]) for p1, p2 in pairs]

        x = distance(pair12, centroids1, centroids2)
        plt.plot(x, 'o')
        plt.show()

    def rmse(self, df, res=RESOLUTION):
        velocity, _ = interpolate_velocity_field(df, self.frame0, self.frame1)

        x, z = np.meshgrid(np.linspace(width[0], width[1], res),
                            np.linspace(height[0], height[1], res))

        u_pred = np.zeros_like(x)
        w_pred = np.zeros_like(z)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                v = velocity(x[i, j], z[i, j])
                u_pred[i, j] = v[0]
                w_pred[i, j] = v[1]

        u_true, w_true = vector_field(x, z, 0)
        mask = ~np.isnan(u_pred)

        err = (u_pred[mask] - u_true[mask]) ** 2 + \
              (w_pred[mask] - w_true[mask]) ** 2

        return np.sqrt(np.mean(err))

    def plot_rmse_per_numpoints(self, num_points_list):
        results = []
        for num_points in num_points_list:
            df = generate_points(width, height, num_points, num_frames, dt=dt, type='grid')
            result = self.rmse(df)
            results.append(result)  

        plt.figure()
        plt.plot(num_points_list, results, 'o-')
        plt.xlabel("Number of Points")
        plt.ylabel("RMSE")
        plt.show()

# Usage
m = Main(df, 0, 1)
m.plot_interpolated_field()
m.test_pairing()
m.plot_triangle_matches()
m.plot_rmse_per_numpoints([25, 50, 100, 200, 400])
