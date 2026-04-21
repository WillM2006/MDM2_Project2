
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
'''from shapely import points'''
from shapely.geometry import Polygon
import ast
import cv2

'''might have to flip y axis due to video having 0 at the top
also fix the fact that triangles can be formed between points extremely far away they should only be formed within that radius as it produces 
    ax.quiver(x, z, u, -w) remeber neg w'''


video = 'data/ExperimentalVideo.mp4'
csv_file = 'data/tracked_particles_minmass_100.csv'

height = (0, 1440)
width = (0, 1024)
dt = 1/30 # 30 fps

CHECKRADIUS = 2
RES = 25
MASK = False

def video_frame(video, frameidx):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameidx)
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()
    return gray_frame

df = pd.read_csv(csv_file)
df['x'] = df['x'].apply(ast.literal_eval)
df['y'] = df['y'].apply(ast.literal_eval)

def get_points(df, frameidx):
    subset = df.loc[df['frame'] == frameidx]
    
    if len(subset) == 0:
        return np.empty((0, 2))

    row = subset.iloc[0]

    if len(row['x']) == 0:
        return np.empty((0, 2))
    
    points = np.column_stack((row['x'], row['y']))

    if MASK:
        mask = (points[:, 1] >= 550) & (points[:, 1] <= 750)
        points = points[mask]

    return points

def C_IJ(df, frame1idx, frame2idx):
    def inRadius(point1, point2, R):
        r = np.sqrt((point1[0]-point2[0])**2
                    +(point1[1]-point2[1])**2)

        if r <= R:
            return True
        else:
            return False

    def tri_area(tri_pts):
        a, b, c = tri_pts
        return 0.5 * abs(np.linalg.det([b - a, c - a]))

    def tri_overlap_area(tri1_pts, tri2_pts):
        poly1 = Polygon(tri1_pts)
        poly2 = Polygon(tri2_pts)
        return poly1.intersection(poly2).area

    points1 = get_points(df, frame1idx)
    points2 = get_points(df, frame2idx)


    
    
    if len(points1) < 3 or len(points2) < 3:
        raise ValueError('Not enough points for Delaunay triangulation')

    tri1 = Delaunay(points1)
    tri2 = Delaunay(points2)

    centroids1 = np.mean(tri1.points[tri1.simplices], axis=1)
    centroids2 = np.mean(tri2.points[tri2.simplices], axis=1)


    C = np.zeros((len(tri1.simplices), len(tri2.simplices)))

    for i, tri_i in enumerate(tri1.simplices):
        centre_i = centroids1[i]
        
        for j, tri_j in enumerate(tri2.simplices):
            centre_j = centroids2[j]
            if inRadius(centre_i, centre_j, CHECKRADIUS):
                tri_i_pts = tri1.points[tri_i]
                tri_j_pts = tri2.points[tri_j]
                C[i, j] = tri_overlap_area(tri_i_pts, tri_j_pts) / np.sqrt(tri_area(tri_i_pts) * tri_area(tri_j_pts))

    
    return C, tri1, tri2, centroids1, centroids2

def pair(C, centroids1, centroids2, max_dist=CHECKRADIUS):
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

def plot_triangle_matches(df, frame1idx, frame2idx):
    points1 = get_points(df, frame1idx)
    points2 = get_points(df, frame2idx)

    C, tri1, tri2, centroids1, centroids2 = C_IJ(df, frame1idx, frame2idx)

    pair12 = pair(C, centroids1, centroids2)

    fig, ax = plt.subplots(1,2,figsize=(12,10))

    ax[0].scatter(points1[:, 0], points1[:, 1])
    ax[0].scatter(points2[:, 0], points2[:, 1])

    ax[1].scatter(centroids1[:, 0], centroids1[:, 1], marker='x')
    ax[1].scatter(centroids2[:, 0], centroids2[:, 1], marker='x')

    for i, j in pair12:
        p1 = centroids1[i]
        p2 = centroids2[j]

        ax[1].plot([p1[0], p2[0]],
                [p1[1], p2[1]],
                'k-', alpha=0.5)

    plt.show()

def interpolate_velocity_field(df, frame1idx, frame2idx):
    velocities = calc_velocity(df, frame1idx, frame2idx)
    points = np.array([p for p, v in velocities])
    values = np.array([v for p, v in velocities])
    
    if len(points) < 3:
        raise ValueError('Not enough points for Delaunay triangulation')
    
    tri = Delaunay(points)

    def velocity(x, y):
        simplex = tri.find_simplex([[x, y]])
        if simplex < 0:
            return np.array([np.nan, np.nan])

        s = simplex[0]
        vertices = tri.simplices[s]

        T = tri.transform[s]
        bary = np.dot(T[:2], [x, y] - T[2])
        bary = np.append(bary, 1 - bary.sum())

        return np.sum(values[vertices] * bary[:,None], axis=0)

    return velocity, tri



def create_interpolated_field(df, frame1idx, frame2idx, res=RES):
    velocity, _ = interpolate_velocity_field(df, frame1idx, frame2idx)
    
    x, z = np.meshgrid(np.linspace(width[0], width[1], res),
                       np.linspace(height[0], height[1], res))

    u = np.zeros_like(x)
    w = np.zeros_like(z)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            v = velocity(x[i, j], z[i, j])
            u[i, j] = v[0]
            w[i, j] = v[1]

    frame1 = video_frame(video, frame1idx)
    frame2 = video_frame(video, frame2idx)

    points1 = get_points(df, frame1idx)
    points2 = get_points(df, frame2idx)
    tri = Delaunay(points1)


    fig, ax = plt.subplots(figsize=(12,12))
    ax.quiver(x, z, u, -w, color='red')
    ax.imshow(frame1, cmap="gray")
    ax.set_axis_off()

    '''ax.scatter(points1[:, 0], points1[:, 1], marker='x')
    ax.scatter(points2[:, 0], points2[:, 1], marker='x')
    plt.triplot(points1[:,0], points1[:,1], tri.simplices)'''
    

    plt.show()




from matplotlib.animation import FuncAnimation
def update(frameidx,ax):
    ax.clear()

    velocity, _ = interpolate_velocity_field(df, frameidx, frameidx+1)
    
    x, z = np.meshgrid(np.linspace(width[0], width[1], RES),
                       np.linspace(height[0], height[1], RES))

    u = np.zeros_like(x)
    w = np.zeros_like(z)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            v = velocity(x[i, j], z[i, j])
            u[i, j] = v[0]
            w[i, j] = v[1]

    frame = video_frame(video, frameidx)

    ax.imshow(frame, cmap="gray")
    ax.quiver(x, z, u, -w, color='red')

    points1 = get_points(df, frameidx)
    tri = Delaunay(points1)
    '''
    ax.scatter(points1[:, 0], points1[:, 1], marker='x')
    plt.triplot(points1[:,0], points1[:,1], tri.simplices)
    '''
    ax.set_axis_off()

def run_animation():
    fig, ax = plt.subplots(figsize=(12, 12))
    anim = FuncAnimation(
        fig,
        update, 
        frames=range(idx1, 1750),
        interval=50,
        fargs=(ax,) 
    )
    plt.show()


idx1 = 1700
idx2 = idx1+1
create_interpolated_field(df, idx1, idx2)
plot_triangle_matches(df, idx1, idx2)
run_animation()
