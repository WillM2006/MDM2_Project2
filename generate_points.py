import numpy as np
import pandas as pd

def vector_field(x, z, t, dis=0.05):
    # vector field given in brief
    u = np.sin(x) * np.cos(z) * np.exp(-2 * dis * t)
    w = -np.cos(x) * np.sin(z) * np.exp(-2 * dis * t)
    return u, w

def rand_points(width, height, num):
    # create initial random points
    x = np.random.uniform(width[0], width[1], size=num)
    y = np.random.uniform(height[0], height[1], size=num)
    return np.column_stack((x, y))

def grid_points(width, height, num):
    # create initial square grid of points
    x = np.linspace(width[0], width[1], num)
    y = np.linspace(height[0], height[1], num)
    X, Y = np.meshgrid(x, y)
    return np.column_stack((X.ravel(), Y.ravel()))


def new_points(points, t, dt):
    u_p, w_p = vector_field(points[:, 0], points[:, 1], t)
    points[:, 0] += u_p * dt
    points[:, 1] += w_p * dt
    return points

def generate_points(width: tuple, # width is a tuple of (min, max) for x values
                    height: tuple, # height is a tuple of (min, max) for y values
                    num_points: int, # number of points to generate
                    t: float, # t is time to generate points for
                    dt: float = 0.01, # dt is time step between each point generation
                    type: str = 'random' # type of initial points either random or grid
                    ) -> pd.DataFrame:
    
    
    time = int(t / dt) + 1
    data = []

    match type:
        case 'random':
            points = rand_points(width, height, num_points)
        case 'grid':
            points = grid_points(width, height, num_points)

    x_vals = points[:, 0]
    y_vals = points[:, 1]

    data.append({"x": x_vals.copy(), "y": y_vals.copy()})

    for i in range(1, time):
        points = new_points(points, i * dt, dt)
        x_vals = points[:, 0]
        y_vals = points[:, 1]
        data.append({"x": x_vals.copy(), "y": y_vals.copy()})

    # changes index so that it represents time rather than the index
    df = pd.DataFrame(data)
    df.index = np.arange(0, time) * dt
    df.index.name = "time"
    return df

'''
#accessing points

num_points = 30
width = (-5, 5)
height = (-5, 5)

df = generate_points(width, height, num_points, 1.0)

t = 0
x_vals1 = df.loc[t, "x"]
y_vals1 = df.loc[t, "y"]


t = 1
x_vals2 = df.loc[t, "x"]
y_vals2 = df.loc[t, "y"]
print(x_vals2)

import matplotlib.pyplot as plt
plt.scatter(x_vals1, y_vals1)
plt.scatter(x_vals2, y_vals2)
plt.show()

'''