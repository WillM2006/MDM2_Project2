import numpy as np
import matplotlib.pyplot as plt

def mock_velocity_field(x: float, z: float, t: float, nu: float) -> tuple[float, float]:
    # Eq. 1 and 2 from project brief.
    e = np.exp(-2.0 * nu * t)
    u = np.sin(x) * np.cos(z) * e
    w = -np.cos(x) * np.sin(z) * e
    return (u, w)


def initial_positions(count: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    xs = (np.random.random(count) - 0.5) * scale
    zs = (np.random.random(count) - 0.5) * scale
    return (xs, zs)


def integrate_position(x: float, z: float, t: float, nu: float) -> tuple[float, float]:
    (u, w) = mock_velocity_field(x, z, t, nu)
    x += u * t
    z += w * t
    return (x, z)

timescale = 0.01
nu = 1
xs, zs = initial_positions(10000, 10.0)
for i in range(1000):
    print(f"[{i:4}/1000] generating...")
    fig, ax = plt.subplots()
    ax.scatter(xs, zs, s=10)
    fig.savefig(f"frames/{i}.png")
    plt.close(fig)
    (xs, zs) = integrate_position(xs, zs, i * timescale, nu)

print("[1000/1000] done")
