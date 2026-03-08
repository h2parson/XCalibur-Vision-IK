import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots

def bevelVectors(v, theta):
    theta = math.radians(theta)

    v1, v2, v3 = v
    b3 = math.sin(theta)
    A = 2.0 * (v2 * v3) / (v1 ** 2) * math.sin(theta)
    B = 1.0 + (v2 ** 2) / (v1 ** 2)
    C = (v3 ** 2) / (v1 ** 2) * (math.sin(theta) ** 2) - (math.cos(theta) ** 2)
    disc = A ** 2 - 4.0 * B * C
    if disc < 0:
        # If you want, you could skip this vector or set NaNs instead
        raise ValueError(f"No real solution for bevel vector: {v}")
    b2 = (-A + math.sqrt(disc)) / (2.0 * B * C)
    b1 = -(v2 / v1) * b2 - (v3 / v1) * b3
    b = np.array([b1, b2, b3], dtype=float)

    return b

def normal(b,v):
    c = np.cross(b, v)
    c = c/np.linalg.norm(c)
    return c

# Example vectors
v = np.array([-0.12709859,  0.99189009,  0])
b = bevelVectors(v, 15)  # Your function
n = normal(b, v)         # Your function

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original vector v in red
ax.quiver(0, 0, 0, v[0], v[1], v[2], color='r', label='v', linewidth=2, arrow_length_ratio=0.1)

# Plot the bevel vector b in green
ax.quiver(0, 0, 0, b[0], b[1], b[2], color='g', label='b', linewidth=2, arrow_length_ratio=0.1)

# Plot the normal vector n in blue
ax.quiver(0, 0, 0, n[0], n[1], n[2], color='b', label='n', linewidth=2, arrow_length_ratio=0.1)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set equal aspect ratio for all axes
max_val = max(np.linalg.norm(v), np.linalg.norm(b), np.linalg.norm(n))
ax.set_xlim([0, max_val])
ax.set_ylim([0, max_val])
ax.set_zlim([0, max_val])

# Add legend
ax.legend()

plt.show()
