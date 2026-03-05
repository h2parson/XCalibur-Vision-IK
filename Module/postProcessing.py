import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

def profileSmoothing(blade_profile, sigma):
    y = blade_profile[:,0,1]
    
    y_smooth = gaussian_filter1d(y, sigma=sigma, mode='nearest')
    
    blade_smooth = blade_profile.copy()
    blade_smooth[:,0,1] = y_smooth  # only y changes

    # re-add the edges of original
    blade_smooth[:2*sigma,0,1] = y[:2*sigma]
    blade_smooth[-2*sigma:,0,1] = y[-2*sigma:]
    
    return blade_smooth

def sparseArray(arr, n):
    if len(arr) == 0:
        return arr
    indices = np.arange(0, len(arr), n)
    if indices[-1] != len(arr) - 1:
        indices = np.append(indices, len(arr) - 1)
    return arr[indices]

def swapXY(arr):
    for i in range(len(arr)):
        x = np.array([arr[i,0,1],arr[i,0,0]])
        arr[i] = x
    return arr

def tangent(blade_profile, sigma=101):
    x = blade_profile[:,0,0].astype(float)  # <--- convert to float
    y = blade_profile[:,0,1].astype(float)
    
    dx = gaussian_filter1d(x, sigma=sigma, order=1, mode='reflect')
    dy = gaussian_filter1d(y, sigma=sigma, order=1, mode='reflect')
    
    tangents = np.vstack([dx, dy, np.zeros_like(dx, dtype=float)]).T  # make zeros float
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    
    return tangents

def bevelVectors(v_list, theta):
    result = []
    theta = math.radians(theta)

    for v in v_list:
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

        result.append(b)

    return np.array(result)

def normal(b_list,v_list):
    result = []
    for i in range(len(v_list)):
        b = b_list[i]
        v = v_list[i]

        c = np.cross(b, v)
        c = c/np.linalg.norm(c)
        result.append(c)
    return result

def flipZ(normals):
    result = []
    for i in range(len(normals)):
        n = np.array([normals[i][0],normals[i][1],-normals[i][2]])
        result.append(n)
    return result

def to3D(smooth):
    # ensure shape is (n,2)
    smooth_2d = smooth.reshape(-1, 2)
    # add a zero column for z
    result = np.hstack([smooth_2d, np.zeros((smooth_2d.shape[0], 1))])
    return result

def knifeGeo(blade_profile, theta):
    sigmaPos = 51
    sigmaTan = 101
    sampleRatio = 10

    smooth = profileSmoothing(blade_profile,sigmaPos)
    smooth = swapXY(smooth)
    sparseSmooth = sparseArray(smooth, sampleRatio)
    sparseSmooth = swapXY(sparseSmooth)
    tangents = tangent(smooth, sigmaTan)
    sparseTangents = sparseArray(tangents, sampleRatio)
    bevels = bevelVectors(sparseTangents, theta)
    smooth3D = to3D(sparseSmooth)
    normals1 = normal(bevels,sparseTangents)
    normals2 = flipZ(normals1)
    
    return smooth3D, normals1, normals2

# def plot_sparse_and_tangent(blade_profile, theta):
#     sigmaPos = 51
#     sigmaTan = 101
#     sampleRatio = 10

#     # Process the blade profile
#     smooth = profileSmoothing(blade_profile,sigmaPos)
#     smooth = swapXY(smooth)
#     sparseSmooth = sparseArray(smooth, sampleRatio)
#     sparseSmooth = swapXY(sparseSmooth)
#     tangents = tangent(smooth, sigmaTan)
#     sparseTangents = sparseArray(tangents, sampleRatio)
    
#     sparseSmooth

#     # Pick the 300th point
#     p = sparseSmooth[300, :2]       # point on the curve (x, y only)
#     t = sparseTangents[300, :2]     # tangent vector (x, y only)

#     # Define a line along the tangent (for plotting)
#     line_length = 500  # adjust as needed
#     line_points = np.array([p - t*line_length, p + t*line_length])

#     # 2D plot (x-y plane)
#     plt.figure(figsize=(8,6))
#     plt.plot(sparseSmooth[:,0], sparseSmooth[:,1], 'k-', label='Sparse Smooth Profile')
#     plt.plot(line_points[:,0], line_points[:,1], 'r--', label='Tangent at point 300', linewidth=2)
#     plt.scatter(p[0], p[1], color='blue', label='Point 300')

#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.axis('equal')
#     plt.legend()
#     plt.title('Sparse Blade Profile and Tangent Line')
#     plt.show()
