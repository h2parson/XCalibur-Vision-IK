import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# def profileSmoothing(blade_profile, sigma):
#     y = blade_profile[:,0,1]
    
#     y_smooth = gaussian_filter1d(y, sigma=sigma, mode='nearest')
    
#     blade_smooth = blade_profile.copy()
#     blade_smooth[:,0,1] = y_smooth  # only y changes

#     # re-add the edges of original
#     blade_smooth[:2*sigma,0,1] = y[:2*sigma]
#     blade_smooth[-2*sigma:,0,1] = y[-2*sigma:]
    
#     return blade_smooth

# For now I will not smooth and use only crude estimate of velocity

def velocity(q, max_v, start, range_, mid_start, mid_end):
    velocity = [[0]* 5 for i in range(len(q)-1)]

    slope = max_v/(q[mid_start-1][2])                 # First 3rd of blade
    for j in range(mid_start):
        velocity[j][2] = slope * q[j][2]              # Middle portion
    for j in range(mid_start,mid_end+1):
        velocity[j][2] = max_v
    slope = -max_v/(start + range_ - q[mid_end+1][2]) # Last 3rd of blade
    for j in range(mid_end+1,len(velocity)):
        velocity[j][2] = slope * q[j][2]

    # loop over others
    for i in (0,1,3,4):
        for j in range(len(velocity)):
            velocity[j][i] = ((q[j+1][i]-q[j][i])/(q[j+1][2]-q[j][2])) * velocity[j][2]
    
    return velocity
