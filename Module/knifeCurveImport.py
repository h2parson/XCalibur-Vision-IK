from common import log, capture, flipZ
from profileExtraction import profileExtraction
from origin_extraction import originExtraction
from homography import homography
from postProcessing import knifeGeo
from IK import ik, robot
from yaw import process_yaw
from actuator_processing import velocity_ratios

import numpy as np
from math import pi
from time import time, sleep
import math

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

        c = np.cross(v,b)
        c = c/np.linalg.norm(c)
        result.append(c)
    return result

data_raw = np.loadtxt("C:/Vision/blade_points.csv", delimiter=",", skiprows=1)
blade_points   = data_raw[:, :3]*1000         # positions in mm
blade_tangents = data_raw[:, 3:]         # unit tangents (no scaling)

relative_points = blade_points - blade_points[0]
relative_points[:, 2] = 0

#rearrange points
global_points = np.array([[p[1], -p[0], p[2]] for p in relative_points])
global_offset = [90.76,44.61,130.55]
global_points = global_points + global_offset

theta = 15
bevels = np.array(bevelVectors(blade_tangents, theta))
normals = np.array(normal(bevels, blade_tangents)) * np.array([1, -1, 1])

# global_points = np.array(global_points[:, np.newaxis, :])
# normals = np.array(normal(bevels, blade_tangents))[:, np.newaxis, :] * np.array([1, -1, 1])

q0 = [[],[0,0,pi/2,pi/2,0],[robot.links[0].qlim[1],0,pi/2,-pi/2,0]] 

'''****************************************       KINEMATICS SIDE I      ****************************************'''
q1 = ik(robot, global_points, normals, q0[1], debug=False)               # compute first side joint angles
q1 = process_yaw(q1, True)                                         # ensure yaw monotonic and segment profile
ratios1 = velocity_ratios(q1)                                      # calculate the velocity ratios
ratios1 = np.array(ratios1, dtype=np.float64)
log("kinematics 1", debug=True)

# '''****************************************       KINEMATICS SIDE II     ****************************************'''
q2 = ik(robot, global_points, flipZ(normals), q0[2], debug=False) # compute first side joint angles
q2 = process_yaw(q2, True)                                         # ensure yaw monotonic and segment profile
ratios2 = velocity_ratios(q2)  
ratios2 = np.array(ratios2, dtype=np.float64)                                    # calculate the velocity ratios
log("kinematics 2", debug=True) 

'''****************************************       PREPARE OUTPUTS        ****************************************'''
tip_q1 = q1[0]                                                     # joint variables to reach knife tip on first side
tip_q2 = q2[0]                                                     # joint variables to reach knife tip on second side
yaw_indices = q1[:,2]                                              # yaw values are same on both sides
output_array = [tip_q1, tip_q2, yaw_indices, ratios1, ratios2]     # all outputs to the stm

np.savez("knife_data.npz",
    tip_q1=tip_q1,
    tip_q2=tip_q2,
    yaw_indices=yaw_indices,
    ratios1=ratios1,
    ratios2=ratios2,
)