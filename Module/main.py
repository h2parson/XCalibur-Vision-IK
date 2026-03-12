import common
from profileExtraction import profileExtraction
from homography import homography
from postProcessing import knifeGeo
from IK import ik, robot
from yaw import process_yaw
from actuator_processing import velocity

import time
import numpy as np
from math import pi

# Constants
path = "../rpiImages/bigPaper.jpg"
global_offset = np.array([109.5-52,0145.97+19,131.5], dtype=float)   # This is measured from a homed position
bevel_angle = 15 # in degrees one-sided
q0 = [[],[0,0,pi/2,pi/2,0],[robot.links[0].qlim[1],0,pi/2,-pi/2,0]] # index 1 and 2 for q1, q2 resp.
max_v = pi/10 # rad/sec

start_time = time.perf_counter()

# Functions
# Wait for USB prompt
# Need to make somthing to take image and assign path
blade_profile = profileExtraction(path, debug=False)            # pixels uncorrected
relative_profile = homography(path, blade_profile, debug=False) # in mm relative to corner of checkers
profile, normals = knifeGeo(relative_profile, bevel_angle)      # compute normals vectors and switch to global coords
profile = (profile + global_offset)                             # locate within global coords                                               # create kinematic model
q1 = ik(robot, profile, normals, q0[1], debug=False)            # compute first side joint angles
q1, start, range_, mid_start, mid_end = process_yaw(q1, True)   # ensure yaw monotonic and segment profile
velocity = velocity(q1,max_v,start,range_,mid_start,mid_end)

# q2 = ik(robot, profile, common.flipZ(normals), q0[2])            # compute other side
# q2 = trim_yaw(q2, True)

# np.savez("knife_data.npz", q1)
np.savez("knife_data.npz", q1, [], normals, common.flipZ(normals), profile, mid_start, mid_end, velocity)
# np.savez("knife_data.npz", q1, normals, profile, velocity, start, range_, mid_start, mid_end)

# Output over USB