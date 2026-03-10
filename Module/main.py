import common
from profileExtraction import profileExtraction
from homography import homography
from postProcessing import knifeGeo
from IK import ik, robot
from trim_yaw import trim_yaw
from actuator_processing import velocity

import time
import numpy as np
import math

# Constants
path = "../rpiImages/bigPaper.jpg"
global_offset = np.array([104.98-80,-140.97+177,131.5], dtype=float)   # This is measured from a homed position
bevel_angle = 15 # in degrees one-sided
max_v = math.pi/10 # rad/sec

start_time = time.perf_counter()

# Functions
# Wait for USB prompt
# Need to make somthing to take image and assign path
blade_profile = profileExtraction(path, debug=False)            # pixels uncorrected
relative_profile = homography(path, blade_profile, debug=True) # in mm relative to corner of checkers
profile, normals = knifeGeo(relative_profile, bevel_angle)      # compute normals vectors and switch to global coords
profile = (profile + global_offset)                             # locate within global coords
robot = robot()                                                 # create kinematic model
q1 = ik(robot, profile, normals)                                # compute first side joint angles
q1, start, range_, mid_start, mid_end = trim_yaw(q1, False)
velocity = velocity(q1,max_v,start,range_,mid_start,mid_end)

# q2 = ik(robot, profile, common.flipZ(normals))                  # compute other side
# q2 = trim_yaw(q2, True)

np.savez("knife_data.npz", q1, normals, profile, velocity, start, range_, mid_start, mid_end)

# Output over USB