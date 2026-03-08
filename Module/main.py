import common
from profileExtraction import profileExtraction
from homography import homography
from postProcessing import knifeGeo
from IK import ik, robot

import numpy as np

# Constants
path = "../rpiImages/bigPaper.jpg"
global_offset = np.array([104.98-80,-140.97+177,131.5], dtype=float)   # This is measured from a homed position
bevel_angle = 15                                        # in degrees one-sided

# Functions
# Wait for USB prompt
# Need to make somthing to take image and assign path
blade_profile = profileExtraction(path, debug=False)            # pixels uncorrected
relative_profile = homography(path, blade_profile, debug=False) # in mm relative to corner of checkers
profile, normals = knifeGeo(relative_profile, bevel_angle)      # compute normals vectors and switch to global coords
profile = (profile + global_offset)                             # locate within global coords
robot = robot()                                                 # create kinematic model
q1 = ik(robot, profile, normals)                                # compute first side joint angles
q2 = ik(robot, profile, common.flipZ(normals))                  # compute other side

# Trim list to give dead zone for yaw
# Given a desired max velocity give angular velocities for each parameter
# Output over USB