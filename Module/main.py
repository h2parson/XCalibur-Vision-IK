from profileExtraction import profileExtraction
from homography import homography
from postProcessing import knifeGeo
import common
import cv2
import numpy as np

# Constants
path = "../orangeCalib/calibration2/image1.jpg"
global_offset = np.array([2611.5,415.03,131.5], dtype=float)   # This is measured from a homed position
bevel_angle = 15                                        # in degrees one-sided

# Functions
# Wait for USB prompt
# Need to make somthing to take image and assign path
blade_profile = profileExtraction(path, debug=True)            # pixels uncorrected
relative_profile = homography(path, blade_profile, debug=True)  # in mm relative to corner of checkers
# TODO: change so normals is one array and z switched later to save space
profile, normals1, normals2 = knifeGeo(relative_profile, bevel_angle)
profile = (profile + global_offset).astype(np.int32)

# np.savez("knife_data.npz",
#          profile=profile,
#          normals1=normals1,
#          normals2=normals2)
# Feed each profile and normal vector pair to inverse kinematics to give associated angles
# Trim list to give dead zone for yaw
# Given a desired max velocity give angular velocities for each parameter
# Output over USB