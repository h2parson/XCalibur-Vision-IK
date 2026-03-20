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
import sys

def detect_geometry(debug=False):
    start = time()
    benchmarking = False

    '''****************************************           CONSTANTS           *****************************************'''
    path = "temp.jpg"
    knife_dist = 170
    wall_dist = knife_dist + 184
    plane_ratio = knife_dist/wall_dist
    global_offset = np.array([52.09, 45.8, 138.55], dtype=float)  # This is measured from a homed position
    bevel_angle = 15                                                             # in degrees one-sided
    q0 = [[],[0,0,pi/2,pi/2,0],[robot.links[0].qlim[1],0,pi/2,-pi/2,0]]          # index 1 and 2 for q1, q2 resp.

    '''****************************************       IMAGE CAPTURE           ****************************************'''
    # max_attempts = 5
    # for i in range(max_attempts):
    #     sleep(2)
    #     if capture():
    #         log("image succesfully captured", debug=debug)
    #         break
    #     else:
    #         log("failed to capture an image", debug=debug)
    #         if i == max_attempts-1:
    #             return False

    '''****************************************       PROFILE EXTRACTION      ****************************************'''
    blade_profile = profileExtraction(path, debug=False)                                          # pixels uncorrected
    log("profile extracted", debug=debug)
    plane_origin = originExtraction(path, debug=False)                                             # find an in-plane reference
    log("origin extracted", debug=debug)
    if benchmarking: profile_extraction_time = time() - start
    relative_profile = homography(path, blade_profile, plane_origin, plane_ratio, debug=False)     # in mm relative to corner of checkers
    log("homography performed", debug=debug)
    profile, normals = knifeGeo(relative_profile, bevel_angle)                                    # compute normals vectors and switch to global coords
    profile = (profile + global_offset)                                                      # locate within global coords
    log("profile processed", debug=debug)
    if benchmarking: homography_time = time() - start - profile_extraction_time

    '''****************************************       KINEMATICS SIDE I      ****************************************'''
    print(np.shape(profile))
    print(np.shape(normals))
    q1 = ik(robot, profile, normals, q0[1], debug=False)               # compute first side joint angles
    q1 = process_yaw(q1, True)                                         # ensure yaw monotonic and segment profile
    ratios1 = velocity_ratios(q1)                                      # calculate the velocity ratios
    ratios1 = np.array(ratios1, dtype=np.float64)
    log("kinematics 1", debug=debug)

    # '''****************************************       KINEMATICS SIDE II     ****************************************'''
    q2 = ik(robot, profile, flipZ(normals), q0[2], debug=False) # compute first side joint angles
    q2 = process_yaw(q2, True)                                         # ensure yaw monotonic and segment profile
    ratios2 = velocity_ratios(q2)  
    ratios2 = np.array(ratios2, dtype=np.float64)                                    # calculate the velocity ratios
    if benchmarking: kinematics_processing_time = time() - start - homography_time
    log("kinematics 2", debug=debug) 

    '''****************************************       PREPARE OUTPUTS        ****************************************'''
    tip_q1 = q1[0]                                                     # joint variables to reach knife tip on first side
    tip_q2 = q2[0]                                                     # joint variables to reach knife tip on second side
    yaw_indices = q1[:,2]                                              # yaw values are same on both sides
    output_array = [tip_q1, tip_q2, yaw_indices, ratios1, ratios2]     # all outputs to the stm
    total_time = time() - start

    if benchmarking: 
        log("profile_extraction_time = ",profile_extraction_time, debug=debug)
        log("homography_time = ",homography_time, debug=debug)
        log("kinematics_processing_time = ",kinematics_processing_time, debug=debug)
        log("total_time = ",total_time, debug=debug)
        total_bytes = sum(
            arr.nbytes if isinstance(arr, np.ndarray) else sys.getsizeof(arr)
            for arr in [tip_q1, tip_q2, yaw_indices, ratios1, ratios2]
        )
        log(f"output size = {total_bytes} bytes", debug=debug)

    # np.savez("knife_data.npz",
    #     tip_q1=tip_q1,
    #     tip_q2=tip_q2,
    #     yaw_indices=yaw_indices,
    #     ratios1=ratios1,
    #     ratios2=ratios2,
    # )

    return output_array

if __name__ == '__main__':
    result = detect_geometry()
    if result:
        log("success")
    else:
        log("fail")

