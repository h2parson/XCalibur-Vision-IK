import numpy as np

def trim_yaw(q, left):
    q = np.array(q)
    trim_frac = 0.15
    ramp_frac = [0.2,0.75]
    range_ = q[0][2]-q[-1][2]

    if left:
        lb = q[-1][2] + trim_frac * range_
        idxs = np.where(q[:, 2] > lb)[0]  # all indices where condition is True
    else:
        ub = q[0][2] + trim_frac * range_
        idxs = np.where(q[:, 2] < ub)[0]

    trimmed =  q[idxs]

    start = trimmed[0][2]
    range_ = trimmed[-1][2] - trimmed[0][2]
    if left:
        mid_start = np.where(trimmed[:,2] <= start+ramp_frac[0]*range_)[0][0]
        mid_end = np.where(trimmed[:,2] < start+ramp_frac[1]*range_)[0][0]
    else:
        mid_start = np.where(trimmed[:,2] >= start+ramp_frac[0]*range_)[0][0]
        mid_end = np.where(trimmed[:,2] > start+ramp_frac[1]*range_)[0][0]

    return trimmed, start, range_, mid_start, mid_end
