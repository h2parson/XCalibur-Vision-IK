import numpy as np

def trim_yaw(q, left):
    q = np.array(q)
    trim_frac = 0.15
    range_ = q[-1][2] - q[0][2]

    if left:
        lb = q[0][2] + trim_frac * range_
        idxs = np.where(q[:, 2] > lb)[0]  # all indices where condition is True
        trimmed =  q[idxs[0] + 1:]  # first index + 1
    else:
        ub = q[-1][2] - trim_frac * range_
        idxs = np.where(q[:, 2] < ub)[0]
        trimmed = q[:idxs[-1]]  # last index satisfying condition

    start = trimmed[0][2]
    range_ = trimmed[-1][2] - q[0][2]
    mid_start = np.where(trimmed[:,2] >= start+range_/3)[0][0]
    mid_end = np.where(trimmed[:,2] > start+2*range_/3)[0][0]

    return trimmed, start, range_, mid_start, mid_end
