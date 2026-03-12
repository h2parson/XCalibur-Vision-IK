import numpy as np

def merge_yaw(q):
    q = np.array(q)
    result = []
    i = 0

    while i < len(q):
        # Find end of block where q[:,2] is constant
        j = i + 1
        while j < len(q) and q[j][2] == q[i][2]:
            j += 1

        # Average the block q[i:j] across all joint variables
        block = q[i:j]
        result.append(np.mean(block, axis=0))

        i = j

    return np.array(result)

def trim_yaw(q, left):
    q = np.array(q)
    trim_frac = 0.15
    ramp_frac = [0.2,0.8]
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
        print(mid_start)
        print(mid_end)

    return trimmed, start, range_, mid_start, mid_end

def process_yaw(q, left):
    q = merge_yaw(q)
    q, start, range_, mid_start, mid_end = trim_yaw(q,left)
    return q, start, range_, mid_start, mid_end
