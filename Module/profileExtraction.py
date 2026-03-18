import cv2
import numpy as np
import common

'''
def fwd_avg(vals, n):
    if n <= 0:
        raise ValueError("n must be positive")
    if n > len(vals):
        return np.array([])

    kernel = np.ones(n) / n
    return np.convolve(vals, kernel, mode='valid')

def getBladeContour(img, debug=False):
    # Crop region containing knife
    crop_x = [1900, 4500]  # x range
    crop_y = [2100, 2800]   # y range
    crop_offset = np.array([crop_x[0], crop_y[0]], dtype=np.float64)
    img_crop = img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]

    if debug: common.dispImage(img_crop, "cropped")

    # get dimensions
    _, w_crop = img_crop.shape[:2]
    _, w = img.shape[:2]

    grey_lwr_thr = np.array([0, 0, 100])
    grey_upr_thr = np.array([20, 30, 255])
    mask = cv2.inRange(img_crop, grey_lwr_thr, grey_upr_thr)
    mask = cv2.bitwise_not(mask)
    
    if debug: common.dispContour(mask, None, "original mask")

    # Denoise mask by morphological closing and opening
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if debug: common.dispContour(mask, None, "closed and opened mask")

    # Extract largest contour of mask to enclose the knife and optionally display
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blade_contour = max(contours, key=cv2.contourArea)

    # Convert profile back to global image
    blade_contour = blade_contour + crop_offset
    blade_contour = blade_contour.astype(np.int32)

    if debug: common.dispContour(img, blade_contour, "largest contour")

    # Take upper half of contour
    top_contour = []

    for x in range(w):
        slice = np.where(blade_contour[:, 0, 0] == x)[0]
        if slice.size > 0:
            top_pt = slice[np.argmin(blade_contour[slice, 0, 1])]
            top_contour.append(blade_contour[top_pt:top_pt+1, 0:1, :])  # shape (1,1,2)

    top_contour = np.concatenate(top_contour, axis=0).astype(np.int32) # shape (N,1,2)
    top_contour = top_contour[np.argsort(top_contour[:, 0, 0])] # sort by ascending x values

    if debug: common.dispContour(img, top_contour, "top contour")

    # Find blade hilt as first point to left of clamp with slope less than threshold
    slope_threshold = -0.4
    slope_samples = 50
    clampLeft = 2500  # approximate coordinate of right end of clamp

    x_full = top_contour[:, 0, 0].astype(np.float64)
    y_full = top_contour[:, 0, 1].astype(np.float64)
    left_mask = x_full <= clampLeft
    x_left = x_full[left_mask] 
    y_left = y_full[left_mask]

    dx = np.diff(x_left)
    dy = np.diff(y_left)
    slopes = dy / (dx + 1e-6) # avoids div by zero
    avg_slopes = fwd_avg(slopes, slope_samples)

    hiltX = x_left[np.where(avg_slopes < slope_threshold)[0][0]]

    # Find blade tip as first point to right of clamp with downward slope moving avg more than threshold
    slope_threshold = 10
    slope_samples = 50
    clampRight = 2500  # approximate coordinate of right part of image

    x_full = top_contour[:, 0, 0].astype(np.float64)
    y_full = top_contour[:, 0, 1].astype(np.float64)
    right_mask = x_full <= clampRight
    x_right = x_full[right_mask] 
    y_right = y_full[right_mask]

    dx = np.diff(x_right)
    dy = np.diff(y_right)
    slopes = dy / (dx + 1e-6) # avoids div by zero
    avg_slopes = fwd_avg(slopes, slope_samples)

    tipX = x_right[np.where(avg_slopes > slope_threshold)[0][-1]] - 1

    # Now split the contour based on the hilt and get the blade profile
    xMask = (top_contour[:,0,0] >= hiltX) & (top_contour[:,0,0] <= tipX)
    blade_profile = top_contour[xMask]
    blade_profile = blade_profile.astype(np.int32)

    if debug: common.dispContour(img, blade_profile, "blade profile")

    return blade_profile

def profileExtraction(path, debug=False):
    img = cv2.imread(path)
    img = cv2.flip(img, 0) # flip it
    blade_profile = getBladeContour(img, debug=debug)
    if debug: common.dispContour(img, blade_profile, "blade profile")
    return blade_profile
'''

def fwd_avg(vals, n):
    if n <= 0:
        raise ValueError("n must be positive")
    if n > len(vals):
        return np.array([])

    kernel = np.ones(n) / n
    return np.convolve(vals, kernel, mode='valid')

def getBladeContour(img, debug=False):
    # Crop region containing knife
    crop_x = [150, 2800]  # x range
    crop_y = [2100, 2800]   # y range
    crop_offset = np.array([crop_x[0], crop_y[0]], dtype=np.float64)
    img_crop = img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]

    if debug: common.dispImage(img_crop, "cropped")

    # get dimensions
    _, w_crop = img_crop.shape[:2]
    _, w = img.shape[:2]

    # Mask red background in HSV and invert to get knife
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_not(cv2.bitwise_or(mask_red1, mask_red2))

    if debug: common.dispContour(mask, None, "original mask")

    # Denoise mask by morphological closing and opening
    kernel = np.ones((11,11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if debug: common.dispContour(mask, None, "closed and opened mask")

    # Extract largest contour of mask to enclose the knife and optionally display
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blade_contour = max(contours, key=cv2.contourArea)

    # Convert profile back to global image
    blade_contour = blade_contour + crop_offset
    blade_contour = blade_contour.astype(np.int32)

    if debug: common.dispContour(img, blade_contour, "largest contour")

    # Take upper half of contour
    top_contour = []

    x_min = int(blade_contour[:, 0, 0].min())
    x_max = int(blade_contour[:, 0, 0].max())

    for x in range(x_min, x_max + 1):
        slice = np.where(blade_contour[:, 0, 0] == x)[0]
        if slice.size > 0:
            top_pt = slice[np.argmin(blade_contour[slice, 0, 1])]
            top_contour.append(blade_contour[top_pt:top_pt+1, 0:1, :])  # shape (1,1,2)

    top_contour = np.concatenate(top_contour, axis=0).astype(np.int32) # shape (N,1,2)
    top_contour = top_contour[np.argsort(top_contour[:, 0, 0])] # sort by ascending x values

    if debug: common.dispContour(img, top_contour, "top contour")

    # Find blade hilt as first point to left of clamp with slope less than threshold
    slope_threshold = 0.4
    slope_samples = 50
    clampRight = 2000  # approximate coordinate of right end of clamp

    x_full = top_contour[:, 0, 0].astype(np.float64)
    y_full = top_contour[:, 0, 1].astype(np.float64)
    right_mask = x_full >= clampRight
    x_right = x_full[right_mask] 
    y_right = y_full[right_mask]

    dx = np.diff(x_right)
    dy = np.diff(y_right)
    slopes = dy / (dx + 1e-6) # avoids div by zero
    avg_slopes = fwd_avg(slopes, slope_samples)

    hiltX = x_right[np.where(avg_slopes > slope_threshold)[0][0]]

    # Find blade tip as rightmost point before steep upward slope
    slope_threshold = -10
    slope_samples = 50
    clampLeft = 1500  # approximate coordinate of right part of image

    x_full = top_contour[:, 0, 0].astype(np.float64)
    y_full = top_contour[:, 0, 1].astype(np.float64)
    left_mask = x_full >= clampLeft
    x_left = x_full[left_mask] 
    y_left = y_full[left_mask]

    dx = np.diff(x_left)
    dy = np.diff(y_left)
    slopes = dy / (dx + 1e-6) # avoids div by zero
    avg_slopes = fwd_avg(slopes, slope_samples)

    matches = np.where(avg_slopes < slope_threshold)[0]
    if len(matches) > 0:
        tipX = x_right[matches[-1]] - 1
    else:
        tipX = top_contour[:, 0, 0].min()

    # Now split the contour based on the hilt and get the blade profile
    xMask = (top_contour[:,0,0] <= hiltX) & (top_contour[:,0,0] >= tipX)
    blade_profile = top_contour[xMask]
    blade_profile = blade_profile.astype(np.int32)

    if debug: common.dispContour(img, blade_profile, "blade profile")

    return blade_profile

def profileExtraction(path, debug=False):
    img = cv2.imread(path)
    img = cv2.flip(img, -1) # flip it
    blade_profile = getBladeContour(img, debug=debug)
    if debug: common.dispContour(img, blade_profile, "blade profile")
    return blade_profile
