import cv2
import numpy as np
import common

def fwd_avg(vals, n):
    if n <= 0:
        raise ValueError("n must be positive")
    if n > len(vals):
        return np.array([])

    kernel = np.ones(n) / n
    return np.convolve(vals, kernel, mode='valid')

def plane_origin(img, debug=False):
    # Crop region containing knife
    crop_x = [900, 2200]  # x range
    crop_y = [2800, 3200]   # y range
    crop_offset = np.array([crop_x[0], crop_y[0]], dtype=np.float64)
    img_crop = img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]

    if debug: common.dispImage(img_crop, "cropped")

    # get dimensions
    _, w_crop = img_crop.shape[:2]
    _, w = img.shape[:2]

    # Mask red background in HSV and invert to get knife
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    lower_grey = np.array([0, 0, 100])
    upper_grey = np.array([255, 180, 255])
    mask1 = cv2.inRange(hsv, lower_grey, upper_grey)
    lower_red = np.array([0, 0, 0])
    upper_red = np.array([255, 255, 100])
    mask1 = cv2.inRange(hsv, lower_grey, upper_grey)
    mask2 = cv2.inRange(img_crop, lower_red, upper_red)
    mask = cv2.bitwise_and(mask1, mask2)

    if debug: common.dispContour(mask, None, "original mask")

    # Denoise mask by morphological closing and opening
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if debug: common.dispContour(mask, None, "closed and opened mask")

    # Find point where y - x is maximized
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        print("No mask pixels found")
        return

    idx = np.argmax(ys - xs)
    best_x, best_y = int(xs[idx]), int(ys[idx])

    if debug:
        vis = img_crop.copy()
        pt = (int(best_x), int(best_y))
        cv2.circle(vis, pt, 20, (0, 255, 0), -1)
        common.dispImage(vis, f"origin: {pt}")

    origin = np.array([best_x, best_y])
    origin = origin + crop_offset

    return origin

def originExtraction(path, debug=False):
    img = cv2.imread(path)
    img = cv2.flip(img, -1) # flip it
    blade_profile = plane_origin(img, debug=debug)
    return blade_profile
