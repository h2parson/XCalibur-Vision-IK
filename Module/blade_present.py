import cv2
import numpy as np
import common

path = "../rpiImages/noBlade.jpg"
img = cv2.imread(path)

# Crop region containing knife
crop_x = [0,2500]   # x range
crop_y = [900, 2000]   # y range
crop_offset = np.array([crop_x[0], crop_y[0]], dtype=np.float64)
img_crop = img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
common.dispImage(img_crop, "cropped")

# get dimensions
h, w = img_crop.shape[:2]
grey_lwr_thr = np.array([0, 0, 75])
grey_upr_thr = np.array([60, 30, 255])
mask = cv2.inRange(img_crop, grey_lwr_thr, grey_upr_thr)
mask = cv2.bitwise_not(mask)

common.dispContour(mask, None, "original mask")
# Denoise mask by morphological closing and opening
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
common.dispContour(mask, None, "closed and opened mask")

area_pixels = cv2.countNonZero(mask)
total_pixels = h * w
area_fraction = area_pixels / total_pixels
print(f"Mask area: {area_pixels} px ({area_fraction*100:.2f}% of crop)")

# threshold should be around 12%