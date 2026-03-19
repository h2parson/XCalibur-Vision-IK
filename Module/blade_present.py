import cv2
import numpy as np
import common
from common import capture
from time import sleep, time

def blade_present(path, debug=False):
    img = cv2.imread(path)
    img = cv2.flip(img, -1) # flip it

    # Crop region containing knife
    crop_x = [250, 2800]  # x range
    crop_y = [2100, 2800]   # y range
    img_crop = img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
    common.dispImage(img_crop, "cropped")
    # get dimensions
    h_crop, w_crop = img_crop.shape[:2]
    # Mask red background in HSV and invert to get knife
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 45])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 45])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_not(cv2.bitwise_or(mask_red1, mask_red2))

    if debug: common.dispContour(mask, None, "original mask")

    area_pixels = cv2.countNonZero(mask)
    total_pixels = h_crop * w_crop
    area_fraction = area_pixels / total_pixels
    if debug: print(f"Mask area: {area_pixels} px ({area_fraction*100:.2f}% of crop)")

    threshold = 0.535
    return area_fraction >= threshold

def wait_for_blade(timeout=60):
    print("waiting for blade")

    path = "temp.jpg"
    start = time()

    while time() < start + timeout:
        capture()
        print("image taken at time ", time()-start)
        if blade_present(path, debug=False):
            print("Blade found")
            return True
        
    print("Did not find a blade")
    return False

if __name__ == "__main__":
    result = wait_for_blade()