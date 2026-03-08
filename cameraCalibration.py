import numpy as np
import cv2
import glob
import pyautogui

def dispImage(image, text):
    screen_width, screen_height = pyautogui.size()
    h, w = image.shape[:2]
    factor = min(0.9*screen_width / w, 0.9*screen_height / h)
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    cv2.imshow(text, cv2.resize(image, (new_w, new_h)))
    cv2.waitKey(0)
 
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# Chessboard size: 10 columns × 6 rows
cols, rows = 10, 6
square_size = 3.0  # spacing in mm

# Initialize 3D points
objp = np.zeros((rows*cols, 3), np.float32)

# Set x, y coordinates, scaled by square size
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob('../orangeCalib/calibration2/*.jpg')
 
for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (10,6))
    print(ret)
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        # cv2.drawChessboardCorners(img, (10,6), corners2, ret)
        # dispImage(img, '')
        # cv2.waitKey(500)
 
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("mtx", mtx)
print("dist", dist)

img = cv2.imread('../orangeCalib/calibration2/image1.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

print(newcameramtx)
print(roi)

'''
[[1.55524965e+04 0.00000000e+00 2.15945196e+03]
 [0.00000000e+00 1.32500926e+04 1.81971557e+03]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

(1697, 1483, 1798, 1025)
'''

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(mean_error/len(objpoints)) )

mtx = np.array([
    [1.25600312e+04, 0.00000000e+00, 2.55110203e+03],
    [0.00000000e+00, 1.18360511e+04, 1.95108067e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float64)

newcameramtx = np.array([
    [1.55524965e+04, 0.00000000e+00, 2.15945196e+03],
    [0.00000000e+00, 1.32500926e+04, 1.81971557e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float64)

roi = [1697, 1483, 1798, 1025]

dist = np.array([[1.70927262e+01, -1.00372666e+03, -1.77595261e-01, -3.35225070e-01,
   2.56109929e+04]], dtype=np.float64) 

img = cv2.imread('../orangeCalib/calibration2/image1.jpg')

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]

dispImage(dst,"")
