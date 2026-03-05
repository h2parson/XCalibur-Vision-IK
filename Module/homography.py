import cv2
import numpy as np
import common

def makeBaseArray(origin, spacing):
    cols, rows = 10, 6  # 10 columns (x), 6 rows (y)

    # Create grid of points
    xs = np.arange(cols) * spacing[0] + origin[0]  # 10 x-values
    ys = np.arange(rows) * spacing[1] + origin[1]  # 6 y-values

    # Create meshgrid
    grid_x, grid_y = np.meshgrid(xs, ys)  # shape (rows, cols)
    grid_points = np.stack([grid_x, grid_y], axis=2).reshape(-1, 2)  # shape (60, 2)

    # Reshape to cv2 format (N,1,2)
    cornersBase = grid_points.reshape(-1,1,2).astype(np.float32)

    return cornersBase

def viewCorners(img, corners):
    h, w = img.shape[:2]
    output = img.copy()

    for c in corners:
        cx, cy = c[0]  # c[0] is [x, y]
        thickness = int(0.005 * h)
        cv2.circle(output, (int(cx), int(cy)), thickness, (0,255,0), -1)

    common.dispImage(output, "centers drawn")

def homography(path, blade_profile, debug=False):
    img = cv2.imread(path)
    h, w = img.shape[:2]

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCornersSB(grayImg, (10,6))
    if debug: viewCorners(img, corners)
    
    pixel_origin = [3060, 2260]
    pixel_spacing = [60,60] # pixel
    [3060, 2260]
    real_spacing = [3,3]    # mm
    scale = [real_spacing[0]/real_spacing[0],real_spacing[1]/real_spacing[1]]

    cornersBase = makeBaseArray(pixel_origin, pixel_spacing)

    if not ret:
        print(f"No chessboard found in {path}")
        return

    # Compute homography
    H, mask = cv2.findHomography(corners, cornersBase, cv2.RANSAC)

    warped_profile = cv2.perspectiveTransform(blade_profile.astype(np.float32), H)
    warped_profile = warped_profile.astype(np.int32)  # <-- fix for polylines

    if debug: 
        warped_img = cv2.warpPerspective(img, H, (w, h))
        common.dispContour(warped_img,warped_profile,"warped profile")

    relative_profile = ((warped_profile - pixel_origin) * scale).astype(np.int32)

    return relative_profile
