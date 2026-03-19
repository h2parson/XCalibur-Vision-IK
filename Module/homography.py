import cv2
import numpy as np
import common

def makeBaseArray(origin, spacing, checker_dimensions):
    cols, rows = checker_dimensions

    # Adjust origin to be bottom-left by offsetting y by full grid height
    adjusted_origin = [origin[0], origin[1] - (rows - 1) * spacing[1]]

    xs = np.arange(cols) * spacing[0] + adjusted_origin[0]
    ys = np.arange(rows) * spacing[1] + adjusted_origin[1]

    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.stack([grid_x, grid_y], axis=2).reshape(-1, 2)

    cornersBase = grid_points.reshape(-1,1,2).astype(np.float32)

    return cornersBase

def orderCorners(checker_dimensions, corners):
    cols, rows = checker_dimensions
    corners2 = corners.reshape(-1, 2)
    corners_rows = corners2.reshape(rows, cols, 2)
    order = np.argsort(np.mean(corners_rows[:,:,1], axis=1))
    corners_rows = corners_rows[order]
    for i in range(rows):
        order = np.argsort(corners_rows[i,:,0])
        corners_rows[i] = corners_rows[i][order]
    corners = corners_rows.reshape(-1,1,2).astype(np.float64)
    return corners

def viewCorners(img, corners):
    h, w = img.shape[:2]
    output = img.copy()

    for c in corners:
        cx, cy = c[0]  # c[0] is [x, y]
        thickness = int(0.005 * h)
        cv2.circle(output, (int(cx), int(cy)), thickness, (0,255,0), -1)

    common.dispImage(output, "centers drawn")

def homography(path, blade_profile, plane_origin, plane_ratio, debug=False):
    img = cv2.imread(path)
    img = cv2.flip(img, -1) # flip it
    h, w = img.shape[:2]

    # Crop region containing checkerboard (adjust these values to your setup)
    crop_x = [700, 2350]  # x range
    crop_y = [1100, 1650]   # y range
    crop_offset = np.array([crop_x[0], crop_y[0]], dtype=np.float64)
    img_crop = img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    if debug: common.dispImage(gray, "cropped")

    checker_dimensions = [15,4]

    ret, corners = cv2.findChessboardCorners(gray, checker_dimensions)
    if not ret:
        print(f"No chessboard found in {path}")
        return

    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    # Convert corners back to full image coordinates
    corners = corners + crop_offset

    corners = orderCorners(checker_dimensions, corners)
    if debug: viewCorners(img, corners)
    
    pixel_origin = [800, 1200]
    pixel_spacing = [100,100]
    real_spacing = [9.36363636364*plane_ratio, 9.36363636364*plane_ratio]
    scale = [real_spacing[0]/pixel_spacing[0],real_spacing[1]/pixel_spacing[1]]

    cornersBase = makeBaseArray(pixel_origin, pixel_spacing, checker_dimensions)

    H, _ = cv2.findHomography(corners, cornersBase, cv2.RANSAC)

    warped_profile = cv2.perspectiveTransform(blade_profile.astype(np.float64), H)
    warped_origin = cv2.perspectiveTransform(plane_origin.reshape(1, 1, 2).astype(np.float64), H)
    warped_origin = warped_origin.reshape(2)  # flatten to (2,) for arithmetic and circle

    # Post-compose with a rotation about the image centre
    angle_deg = -3.0
    cx, cy = warped_origin[0], warped_origin[1]  # or whatever centre makes sense

    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    R = np.array([
        [cos_t, -sin_t, cx * (1 - cos_t) + cy * sin_t],
        [sin_t,  cos_t, cy * (1 - cos_t) - cx * sin_t],
        [0,      0,     1]
    ])

    warped_profile = cv2.perspectiveTransform(warped_profile, R)

    if debug:
        warped_profile = warped_profile.astype(np.int32)
        warped_img = cv2.warpPerspective(img, H, (int(1.1*w), int(1.1*h)))
        warped_img = cv2.warpPerspective(warped_img, R, (int(1.1*w), int(1.1*h)))
        cv2.circle(warped_img, (int(warped_origin[0]), int(warped_origin[1])), 50, (0, 0, 255), -1)
        common.dispContour(warped_img, warped_profile, "warped profile")

    relative_profile = ((warped_profile - warped_origin) * scale).astype(np.int32)


    return relative_profile