import cv2
import numpy as np
import pyautogui

def dispImage(image, text):
    screen_width, screen_height = pyautogui.size()
    h, w = image.shape[:2]
    factor = min(0.9*screen_width / w, 0.9*screen_height / h)
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    cv2.imshow(text, cv2.resize(image, (new_w, new_h)))
    cv2.waitKey(0)

def dispContour(img, contour, text, isClosed=False):
    output = img.copy()
    if contour is not None:
        cv2.polylines(output, [contour], isClosed=isClosed, color=(0, 255, 0), thickness=10)
    dispImage(output, text)

def getTopContour(img, debug=False):
    # get dimensions
    _, w_crop = img.shape[:2]
    _, w = img.shape[:2]

    # Mask red background in HSV and invert to get knife
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_not(cv2.bitwise_or(mask_red1, mask_red2))

    # Denoise mask by morphological closing and opening
    kernel = np.ones((11,11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Extract largest contour of mask to enclose the knife and optionally display
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blade_contour = max(contours, key=cv2.contourArea)

    blade_contour = blade_contour.astype(np.int32)

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

    return top_contour

def fwd_avg(vals, n):
    if n <= 0:
        raise ValueError("n must be positive")
    if n > len(vals):
        return np.array([])

    kernel = np.ones(n) / n
    return np.convolve(vals, kernel, mode='valid')

# ----------------------------
# Interactive slope viewer
# ----------------------------
def interactiveSlopeViewer(image_path):
    img = cv2.imread(image_path)
    img = cv2.flip(img, -1) # flip it
    # Crop region containing knife
    crop_x = [150, 2800]  # x range
    crop_y = [2100, 2800]   # y range
    img = img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
    if img is None:
        print("Failed to load image.")
        return

    contour = getTopContour(img)
    if contour is None or len(contour) == 0:
        print("No contour found.")
        return

    # Original contour coordinates
    x = contour[:, 0, 0].astype(np.float32)
    y = contour[:, 0, 1].astype(np.float32)

    # ----------------------------
    # Compute slopes and smooth
    # ----------------------------
    dx = np.diff(x)
    dy = np.diff(y)
    slopes = dy / (dx + 1e-6)
    slopes_smooth = fwd_avg(slopes, 50)

    x_slope = x[50:]
    y_slope = y[50:]

    # ----------------------------
    # Scale image to fit screen
    # ----------------------------
    screen_w, screen_h = pyautogui.size()
    h, w = img.shape[:2]
    scale = min(0.9 * screen_w / w, 0.9 * screen_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_small = cv2.resize(img, (new_w, new_h))

    # Scale slope x/y coordinates
    x_scaled = (x_slope * scale).astype(int)
    y_scaled = (y_slope * scale).astype(int)
    contour_scaled = np.stack([x * scale, y * scale], axis=1).astype(int).reshape(-1,1,2)

    # ----------------------------
    # Build slope lookup arrays
    # ----------------------------
    slope_array = np.full(new_w, -np.inf, dtype=float)
    y_array = np.zeros(new_w, dtype=int)
    for i in range(len(x_scaled)):
        px = x_scaled[i]
        if px >= new_w:
            continue
        slope_val = abs(slopes_smooth[i])
        py = y_scaled[i]
        if slope_val > slope_array[px]:
            slope_array[px] = slope_val
            y_array[px] = py

    # ----------------------------
    # Prepare callback parameters
    # ----------------------------
    params = {
        "scale": scale,
        "x_slope_orig": x_slope,
        "y_slope_orig": y_slope,
        "slopes_smooth": slopes_smooth
    }

    # ----------------------------
    # Mouse callback
    # ----------------------------
    display = img_small.copy()
    cv2.namedWindow("Contour Viewer")

    def mouse_callback(event, mx, my, flags, param):
        nonlocal display
        if event == cv2.EVENT_MOUSEMOVE:
            display = img_small.copy()
            cv2.polylines(display, [contour_scaled], False, (0,255,0), 2)

            scale = param["scale"]
            x_slope_orig = param["x_slope_orig"]
            y_slope_orig = param["y_slope_orig"]
            slopes_smooth = param["slopes_smooth"]

            if 0 <= mx < new_w and slope_array[mx] > -np.inf:
                slope_val = slope_array[mx]
                py_scaled = y_array[mx]

                # Find nearest original x
                idx = np.argmin(np.abs((x_slope_orig * scale).astype(int) - mx))
                orig_x = x_slope_orig[idx]
                orig_y = y_slope_orig[idx]

                # Draw representative point
                cv2.circle(display, (mx, py_scaled), 6, (0,0,255), -1)
                text = f"x={mx} (orig {orig_x:.1f}), max|slope|={slope_val:.3f}"
                cv2.putText(display, text, (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,0,255), 2)

            cv2.imshow("Contour Viewer", display)

    cv2.setMouseCallback("Contour Viewer", mouse_callback, params)

    # Initial display
    cv2.polylines(display, [contour_scaled], False, (0,255,0), 2)
    cv2.imshow("Contour Viewer", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------------
# CHANGE THIS PATH
# ----------------------------
interactiveSlopeViewer(
    r"C:\Vision\enclosure.jpg"
)
