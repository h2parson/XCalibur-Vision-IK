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

# Load image
image_path = "../rpiImages/nine.jpg"
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize to fit screen
screen_width, screen_height = pyautogui.size()
h, w = gray.shape[:2]
factor = min(0.9 * screen_width / w, 0.9 * screen_height / h)

print(factor)

new_w = max(1, int(w * factor))
new_h = max(1, int(h * factor))

# gray = cv2.resize(gray, (new_w, new_h))
# gray = cv2.equalizeHist(gray)

white_lwr_thr = 0
white_upr_thr = 45
mask = cv2.inRange(gray, white_lwr_thr, white_upr_thr)

# Denoise mask by morphological closing and opening
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

dispContour(mask, None, "closed and opened mask")

checker_dimensions = (8,8)

ret, corners = cv2.findChessboardCorners(gray, checker_dimensions)
if not ret:
    print(f"No chessboard found")
    
print(corners)
