import cv2
import numpy as np
import pyautogui

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

gray = cv2.resize(gray, (new_w, new_h))
# gray = cv2.equalizeHist(gray)

# Mouse callback
def show_gray(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        gray_value = gray[y, x]

        output = gray.copy()

        # draw cursor point
        cv2.circle(output, (x, y), 5, (0, 0, 255), -1)

        text = f"X:{x} Y:{y} Gray:{gray_value}"

        cv2.putText(output, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        cv2.imshow("Gray Inspector", output)

# Window + callback
cv2.namedWindow("Gray Inspector")
cv2.setMouseCallback("Gray Inspector", show_gray)

cv2.imshow("Gray Inspector", img)

cv2.waitKey(0)
cv2.destroyAllWindows()