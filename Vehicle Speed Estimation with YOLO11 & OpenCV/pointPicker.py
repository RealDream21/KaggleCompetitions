import cv2 as cv
import numpy as np

selected_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        print(f"Point {len(selected_points)}: ({x}, {y})")
        cv.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv.imshow("Point Selection", img)

# Load image
cap = cv.VideoCapture("https://p.webcamromania.ro/piataromanab/tracks-v1/mono.m3u8")
ret, img = cap.read()
cap.release()

if not ret:
    print("Error loading image")
    exit()

# Convert to grayscale and back to BGR
img = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)

# Set up window
cv.namedWindow("Point Selection")
cv.setMouseCallback("Point Selection", mouse_callback)
cv.imshow("Point Selection", img)

print("Click to select points. Press 'q' to quit.")
while cv.waitKey(1) & 0xFF != ord('q'):
    pass

cv.destroyAllWindows()

print("\nFinal selected points:")
for i, (x, y) in enumerate(selected_points, 1):
    print(f"Point {i}: ({x}, {y})")