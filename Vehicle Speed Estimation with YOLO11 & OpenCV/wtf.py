import cv2
import numpy as np

# Stream URL
stream_url = "https://p.webcamromania.ro/piataromanab/tracks-v1/mono.m3u8"
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("❌ Failed to open stream.")
    exit()

# Define pixel points from the original frame
src_pts = np.array([
    [616, 515],   # Top-left
    [1239, 428],  # Top-right
    [1916, 806],  # Bottom-right
    [681, 946]    # Bottom-left
], dtype=np.float32)

# Default real-world measurements in meters (edit these as needed)
x = 12.0  # top side
y = 18.0  # right side
z = 20.0  # bottom side
w = 16.0  # left side

# We'll define a rectangle in real-world space
dst_width = int(max(x, z) * 10)   # scale: 10 pixels per meter
dst_height = int(max(y, w) * 10)

dst_pts = np.array([
    [0, 0],                       # Top-left
    [dst_width - 1, 0],           # Top-right
    [dst_width - 1, dst_height - 1],  # Bottom-right
    [0, dst_height - 1]           # Bottom-left
], dtype=np.float32)

# Compute the perspective transform
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not available")
        break

    # Draw source polygon on original frame
    display = frame.copy()
    cv2.polylines(display, [src_pts.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=3)

    # Warp perspective
    warped = cv2.warpPerspective(frame, matrix, (dst_width, dst_height))

    # Show both frames
    cv2.imshow("Original with Trapezoid", display)
    cv2.imshow("Warped Top-Down View", warped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
