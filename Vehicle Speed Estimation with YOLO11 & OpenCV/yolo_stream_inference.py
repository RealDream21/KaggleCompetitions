import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO
import time
from collections import defaultdict
import numpy as np
import PointMaker
from Speedometer import Speedometer

model = YOLO('yolo11n.pt')

stream_url = "https://p.webcamromania.ro/piataromanab/tracks-v1/mono.m3u8"

cap = cv2.VideoCapture(stream_url)


src_pts = np.array([
    [616, 515],   # Top-left
    [1239, 428],  # Top-right
    [1916, 806],  # Bottom-right
    [681, 946]    # Bottom-left
], dtype=np.float32)

# Default real-world measurements in meters (edit these as needed)
x = 120  # Top-left -> Top-right: x meters
y = 80  # Top-right -> Bottom-right y meters
z = 60  # Bottom-right -> Bottom-left z meters
w = 20  # Bottom-left -> Top-left : w meters 

target_pixels = 1000
max_meters = max(x, y, z, w)
scale = target_pixels / max_meters
dst_width = int(max(x, z))
dst_height = int(max(y, w))

dst_pts = np.array([
    [0, 0],                       # Top-left
    [dst_width - 1, 0],           # Top-right
    [dst_width - 1, dst_height - 1],  # Bottom-right
    [0, dst_height - 1]           # Bottom-left
], dtype=np.float32)

matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
def warp_to_world(points: list[tuple[float, float]]) -> np.ndarray:
    global matrix
    points_np = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    world_coords = cv2.perspectiveTransform(points_np, matrix)
    return world_coords.reshape(-1, 2)


if not cap.isOpened():
    print("❌ Failed to open stream.")
    exit()

stream_fps = cap.get(cv2.CAP_PROP_FPS)
if stream_fps == 0 or stream_fps is None or stream_fps != stream_fps:
    stream_fps = 5
    print(f'Defaulting to {stream_fps} fps')
print(f'Stream fps: {stream_fps}')

frame_interval = 1.0 / stream_fps
last_frame_time = 0

track_history = defaultdict(lambda: [])

speedometer = Speedometer(mapper=warp_to_world, fps=int(stream_fps))
while cap.isOpened():
    current_time = time.time()
    if current_time - last_frame_time < frame_interval:
        continue

    ret, frame = cap.read()
    if not ret:
        break

    last_frame_time = current_time

    results = model.track(frame, classes = [2], persist=True)[0]
    
    annotated_frame = results.plot()

    cv2.polylines(annotated_frame, [src_pts.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=3)

    if results.boxes and results.boxes.id is not None:
        boxes = results.boxes.xywh.cpu()
        track_ids = results.boxes.id.int().cpu().tolist()

        frame = results.plot()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            # cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10) #this adds a trail to the car back

            speedometer.update_with_trace(track_id, track)
            speed = speedometer.get_current_speed(track_id)

            cv2.putText(
                frame,
                f"{speed} km/h",
                org=(int(x - w / 2), int(y - h / 2 - 10 - h)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 0),
                thickness=3
            )


    cv2.imshow("Tracked", frame)
    # cv2.imshow("Warped Top-Down View", warped)
    # cv2.imshow("YOLOv8 Live Stream", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
