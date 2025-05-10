import numpy as np
import cv2 as cv
import time

cap = cv.VideoCapture("https://p.webcamromania.ro/piataromanab/tracks-v1/mono.m3u8")
stream_fps = cap.get(cv.CAP_PROP_FPS)
if stream_fps == 0 or stream_fps is None or stream_fps != stream_fps:
    stream_fps = 5
    print(f'Defaulting to {stream_fps} fps')
print(f'Stream fps: {stream_fps}')
frame_interval = 1.0 / stream_fps
last_frame_time = 0

ret, frame1 = cap.read()

prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
while cap.isOpened():
    current_time = time.time()
    if current_time - last_frame_time < frame_interval:
        continue

    ret, frame2 = cap.read()
    if not ret:
        print('no frames')
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    if cv.waitKey(1) & 0xFF == ord('q'):
        print('exiting')
        break
    prvs = next

cap.release()
cv.destroyAllWindows()

