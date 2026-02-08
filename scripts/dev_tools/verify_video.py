import cv2
import os

video_path = "adroit_simulation.mp4"
if not os.path.exists(video_path):
    print(f"File not found: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video verified: {video_path}")
print(f"Frames: {frame_count}")
print(f"Resolution: {width}x{height}")
cap.release()
