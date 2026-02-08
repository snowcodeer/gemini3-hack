import cv2
import numpy as np

video_path = "adroit_simulation.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    exit()

frame_count = 0
non_black_frames = 0
total_intensity = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    mean_val = np.mean(frame)
    total_intensity += mean_val
    
    if mean_val > 10: # Threshold for "not black"
        non_black_frames += 1

print(f"Total Frames: {frame_count}")
print(f"Non-Black Frames: {non_black_frames}")
print(f"Average Intensity: {total_intensity / frame_count if frame_count else 0}")

if non_black_frames == 0:
    print("DIAGNOSIS: Video is completely black.")
else:
    print("DIAGNOSIS: Video has content.")
