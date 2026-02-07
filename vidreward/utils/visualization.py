import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from ..extraction.trajectory import HandTrajectory, ObjectTrajectory

def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray, confidence: float, threshold: float = 0.5):
    """
    Draw 21 MediaPipe landmarks on the frame.
    frame: BGR image
    landmarks: (21, 3) normalized (x, y, z)
    """
    if confidence < threshold:
        return frame
    
    h, w, _ = frame.shape
    # MediaPipe connections for hands
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8), # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17) # Knuckles
    ]
    
    # Scale landmarks to pixel coordinates
    pts = []
    for lm in landmarks:
        px = int(lm[0] * w)
        py = int(lm[1] * h)
        pts.append((px, py))
        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
    
    for start, end in connections:
        cv2.line(frame, pts[start], pts[end], (0, 255, 0), 1)
        
    return frame

def draw_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    """
    Overlay a binary mask on the frame.
    mask: (H, W) binary
    """
    colored_mask = np.zeros_like(frame)
    colored_mask[mask > 0] = [0, 0, 255] # Red for object
    
    overlap = cv2.addWeighted(frame, 1.0, colored_mask, alpha, 0)
    return overlap

def plot_distances(distances: List[float], output_path: str, title: str = "Hand-Object Distance"):
    """Plot distance curve over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(distances)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Normalized Distance")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
