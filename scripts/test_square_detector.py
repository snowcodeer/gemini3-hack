import cv2
import numpy as np
from typing import Optional, List

def detect_rubiks_cube_classical(frame: np.ndarray) -> Optional[tuple]:
    """
    Detects a Rubik's cube by looking for clusters of small squares.
    Returns (x, y, w, h) bounding box.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    square_boxes = []
    for cnt in contours:
        # Approximate the contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        
        # A square should have 4 corners, be convex, and have a reasonable area
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if 100 < area < 5000:
                # Check aspect ratio
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.8 <= aspect_ratio <= 1.2:
                    square_boxes.append((x, y, w, h))
    
    if not square_boxes:
        return None
        
    # Find the largest cluster of squares (using a simple distance threshold)
    # Most Rubik's cubes show 9-27 square stickers
    # For now, let's just return the bounding box of all detected squares that are close together
    if len(square_boxes) < 2:
        # If only one, it might be the whole cube or a noise
        return square_boxes[0] if square_boxes else None

    # Simple heuristic: filter squares that have neighbors
    cluster_boxes = []
    for i, b1 in enumerate(square_boxes):
        neighbors = 0
        c1 = (b1[0] + b1[2]/2, b1[1] + b1[3]/2)
        for j, b2 in enumerate(square_boxes):
            if i == j: continue
            c2 = (b2[0] + b2[2]/2, b2[1] + b2[3]/2)
            dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
            if dist < max(b1[2], b1[3]) * 3: # Within 3 sticker widths
                neighbors += 1
        if neighbors >= 2: # At least 2 other stickers found nearby
            cluster_boxes.append(b1)
            
    if not cluster_boxes:
        # Fallback to the one with most neighbors or just all
        return cv2.boundingRect(np.array(square_boxes).reshape(-1, 4)) if square_boxes else None
        
    # Return bounding box of all squares in the cluster
    pts = []
    for x, y, w, h in cluster_boxes:
        pts.extend([[x, y], [x+w, y+h]])
    
    return cv2.boundingRect(np.array(pts))

# Test on first frame of video
if __name__ == "__main__":
    video_path = "data/pick-rubiks-cube.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        bbox = detect_rubiks_cube_classical(frame)
        print(f"Detected BBox: {bbox}")
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite("test_detection.png", frame)
            print("Saved test_detection.png")
    else:
        print("Failed to read video")
