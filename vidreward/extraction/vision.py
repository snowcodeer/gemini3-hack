import cv2
import numpy as np
from typing import Optional, Tuple

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
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if 100 < area < 10000:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.7 <= aspect_ratio <= 1.3:
                    square_boxes.append((x, y, w, h))
    
    if not square_boxes:
        return None

    # Filter for clusters of squares (Rubik's cube signature)
    cluster_boxes = []
    for i, b1 in enumerate(square_boxes):
        neighbors = 0
        c1 = (b1[0] + b1[2]/2, b1[1] + b1[3]/2)
        for j, b2 in enumerate(square_boxes):
            if i == j: continue
            c2 = (b2[0] + b2[2]/2, b2[1] + b2[3]/2)
            dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
            if dist < max(b1[2], b1[3]) * 4:
                neighbors += 1
        if neighbors >= 2:
            cluster_boxes.append(b1)
            
    if not cluster_boxes:
        # No clusters found, use all square boxes
        if not square_boxes:
            return None
        pts = []
        for x, y, w, h in square_boxes:
            pts.extend([[x, y], [x+w, y+h]])
        return cv2.boundingRect(np.array(pts))
        
    pts = []
    for x, y, w, h in cluster_boxes:
        pts.extend([[x, y], [x+w, y+h]])
    return cv2.boundingRect(np.array(pts))
