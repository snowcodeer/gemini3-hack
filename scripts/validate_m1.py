import os
import argparse
from typing import Optional, Tuple
import numpy as np
import cv2
from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
from vidreward.extraction.trajectory import HandTrajectory, ObjectTrajectory
from vidreward.phases.phase_detector import PhaseDetector
from vidreward.phases.phase_features import compute_hand_object_distance
from vidreward.phases.phase_grammar import PhaseGrammar
from vidreward.utils.video_io import VideoReader, VideoWriter
from vidreward.utils.visualization import draw_landmarks, plot_distances

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
        return cv2.boundingRect(np.array(square_boxes).reshape(-1, 4)) if square_boxes else None
        
    pts = []
    for x, y, w, h in cluster_boxes:
        pts.extend([[x, y], [x+w, y+h]])
    return cv2.boundingRect(np.array(pts))

def validate_m1(video_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(video_path).split('.')[0]
    
    # 1. Extraction
    print("Running extraction...")
    tracker = MediaPipeTracker()
    hand_traj = tracker.process_video(video_path)
    tracker.close()
    
    # 2. Real Object Tracking (Vision Model: Color Detection + CSRT)
    num_frames = len(hand_traj.landmarks)
    reader_temp = VideoReader(video_path)
    frames = list(reader_temp.read_frames())
    reader_temp.release()
    
    h, w = frames[0].shape[:2]
    obj_centroids = np.zeros((num_frames, 2))
    
    # Try to find object in early frames (before hand arrives)
    bbox = None
    init_frame = 0
    for i in range(min(50, num_frames)):
        bbox = detect_rubiks_cube_classical(frames[i])
        if bbox:
            init_frame = i
            print(f"Classical Vision Model detected Rubik's cube at frame {i}")
            break
            
    if bbox is None:
        print("Warning: Vision model could not find object in early frames. Using interaction heuristic.")
        palm_centers = (hand_traj.landmarks[:, 5, :2] + hand_traj.landmarks[:, 17, :2]) / 2.0
        safe_range = slice(num_frames//4, 3*num_frames//4)
        velocities = np.linalg.norm(np.diff(palm_centers, axis=0), axis=1)
        arrival_idx = np.argmin(velocities[safe_range]) + num_frames // 4
        interaction_pt = palm_centers[arrival_idx]
        init_frame = arrival_idx
        start_x = int(interaction_pt[0] * w) - 40
        start_y = int(interaction_pt[1] * h) - 40
        bbox = (start_x, start_y, 80, 80)
    
    # Tracking
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frames[init_frame], bbox)
    
    # Fill before initialization
    for i in range(init_frame):
        bx, by, bw, bh = bbox
        obj_centroids[i] = [ (bx + bw/2) / w, (by + bh/2) / h ]
        
    # Track forwards
    for i in range(init_frame, num_frames):
        success, box = tracker.update(frames[i])
        if success:
            x, y, bw, bh = box
            obj_centroids[i] = [ (x + bw/2) / w, (y + bh/2) / h ]
        else:
            # Re-detect if possible, otherwise freeze
            new_bbox = detect_rubiks_cube_classical(frames[i])
            if new_bbox:
                bx, by, bw, bh = new_bbox
                obj_centroids[i] = [ (bx + bw/2) / w, (by + bh/2) / h ]
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frames[i], new_bbox)
            else:
                obj_centroids[i] = obj_centroids[i-1]
            
    obj_traj = ObjectTrajectory(obj_centroids)
    
    # 3. Phase Detection (M1 Refined)
    print("Detecting phases (Refined)...")
    detector = PhaseDetector()
    phases = detector.detect_phases(hand_traj, obj_traj)
    print(f"Detected Phases: {[p.label for p in phases]}")
    
    # 4. Grammar Validation
    print("Validating phase grammar...")
    validation = PhaseGrammar.validate_sequence(phases)
    print(f"Grammar valid: {validation['is_complete']}")
    if validation['violations']:
        print(f"Violations: {validation['violations']}")
    
    # 5. Visualization
    print("Saving visualizations...")
    distances = compute_hand_object_distance(hand_traj, obj_traj, smooth=True)
    plot_distances(distances, os.path.join(output_dir, f"{basename}_m1_distance.png"))
    
    reader = VideoReader(video_path)
    output_video = os.path.join(output_dir, f"{basename}_m1_validated.mp4")
    
    with VideoWriter(output_video, reader.fps, reader.width, reader.height) as writer:
        for i, frame in enumerate(reader.read_frames()):
            frame = draw_landmarks(frame, hand_traj.landmarks[i], hand_traj.confidences[i])
            
            # Determine current phase
            current_phase = "IDLE"
            for p in phases:
                if p.start_frame <= i <= p.end_frame:
                    current_phase = p.label
                    break
            
            # Draw mock object recognition overlay
            obj_x, obj_y = int(obj_traj.centroids[i, 0] * reader.width), int(obj_traj.centroids[i, 1] * reader.height)
            cv2.rectangle(frame, (obj_x - 30, obj_y - 30), (obj_x + 30, obj_y + 30), (0, 255, 0), 2)
            cv2.putText(frame, "Object", (obj_x - 30, obj_y - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw phase and distance info
            cv2.putText(frame, f"Phase: {current_phase}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Dist: {distances[i]:.3f}", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            writer.write_frame(frame)
    
    print(f"M1 Validation complete! Check {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out", type=str, default="validation_results_m1")
    args = parser.parse_args()
    
    validate_m1(args.video, args.out)
