import os
import argparse
import numpy as np
import cv2
from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
from vidreward.extraction.trajectory import HandTrajectory, ObjectTrajectory
from vidreward.phases.phase_detector import PhaseDetector
from vidreward.phases.phase_features import compute_hand_object_distance
from vidreward.phases.phase_grammar import PhaseGrammar
from vidreward.utils.video_io import VideoReader, VideoWriter
from vidreward.utils.visualization import draw_landmarks, plot_distances

def validate_m1(video_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(video_path).split('.')[0]
    
    # 1. Extraction
    print("Running extraction...")
    tracker = MediaPipeTracker()
    hand_traj = tracker.process_video(video_path)
    tracker.close()
    
    # 2. Mock Object Tracking
    num_frames = len(hand_traj.landmarks)
    obj_centroids = np.zeros((num_frames, 2))
    # Mock some movement for TRANSPORT
    for i in range(num_frames):
        if i < num_frames // 2:
            obj_centroids[i] = [0.5, 0.5]
        else:
            obj_centroids[i] = [0.5 + 0.1 * (i - num_frames//2) / (num_frames//2), 0.5]
    obj_traj = ObjectTrajectory(obj_centroids)
    
    # 3. Phase Detection (M1 Refined)
    print("Detecting phases (Refined)...")
    detector = PhaseDetector(consistency_frames=5)
    phases = detector.detect_phases(hand_traj, obj_traj)
    
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
            
            current_phase = "IDLE"
            for p in phases:
                if p.start_frame <= i <= p.end_frame:
                    current_phase = p.label
                    break
            
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
