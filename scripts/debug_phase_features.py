import numpy as np
import os
import argparse
from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
from vidreward.extraction.trajectory import ObjectTrajectory
from vidreward.phases.phase_features import compute_hand_object_distance, compute_finger_spread

def debug_features(video_path: str):
    print(f"Analyzing features for {video_path}...")
    tracker = MediaPipeTracker()
    hand_traj = tracker.process_video(video_path)
    tracker.close()
    
    # We don't have real object centroids, let's assume one is provided or use target folder
    # For now, let's assume the rubiks cube is near the center [0.5, 0.5] if it's a pick task
    # A better way is to see WHERE the hand is when it stops moving
    num_frames = len(hand_traj.landmarks)
    
    # Let's try to infer object center from high-confidence hand stillness frames
    # Or just use a range of mocks to see what thresholds look like
    mock_center = [0.5, 0.5] 
    obj_traj = ObjectTrajectory(np.tile(mock_center, (num_frames, 1)))
    
    distances = compute_hand_object_distance(hand_traj, obj_traj, smooth=True)
    spreads = compute_finger_spread(hand_traj, smooth=True)
    
    # Find the frame with minimum distance
    min_idx = np.argmin(distances)
    print(f"Distance stats: Min={np.min(distances):.4f}, Max={np.max(distances):.4f}, Mean={np.mean(distances):.4f}")
    print(f"Spread stats:   Min={np.min(spreads):.4f}, Max={np.max(spreads):.4f}, Mean={np.mean(spreads):.4f}")
    
    # Find potential interaction point (where hand is most still near middle of video)
    mid_start, mid_end = num_frames // 3, 2 * num_frames // 3
    palm_centers = (hand_traj.landmarks[:, 5, :2] + hand_traj.landmarks[:, 17, :2]) / 2.0
    
    # Simple heuristic: find where palm is closest to "still" in the middle
    velocities = np.linalg.norm(np.diff(palm_centers, axis=0), axis=1)
    min_vel_idx = np.argmin(velocities[mid_start:mid_end-1]) + mid_start
    interaction_point = palm_centers[min_vel_idx]
    
    print(f"Heuristic Interaction Point: {interaction_point}")
    print(f"Palm Pos at min interaction frame ({min_vel_idx}): {palm_centers[min_vel_idx]}")
    print(f"Palm Pos at absolute min dist frame ({min_idx}): {palm_centers[min_idx]}")
    
    # Absolute min spread
    min_spread_idx = np.argmin(spreads)
    print(f"Absolute min spread at frame {min_spread_idx}: {spreads[min_spread_idx]:.4f}")
    
    # Check if spread ever drops below threshold
    threshold = 0.06
    frames_below = np.where(spreads < threshold)[0]
    print(f"Frames with spread < {threshold}: {len(frames_below)} / {num_frames}")

if __name__ == "__main__":
    video = "data/pick-rubiks-cube.mp4"
    if os.path.exists(video):
        debug_features(video)
    else:
        print("Video not found.")
