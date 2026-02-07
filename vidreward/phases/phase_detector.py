from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .phase_features import compute_hand_object_distance, compute_finger_spread, compute_velocity
from ..extraction.trajectory import HandTrajectory, ObjectTrajectory

@dataclass
class PhaseSegment:
    label: str
    start_frame: int
    end_frame: int
    spatial_params: dict = None

class PhaseDetector:
    """
    Detects manipulation phases using heuristic thresholds and temporal consistency.
    """
    def __init__(self, fps: float = 30.0, consistency_frames: int = 5):
        self.fps = fps
        self.consistency_frames = consistency_frames
        
        # Thresholds (normalized units)
        self.approach_dist_threshold = 0.15
        self.grasp_spread_threshold = 0.04
        self.transport_velocity_threshold = 0.08

    def detect_phases(self, hand_traj: HandTrajectory, obj_traj: ObjectTrajectory) -> List[PhaseSegment]:
        # Use smoothed features
        distances = compute_hand_object_distance(hand_traj, obj_traj, smooth=True)
        spreads = compute_finger_spread(hand_traj, smooth=True)
        obj_velocities = compute_velocity(obj_traj.centroids, self.fps)
        obj_speed = np.linalg.norm(obj_velocities, axis=1)
        
        num_frames = len(distances)
        phases = []
        
        current_phase = "IDLE"
        start_idx = 0
        
        # For temporal consistency
        potential_phase = current_phase
        potential_count = 0
        
        for i in range(num_frames):
            dist = distances[i]
            spread = spreads[i]
            speed = obj_speed[i]
            
            target_phase = current_phase
            
            # State machine logic
            if current_phase == "IDLE":
                if dist < 0.4: target_phase = "APPROACH"
            elif current_phase == "APPROACH":
                if dist < self.approach_dist_threshold: target_phase = "PRE_GRASP"
            elif current_phase == "PRE_GRASP":
                if spread < self.grasp_spread_threshold: target_phase = "GRASP"
            elif current_phase == "GRASP":
                if speed > self.transport_velocity_threshold: target_phase = "TRANSPORT"
            elif current_phase == "TRANSPORT":
                if speed < 0.02 and dist < self.approach_dist_threshold: target_phase = "PLACE"
                elif spread > self.grasp_spread_threshold * 1.5: target_phase = "RELEASE"
            elif current_phase == "PLACE":
                if spread > self.grasp_spread_threshold * 1.5: target_phase = "RELEASE"
            elif current_phase == "RELEASE":
                if dist > 0.3: target_phase = "RETREAT"
            
            # Apply temporal consistency
            if target_phase != current_phase:
                if target_phase == potential_phase:
                    potential_count += 1
                else:
                    potential_phase = target_phase
                    potential_count = 1
                
                if potential_count >= self.consistency_frames:
                    # Commit to new phase
                    phases.append(PhaseSegment(current_phase, start_idx, i))
                    current_phase = target_phase
                    start_idx = i
                    potential_count = 0
            else:
                potential_count = 0
                
        # Append the final phase
        if start_idx < num_frames:
            phases.append(PhaseSegment(current_phase, start_idx, num_frames - 1))
                
        return phases
