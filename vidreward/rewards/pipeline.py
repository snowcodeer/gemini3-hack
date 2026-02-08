import numpy as np
import cv2
from typing import List, Dict, Any, Callable

from ..extraction.mediapipe_tracker import MediaPipeTracker
from ..extraction.trajectory import HandTrajectory, ObjectTrajectory
from ..phases.phase_detector import PhaseDetector
from ..retargeting.landmarks_to_angles import AdroitRetargeter
from .composer import RewardComposer
from .primitives import ReachReward, GraspReward, LiftReward, JointTrackingReward, SmoothnessPenalty

def create_default_reward_function(video_path: str, obj_centroids: np.ndarray) -> Callable:
    """
    End-to-end pipeline: Video -> Phases + Reference Trajectory -> Reward Function.
    """
    # 1. Extract Trajectory
    tracker = MediaPipeTracker()
    hand_traj = tracker.process_video(video_path)
    tracker.close()
    
    obj_traj = ObjectTrajectory(obj_centroids)
    
    # 2. Detect Phases
    detector = PhaseDetector()
    phases = detector.detect_phases(hand_traj, obj_traj)
    
    # 3. Retarget to Reference Joint Angles
    retargeter = AdroitRetargeter()
    ref_qpos_seq = retargeter.retarget_sequence(hand_traj.landmarks)
    
    # 4. Compose Reward
    composer = RewardComposer()
    composer.add_primitive("reach", ReachReward())
    composer.add_primitive("grasp", GraspReward())
    composer.add_primitive("lift", LiftReward())
    composer.add_primitive("tracking", JointTrackingReward())
    composer.add_primitive("smoothness", SmoothnessPenalty())
    
    def reward_fn(state: Dict[str, Any], info: Dict[str, Any], step_idx: int) -> float:
        """
        The generated reward function to be used in the RL loop.
        """
        # Determine phase based on step_idx (mapping step_idx to video frame)
        # Note: This assumes 1:1 mapping or requires interpolation
        current_phase = "IDLE"
        for p in phases:
            if p.start_frame <= step_idx <= p.end_frame:
                current_phase = p.label
                break
        
        # Add reference qpos to info for tracking reward
        if step_idx < len(ref_qpos_seq):
            info["ref_qpos"] = ref_qpos_seq[step_idx]
        
        return composer.compute_reward(state, info, current_phase)
    
    return reward_fn
