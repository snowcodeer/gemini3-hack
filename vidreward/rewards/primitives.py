import numpy as np
from typing import Dict, Any

class BaseReward:
    def __call__(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        raise NotImplementedError

class ReachReward(BaseReward):
    """
    Reward for minimizing distance between end-effector and object.
    """
    def __init__(self, weight: float = 1.0, scale: float = 10.0):
        self.weight = weight
        self.scale = scale

    def __call__(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        # Expected keys from Adroit info/state
        dist = info.get("hand_to_obj_dist", 1.0)
        return self.weight * np.exp(-self.scale * dist)

class GraspReward(BaseReward):
    """
    Reward for establishing stable contact and proper finger enclosure.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def __call__(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        # Adroit envs usually provide 'has_grasped' or contact info
        is_grasped = info.get("is_grasped", False)
        reward = 1.0 if is_grasped else 0.0
        
        # Add finger spread bonus (smaller spread = tighter grip)
        spread = info.get("finger_spread", 1.0)
        reward += np.exp(-5.0 * spread)
        
        return self.weight * reward

class LiftReward(BaseReward):
    """
    Reward for moving the object towards a target height or location.
    """
    def __init__(self, weight: float = 1.0, scale: float = 5.0):
        self.weight = weight
        self.scale = scale

    def __call__(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        obj_to_target = info.get("obj_to_target_dist", 1.0)
        return self.weight * np.exp(-self.scale * obj_to_target)

class JointTrackingReward(BaseReward):
    """
    Reward for tracking a reference joint trajectory.
    """
    def __init__(self, weight: float = 1.0, scale: float = 1.0):
        self.weight = weight
        self.scale = scale

    def __call__(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        # qpos: (28,) for Adroit
        current_qpos = state.get("qpos", np.zeros(28))
        # reference_qpos should be provided in info for the current timestep
        ref_qpos = info.get("ref_qpos", current_qpos)
        
        # Compute joint-wise error
        error = np.linalg.norm(current_qpos - ref_qpos)
        return self.weight * np.exp(-self.scale * error)

class SmoothnessPenalty(BaseReward):
    """
    Penalty for high accelerations or jerky movements.
    """
    def __init__(self, weight: float = -0.01):
        self.weight = weight

    def __call__(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        v = state.get("qvel", np.zeros(28))
        return self.weight * np.sum(np.square(v))
