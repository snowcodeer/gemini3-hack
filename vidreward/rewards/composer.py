import numpy as np
from typing import Dict, Any, List, Optional
from .primitives import BaseReward

class RewardComposer:
    """
    Combines multiple reward primitives with weights that can change based on the task phase.
    """
    def __init__(self):
        self.primitives: Dict[str, BaseReward] = {}
        # Default weights for each phase
        self.phase_weights: Dict[str, Dict[str, float]] = {
            "IDLE": {"reach": 1.0, "smoothness": 1.0},
            "APPROACH": {"reach": 10.0, "smoothness": 1.0},
            "GRASP": {"grasp": 10.0, "reach": 5.0, "smoothness": 1.0},
            "TRANSPORT": {"lift": 10.0, "grasp": 5.0, "smoothness": 1.0},
            "RELEASE": {"grasp": -5.0, "smoothness": 1.0},
            "RETREAT": {"reach": -5.0, "smoothness": 1.0}
        }

    def add_primitive(self, name: str, primitive: BaseReward):
        self.primitives[name] = primitive

    def compute_reward(self, state: Dict[str, Any], info: Dict[str, Any], phase: str = "IDLE") -> float:
        """
        Compute the weighted sum of rewards based on the current phase.
        """
        total_reward = 0.0
        weights = self.phase_weights.get(phase, self.phase_weights["IDLE"])
        
        for name, weight in weights.items():
            if name in self.primitives:
                total_reward += weight * self.primitives[name](state, info)
        
        return total_reward
