from typing import Dict, Any, List
from .env_wrapper import AdroitRewardWrapper

class PhaseCurriculum:
    """
    Manages training stages by adjusting reward weights or phase focus.
    """
    def __init__(self, stages: List[Dict[str, Any]]):
        self.stages = stages
        self.current_stage_idx = 0

    def get_current_weights(self) -> Dict[str, Dict[str, float]]:
        return self.stages[self.current_stage_idx]["weights"]

    def update(self, success_rate: float):
        """
        Advance to next stage if success rate exceeds threshold.
        """
        if self.current_stage_idx < len(self.stages) - 1:
            threshold = self.stages[self.current_stage_idx].get("threshold", 0.7)
            if success_rate > threshold:
                self.current_stage_idx += 1
                print(f"Advancing to Curriculum Stage {self.current_stage_idx}: {self.stages[self.current_stage_idx]['name']}")
                return True
        return False

# Example Curry Stages
DEFAULT_CURRICULUM = [
    {
        "name": "Phase 1: Approach",
        "weights": {"APPROACH": 10.0, "IDLE": 1.0},
        "threshold": 0.8
    },
    {
        "name": "Phase 2: Grasping",
        "weights": {"APPROACH": 2.0, "PRE_GRASP": 5.0, "GRASP": 10.0},
        "threshold": 0.5
    },
    {
        "name": "Phase 3: Full Task",
        "weights": "DEFAULT", # Uses RewardComposer defaults
        "threshold": 1.0
    }
]
