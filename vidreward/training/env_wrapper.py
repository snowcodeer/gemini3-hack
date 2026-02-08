import gymnasium as gym
import numpy as np
from typing import Dict, Any, Callable, Optional

class AdroitRewardWrapper(gym.Wrapper):
    """
    Wrapper for Adroit environments that replaces the default reward 
    with a custom function generated from a video.
    """
    def __init__(self, env: gym.Env, reward_fn: Callable):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.current_step = 0
        self.weights: Optional[Dict[str, Dict[str, float]]] = None

    def set_weights(self, weights: Dict[str, Dict[str, float]]):
        self.weights = weights

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Prepare state/info for our custom reward function
        # Adroit's qpos/qvel are usually in the obs or can be extracted from the simulated model
        state = {
            "qpos": self.env.unwrapped.data.qpos.copy(),
            "qvel": self.env.unwrapped.data.qvel.copy()
        }
        
        # Enrich info with distance features if not present
        # Note: Key names should match what reward primitives expect
        if "hand_to_obj_dist" not in info:
            # Placeholder calculation (should use MJCF site distance)
            info["hand_to_obj_dist"] = np.linalg.norm(state["qpos"][:3]) # Dummy
        
        custom_reward = self.reward_fn(state, info, self.current_step)
        
        self.current_step += 1
        return obs, custom_reward, terminated, truncated, info
