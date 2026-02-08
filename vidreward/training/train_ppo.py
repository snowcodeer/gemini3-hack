import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from vidreward.training.env_wrapper import AdroitRewardWrapper
from vidreward.rewards.pipeline import create_default_reward_function
import os

from stable_baselines3.common.callbacks import BaseCallback
from vidreward.training.curriculum import PhaseCurriculum, DEFAULT_CURRICULUM

class CurriculumCallback(BaseCallback):
    """
    Callback for updating curriculum based on periodic evaluations.
    """
    def __init__(self, curriculum: PhaseCurriculum, verbose=0):
        super().__init__(verbose)
        self.curriculum = curriculum

    def _on_step(self) -> bool:
        # Periodically update weights in the environment
        if self.n_calls % 1000 == 0:
            weights = self.curriculum.get_current_weights()
            if weights != "DEFAULT":
                # Assuming single vec env for simplicity
                self.training_env.env_method("set_weights", weights)
        return True

def train_adroit(video_path: str, env_id: str = "AdroitHandDoor-v1", total_timesteps: int = 1000000):
    # 1. Setup Mock Centroids
    mock_centroids = np.zeros((2000, 2)) + 0.5 
    
    # 2. Create Reward Function
    reward_fn = create_default_reward_function(video_path, mock_centroids)
    
    # 3. Setup Curriculum
    curriculum = PhaseCurriculum(DEFAULT_CURRICULUM)
    
    # 4. Setup Environment
    def make_env():
        env = gym.make(env_id)
        env = AdroitRewardWrapper(env, reward_fn)
        return env
    
    env = DummyVecEnv([make_env])
    
    # 5. Initialize Model
    model = PPO("MlpPolicy", env, verbose=1)
    
    # 6. Train with Callback
    callback = CurriculumCallback(curriculum)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # 6. Save
    model_path = f"ppo_{env_id}_vidreward.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    import numpy as np
    video = "data/pick-rubiks-cube.mp4"
    if os.path.exists(video):
        train_adroit(video)
    else:
        print("Provide a valid video path for reward generation.")
