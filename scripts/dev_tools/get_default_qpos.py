import gymnasium as gym
import gymnasium_robotics
import numpy as np

env = gym.make("AdroitHandRelocate-v1")
env.reset()
print(f"Default QPos[0:3] (Root Pos): {env.unwrapped.data.qpos[0:3]}")
print(f"Default QPos[3:6] (Root Rot): {env.unwrapped.data.qpos[3:6]}")
