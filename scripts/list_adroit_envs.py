import gymnasium as gym
import gymnasium_robotics
envs = [e for e in gym.envs.registry.keys() if "Adroit" in e]
print("Available Adroit environments:")
for e in sorted(envs):
    print(f"  {e}")
