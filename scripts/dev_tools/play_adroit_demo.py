"""
Replay human demonstrations from Adroit to validate grasping works.
"""
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time

# Create environment with human rendering
env = gym.make('AdroitHandRelocate-v1', render_mode='human')

# Get the demo data that comes with the environment
# The demos are stored in the environment's data directory
import os
from gymnasium_robotics.envs.adroit_hand import adroit_relocate

# Find the demo file
env_module_path = os.path.dirname(adroit_relocate.__file__)
demo_path = os.path.join(env_module_path, '..', '..', 'assets', 'adroit_hand', 'relocate-v0.pickle')

print(f"Looking for demo at: {demo_path}")

if os.path.exists(demo_path):
    import pickle
    with open(demo_path, 'rb') as f:
        demo_data = pickle.load(f)

    print(f"Loaded demo with {len(demo_data)} trajectories")

    # Play first trajectory
    traj = demo_data[0]
    print(f"Trajectory keys: {traj.keys() if isinstance(traj, dict) else 'list'}")

    if isinstance(traj, dict) and 'observations' in traj:
        obs = traj['observations']
        actions = traj['actions']
        print(f"Playing trajectory with {len(actions)} steps")

        env.reset()
        for i, action in enumerate(actions):
            env.step(action)
            time.sleep(0.03)
    else:
        print("Demo format not recognized, trying alternative...")

else:
    print("Demo file not found. Trying alternative approach...")
    print("\nRunning with random actions to show the environment:")

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Just show the environment running
    for i in range(200):
        # Use zero actions to keep hand still, observe the setup
        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.02)

env.close()
