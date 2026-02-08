try:
    import gymnasium as gym
    import gymnasium_robotics
    import stable_baselines3
    import torch
    print("SUCCESS: RL packages available")
except ImportError as e:
    print(f"FAILURE: Missing package: {e}")
