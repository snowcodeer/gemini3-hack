import gymnasium as gym
import gymnasium_robotics
import numpy as np

try:
    env = gym.make("AdroitHandRelocate-v1")
    print("Action Space:", env.action_space)
    
    # Access MuJoCo model
    # Gymnasium Robotics usually wraps MuJoCoEnv
    if hasattr(env.unwrapped, 'model'):
        model = env.unwrapped.model
        print("Number of joints:", model.njnt)
        names = [model.joint(i).name for i in range(model.njnt)]
        print("Joint names:", names)
        
        act_names = [model.actuator(i).name for i in range(model.nu)]
        with open("adroit_actuators.txt", "w") as f:
            for name in act_names:
                f.write(f"{name}\n")
        print(f"Saved {len(act_names)} actuator names to adroit_actuators.txt")
    else:
        print("Could not access model.")

except Exception as e:
    print(f"Error: {e}")
