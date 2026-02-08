import gymnasium as gym
import gymnasium_robotics
import time
import argparse

def view_adroit(env_id: str = "AdroitHandRelocate-v1", num_episodes: int = 5):
    """
    Load an Adroit environment and render it using the MuJoCo 'human' mode.
    """
    print(f"Loading environment: {env_id}")
    # 'human' render mode opens the MuJoCo viewer window
    env = gym.make(env_id, render_mode="human")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        print(f"Starting Episode {episode + 1}")
        
        step = 0
        while not (terminated or truncated):
            # Take a random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Slow down slightly so it's viewable
            time.sleep(0.01)
            step += 1
            
            if step % 100 == 0:
                print(f"Step {step}")
        
        print(f"Episode {episode + 1} finished after {step} steps.")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="AdroitHandRelocate-v1", 
                        help="Environment ID (e.g., AdroitHandDoor-v1, AdroitHandHammer-v1, AdroitHandRelocate-v1)")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    
    view_adroit(args.env, args.episodes)
