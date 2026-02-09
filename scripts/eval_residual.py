"""
Evaluate and Visualize Residual RL Policy
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import argparse
import os
import pickle
import cv2
import sys
from pathlib import Path
from stable_baselines3 import TD3

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_grasp_residual import GraspResidualEnv, calibrate_grasp_pose

def evaluate(args):
    print(f"Loading run: {args.run_dir}")
    
    # Load config
    config_path = os.path.join(args.run_dir, "config.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)
        
    # Handle missing keys in older configs
    video_name = config.get('video_name')
    if not video_name:
        if args.video:
            video_name = Path(args.video).stem
        else:
            # Infer from run_dir name (format: video_timestamp)
            run_name = Path(args.run_dir).name
            video_name = run_name.rsplit('_', 1)[0]
    
    video_path = config.get('video_path')
    if not video_path:
        if args.video:
            video_path = args.video
        else:
            # Guess path
            video_path = f"data/{video_name}.mp4"
            
    print(f"Video Name: {video_name}")
    print(f"Video Path: {video_path}")
    
    # Create Env
    base_env = gym.make("AdroitHandRelocate-v1", render_mode="rgb_array")
    
    # We must use the EXACT SAME grasp_qpos and target_pos as training
    env = GraspResidualEnv(
        base_env,
        grasp_qpos=config['grasp_qpos'],
        target_pos=config['target_pos'],
        transport_traj=config['transport_traj'],
        residual_scale=config['residual_scale'],
        max_steps=200 # Longer for eval
    )
    
    # Re-extract hand_traj if missing
    if 'hand_traj' not in config:
        print("Re-extracting hand trajectory for observation...")
        from vidreward.utils.video_io import VideoReader
        from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
        
        if not os.path.exists(video_path):
             raise FileNotFoundError(f"Video not found at {video_path}. Please provide --video argument.")

        reader = VideoReader(video_path)
        frames = list(reader.read_frames())
        tracker = MediaPipeTracker()
        hand_traj = tracker.process_frames(frames)
        tracker.close()
        env.hand_traj = hand_traj
    else:
        env.hand_traj = config['hand_traj']

    # Load Model
    model_path = os.path.join(args.run_dir, "td3_final.zip")
    if not os.path.exists(model_path):
        # unexpected, maybe check checkpoints?
        checkpoints = sorted([f for f in os.listdir(os.path.join(args.run_dir, "checkpoints")) if f.endswith(".zip")])
        if checkpoints:
            model_path = os.path.join(args.run_dir, "checkpoints", checkpoints[-1])
            print(f"Loading checkpoint: {model_path}")
        else:
            print("No model found!")
            return

    model = TD3.load(model_path)
    
    # Run Episodes
    frames = []
    
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        # Initial Render
        frames.append(base_env.render())
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            frames.append(base_env.render())
            
        print(f"Ep {ep+1}: Reward={total_reward:.1f}, Success={info.get('success', False)}, Lifted={info.get('lifted', False)}")
        
    env.close()
    
    # Save Video
    output_path = os.path.join(args.run_dir, "eval_video.mp4")
    print(f"Saving video to {output_path}...")
    
    height, width, layers = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--video", type=str, help="Path to video (if not in config)")
    
    args = parser.parse_args()
    evaluate(args)
