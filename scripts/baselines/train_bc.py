"""
Behavior Cloning Baseline for Adroit
M6 Validation Script

Trains a BC policy on the same video trajectory used for Residual RL.
Evaluates in the *exact same* GraspResidualEnv for fair comparison.
"""

import argparse
import os
import sys
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vidreward.training.bc_policy import BCTrainer, BCConfig, extract_features_from_sim_state
from scripts.train_grasp_residual import extract_grasp_pose, GraspResidualEnv, calibrate_grasp_pose

def train_bc(args):
    # 1. Extract Trajectory (Same as Residual RL)
    print(f"Extracting trajectory from {args.video}...")
    grasp_qpos, target_pos, transport_traj, joint_traj, grasp_frame, hand_traj, task_type = extract_grasp_pose(args.video)
    
    # Use transport_traj (grasp -> release) for relevant task learning
    # Align transport traj start with calibrated start (logic from train_grasp_residual)
    base_env = gym.make("AdroitHandRelocate-v1")
    calibrated_grasp_qpos = calibrate_grasp_pose(base_env, grasp_qpos)
    
    offset = calibrated_grasp_qpos[:3] - transport_traj[0, :3]
    transport_traj[:, :3] += offset
    
    # 2. Collect Data
    print("Collecting BC training data from Sim Replay...")
    config = BCConfig(
        epochs=args.epochs,
        augment_samples_per_frame=args.augment
    )
    trainer = BCTrainer(config)
    
    # We replay the transport_traj to get state-action pairs
    states, actions = trainer.collect_data_from_sim(base_env, transport_traj)
    print(f"Collected {len(states)} samples.")
    
    # 3. Train
    print("Training BC Policy...")
    policy = trainer.train(states, actions, verbose=True)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "bc_policy.pt")
    
    torch.save({
        'state_dict': policy.state_dict(),
        'state_dim': policy.state_dim,
        'action_dim': policy.action_dim,
        'state_mean': policy.state_mean.cpu().numpy(),
        'state_std': policy.state_std.cpu().numpy(),
        'action_scale': policy.action_scale.cpu().numpy(),
    }, save_path)
    print(f"Saved policy to {save_path}")
    
    # 4. Evaluate (Fair Comparison)
    print("Evaluating in GraspResidualEnv...")
    eval_env = gym.make("AdroitHandRelocate-v1", render_mode="rgb_array")
    
    # Initialize exactly like Residual RL
    env = GraspResidualEnv(
        eval_env,
        grasp_qpos=calibrated_grasp_qpos, # Use the calibrated one
        target_pos=target_pos,
        transport_traj=transport_traj,
        grasp_frame_idx=grasp_frame,
        max_steps=args.max_steps,
        residual_scale=0.0 # No residual! Pure policy control? 
        # Wait, GraspResidualEnv adds action * residual_scale to base policy.
        # But here we ARE the policy. We don't want a base policy + residual.
        # We want to Control the robot directly.
    )
    
    # We can't use GraspResidualEnv.step() normally because it enforces the residual dynamics.
    # However, for fair comparison of "Success Rate", we just want the Env dynamics (reset, rewards).
    # We will bypass the wrapper's step() logic for action application, 
    # OR we use the wrapper but pass action=0 and set the control manually?
    # Let's just use the underlying env for stepping, but use wrapper for Reset/Reward/Obs?
    # Actually, GraspResidualEnv.step() does: target_qpos = base_qpos + action * scale.
    # If we want to test BC, we want target_qpos = BC_Output.
    # So we should use the UNWRAPPED env for stepping, but use GraspResidualEnv to setup the episode.
    
    successes = []
    
    for ep in range(args.episodes):
        # Reset using wrapper logic (places object, calibrates hand)
        env.reset() 
        
        # Access internal
        model = env.unwrapped.model
        data = env.unwrapped.data
        
        done = False
        steps = 0
        success = False
        
        while not done and steps < args.max_steps:
            # 1. Get features
            features = extract_features_from_sim_state(model, data)
            
            # 2. Get action
            action_delta = policy.get_action(features)
            
            # 3. Apply action (Absolute position control)
            current_qpos = data.qpos[:30].copy()
            target_qpos = current_qpos + action_delta
            target_qpos = np.clip(target_qpos, -2.5, 2.5) # Bounds
            
            data.ctrl[:30] = target_qpos
            
            # Step sim
            import mujoco
            for _ in range(5):
                mujoco.mj_step(model, data)
                
            steps += 1
            
            # Check success (Using wrapper's logic would be nice, but we are bypassing step)
            # Re-implement simple check
            obj_pos = data.xpos[model.body("Object").id]
            dist = np.linalg.norm(obj_pos - target_pos)
            if dist < 0.1:
                success = True
                done = True # Stop on success?
                
        successes.append(1 if success else 0)
        print(f"Episode {ep+1}: Success={success}")
        
    print(f"Mean Success Rate: {np.mean(successes):.1%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="runs/baselines/bc")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--augment", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    
    args = parser.parse_args()
    train_bc(args)
