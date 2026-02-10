"""
Upload code files to existing runs so they display in the Code tab.
"""

import os
import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

data_dir = Path(__file__).parent.parent / "data"
runs_dir = Path(__file__).parent.parent / "runs"

# Map tasks to run groups
task_run_mapping = {
    'pick-3': ['pick-3'],
    'throw': ['throw', 'throw-experiments'],
    'grasp-ungrasp': ['grasp'],
}


def generate_config(task_name, task_type, milestones, analysis):
    grasp_frame = analysis.get("grasp_frame", 0)
    release_frame = analysis.get("release_frame", 100)

    milestone_info = "\n".join([
        f"#   - {m.get('label', 'unknown')}: frame {m.get('frame', '?')}"
        for m in milestones
    ])

    default_config = {
        "time_penalty": -0.1,
        "contact_scale": 0.1,
        "success_bonus": 500.0,
    }
    for m in milestones:
        label = m.get("label", "unknown").lower().replace(" ", "_")
        default_config[f"{label}_bonus"] = 10.0

    config_str = json.dumps(default_config, indent=4)
    milestone_frames = {m.get("label", "unknown"): m.get("frame", 0) for m in milestones}
    milestone_frames_str = json.dumps(milestone_frames, indent=4)

    return f'''"""
Configuration for: {task_name}
Task Type: {task_type}
"""

TASK_NAME = "{task_name}"
TASK_TYPE = "{task_type}"
GRASP_FRAME = {grasp_frame}
RELEASE_FRAME = {release_frame}

# Milestones:
{milestone_info}

MILESTONES = {milestone_frames_str}

REWARD_CONFIG = {config_str}

BC_CONFIG = {{
    "hidden_dims": (128, 128),
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "epochs": 1000,
    "early_stop_patience": 50,
    "augment_samples_per_frame": 50,
    "position_noise_std": 0.02,
    "joint_noise_std": 0.05,
    "temporal_blend": True,
}}

RESIDUAL_CONFIG = {{
    "initial_scale": 0.1,
    "final_scale": 0.5,
    "warmup_steps": 100_000,
    "penalty_weight": 0.1,
    "penalty_decay_rate": 0.99999,
    "min_penalty_weight": 0.01,
    "use_delta_actions": True,
    "include_base_in_obs": True,
}}

TRAINING_CONFIG = {{
    "algorithm": "TD3",
    "total_timesteps": 50_000,
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "policy_arch": [256, 256],
    "action_noise_sigma": 0.1,
    "max_episode_steps": 100,
}}
'''


def generate_baseline(task_name, task_type):
    return f'''"""
Behavioral Cloning Base Policy for: {task_name}
Task Type: {task_type}

Pipeline: Video -> MediaPipe -> Retarget -> BC Policy -> Residual RL
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from dataclasses import dataclass

from config import BC_CONFIG


@dataclass
class BCConfig:
    hidden_dims: Tuple[int, ...] = BC_CONFIG.get("hidden_dims", (128, 128))
    learning_rate: float = BC_CONFIG.get("learning_rate", 1e-3)
    batch_size: int = BC_CONFIG.get("batch_size", 256)
    epochs: int = BC_CONFIG.get("epochs", 1000)
    augment_samples_per_frame: int = BC_CONFIG.get("augment_samples_per_frame", 50)


def extract_features_from_sim(model, data) -> np.ndarray:
    """Extract relative features from MuJoCo state."""
    try:
        palm_pos = data.xpos[model.body("palm").id].copy()
    except:
        palm_pos = data.qpos[:3].copy()

    try:
        obj_pos = data.xpos[model.body("Object").id].copy()
    except:
        obj_pos = np.zeros(3)

    hand_to_obj = obj_pos - palm_pos
    finger_joints = data.qpos[6:30].copy()

    return np.concatenate([hand_to_obj, finger_joints])


class BCBasePolicy(nn.Module):
    """Behavioral Cloning policy network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(128, 128)):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.ReLU()])
            prev_dim = h
        layers.extend([nn.Linear(prev_dim, action_dim), nn.Tanh()])

        self.net = nn.Sequential(*layers)
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.register_buffer('action_scale', torch.ones(action_dim))

    def forward(self, state):
        state_norm = (state - self.state_mean) / (self.state_std + 1e-8)
        return self.net(state_norm) * self.action_scale

    def get_action(self, state: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            return self.forward(state_t).squeeze(0).numpy()
'''


def generate_residual(task_name, task_type):
    return f'''"""
Residual RL Environment Wrapper for: {task_name}
Task Type: {task_type}

action = base_policy(s) + alpha * residual_policy(s)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple

from config import RESIDUAL_CONFIG
from baseline import BCBasePolicy, extract_features_from_sim


class ResidualRLEnv(gym.Wrapper):
    """
    action = base_action + residual_scale * residual_action
    """

    def __init__(self, env, base_policy, residual_scale=0.1, residual_penalty_weight=0.1):
        super().__init__(env)
        self.base_policy = base_policy
        self.residual_scale = residual_scale
        self.residual_penalty_weight = residual_penalty_weight

    def step(self, residual_action):
        features = self._get_features()
        base_action = self.base_policy.get_action(features)

        scaled_residual = np.clip(
            residual_action * self.residual_scale,
            -self.residual_scale,
            self.residual_scale
        )

        current_qpos = self.env.unwrapped.data.qpos[:30].copy()
        final_action = np.clip(
            current_qpos + base_action + scaled_residual,
            self.env.action_space.low,
            self.env.action_space.high
        )

        obs, env_reward, terminated, truncated, info = self.env.step(final_action)
        residual_penalty = -self.residual_penalty_weight * np.sum(scaled_residual ** 2)

        return obs, env_reward + residual_penalty, terminated, truncated, info

    def _get_features(self):
        return extract_features_from_sim(self.env.unwrapped.model, self.env.unwrapped.data)

    def set_residual_scale(self, scale):
        self.residual_scale = scale


class ResidualCurriculum:
    """Curriculum learning for residual scale."""

    def __init__(self, env, initial_scale=0.1, final_scale=0.5, warmup_steps=100000):
        self.env = env
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.warmup_steps = warmup_steps
        self.num_steps = 0

    def on_step(self):
        self.num_steps += 1
        progress = min(1.0, self.num_steps / self.warmup_steps)
        new_scale = self.initial_scale + progress * (self.final_scale - self.initial_scale)
        self.env.set_residual_scale(new_scale)
        return {{"residual_scale": new_scale, "progress": progress}}
'''


def generate_reward(task_name, task_type, milestones):
    return f'''"""
Reward Function for: {task_name}
Task Type: {task_type}
"""

import numpy as np
from config import REWARD_CONFIG, MILESTONES


def compute_reward(env, obs, action, info):
    data = env.unwrapped.data
    model = env.unwrapped.model

    obj_pos = data.xpos[model.body("Object").id].copy()
    palm_pos = data.xpos[model.body("palm").id].copy()
    dist_to_target = np.linalg.norm(obj_pos - env.target_pos)
    n_contacts = data.ncon
    obj_height = obj_pos[2]

    reward = REWARD_CONFIG.get("time_penalty", -0.1)
    reward += min(n_contacts, 5) * REWARD_CONFIG.get("contact_scale", 0.1)

    completed = getattr(env, "_completed_milestones", set())

    for name, frame in MILESTONES.items():
        if name in completed:
            continue
        bonus = REWARD_CONFIG.get(f"{{name}}_bonus", 10.0)

        if "grasp" in name.lower() and n_contacts >= 3:
            reward += bonus
            completed.add(name)
        elif "lift" in name.lower() and obj_height > 0.1:
            reward += bonus
            completed.add(name)

    env._completed_milestones = completed

    success = dist_to_target < 0.1
    if success:
        reward += REWARD_CONFIG.get("success_bonus", 500.0)

    return reward, {{"success": success, "obj_height": obj_height, "n_contacts": n_contacts}}
'''


def generate_train(task_name, task_type):
    return f'''"""
Residual RL Training Script for: {task_name}
Task Type: {task_type}

Full Pipeline: Video -> MediaPipe -> BC -> Residual RL
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
from datetime import datetime

from config import TASK_NAME, REWARD_CONFIG, BC_CONFIG, RESIDUAL_CONFIG, TRAINING_CONFIG
from baseline import BCBasePolicy, BCConfig
from residual import ResidualRLEnv, ResidualCurriculum
from reward import compute_reward



def generate_eval(task_name, task_type, analysis):
    """Generate eval.py from analysis data."""
    eval_spec = analysis.get('eval_code', {})
    success_criteria = eval_spec.get('success_criteria', [
        {'name': 'is_grasped', 'description': 'Hand has grip on object', 'condition': 'n_contacts >= 3'},
        {'name': 'is_lifted', 'description': 'Object above table', 'condition': 'obj_height > 0.05'},
        {'name': 'is_at_target', 'description': 'Object near target', 'condition': 'dist_to_target < 0.1'}
    ])
    metrics = eval_spec.get('metrics', [
        {'name': 'distance_to_goal', 'description': 'Distance to target', 'formula': 'np.linalg.norm(obj_pos - target_pos)'},
        {'name': 'grasp_quality', 'description': 'Contact points', 'formula': 'min(n_contacts, 5)'},
        {'name': 'lift_height', 'description': 'Object height', 'formula': 'obj_pos[2]'}
    ])

    # Build success functions
    success_fns = ''
    success_list = []
    for c in success_criteria:
        name, desc, cond = c['name'], c['description'], c['condition']
        success_fns += f"""
def {name}(env) -> bool:
    """{desc}"""
    data = env.unwrapped.data
    model = env.unwrapped.model
    obj_pos = data.xpos[model.body('Object').id]
    n_contacts = data.ncon
    obj_height = obj_pos[2]
    target_pos = getattr(env, 'target_pos', np.array([0.3, 0.0, 0.2]))
    dist_to_target = np.linalg.norm(obj_pos - target_pos)
    return {cond}
"""
        success_list.append(f'("{name.replace("is_", "")}", {name})')

    # Build metric functions
    metric_fns = ''
    metric_list = []
    for m in metrics:
        name, desc, formula = m['name'], m['description'], m['formula']
        metric_fns += f"""
def compute_{name}(env) -> float:
    """{desc}"""
    data = env.unwrapped.data
    model = env.unwrapped.model
    obj_pos = data.xpos[model.body('Object').id]
    n_contacts = data.ncon
    target_pos = getattr(env, 'target_pos', np.array([0.3, 0.0, 0.2]))
    return float({formula})
"""
        metric_list.append(f'("{name}", compute_{name})')

    return f""""""
Evaluation Functions for: {task_name}
Task Type: {task_type}
Auto-generated
"""

import numpy as np
from typing import Dict, Tuple, Any

# ============ SUCCESS CRITERIA ============
{success_fns}

# ============ METRICS ============
{metric_fns}

# ============ EVALUATION SUITE ============

class EvalSuite:
    def __init__(self, env):
        self.env = env
        self.success_checks = [{', '.join(success_list)}]
        self.metric_fns = [{', '.join(metric_list)}]

    def check_success(self) -> Tuple[bool, Dict[str, bool]]:
        results = {{}}
        for name, fn in self.success_checks:
            try:
                results[name] = fn(self.env)
            except:
                results[name] = False
        return all(results.values()), results

    def compute_metrics(self) -> Dict[str, float]:
        metrics = {{}}
        for name, fn in self.metric_fns:
            try:
                metrics[name] = fn(self.env)
            except:
                metrics[name] = float('nan')
        return metrics

    def evaluate_step(self) -> Dict[str, Any]:
        success, milestone_progress = self.check_success()
        metrics = self.compute_metrics()
        return {{'success': success, 'metrics': metrics, 'milestone_progress': milestone_progress}}
"""


def main():
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.monitor import Monitor

    print(f"Training: {{TASK_NAME}}")

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / TASK_NAME / f"{{TASK_NAME}}_{{timestamp}}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = gym.make("AdroitHandRelocate-v1")
    env = Monitor(env, str(run_dir))

    # TD3 training
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=TRAINING_CONFIG["action_noise_sigma"] * np.ones(n_actions)
    )

    model = TD3(
        "MlpPolicy", env,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        buffer_size=TRAINING_CONFIG["buffer_size"],
        batch_size=TRAINING_CONFIG["batch_size"],
        action_noise=action_noise,
        verbose=1
    )

    model.learn(total_timesteps=TRAINING_CONFIG["total_timesteps"], progress_bar=True)
    model.save(run_dir / "td3_final")

    print(f"Saved to {{run_dir}}")
    env.close()


if __name__ == "__main__":
    main()
'''



def generate_eval(task_name, task_type, analysis):
    """Generate eval.py from analysis data."""
    eval_spec = analysis.get('eval_code', {})
    success_criteria = eval_spec.get('success_criteria', [
        {'name': 'is_grasped', 'description': 'Hand has grip on object', 'condition': 'n_contacts >= 3'},
        {'name': 'is_lifted', 'description': 'Object above table', 'condition': 'obj_height > 0.05'},
        {'name': 'is_at_target', 'description': 'Object near target', 'condition': 'dist_to_target < 0.1'}
    ])
    metrics = eval_spec.get('metrics', [
        {'name': 'distance_to_goal', 'description': 'Distance to target', 'formula': 'np.linalg.norm(obj_pos - target_pos)'},
        {'name': 'grasp_quality', 'description': 'Contact points', 'formula': 'min(n_contacts, 5)'},
        {'name': 'lift_height', 'description': 'Object height', 'formula': 'obj_pos[2]'}
    ])

    # Build success functions
    success_fns = ''
    success_list = []
    for c in success_criteria:
        name, desc, cond = c['name'], c['description'], c['condition']
        success_fns += f"""
def {name}(env) -> bool:
    """{desc}"""
    data = env.unwrapped.data
    model = env.unwrapped.model
    obj_pos = data.xpos[model.body('Object').id]
    n_contacts = data.ncon
    obj_height = obj_pos[2]
    target_pos = getattr(env, 'target_pos', np.array([0.3, 0.0, 0.2]))
    dist_to_target = np.linalg.norm(obj_pos - target_pos)
    return {cond}
"""
        success_list.append(f'("{name.replace("is_", "")}", {name})')

    # Build metric functions
    metric_fns = ''
    metric_list = []
    for m in metrics:
        name, desc, formula = m['name'], m['description'], m['formula']
        metric_fns += f"""
def compute_{name}(env) -> float:
    """{desc}"""
    data = env.unwrapped.data
    model = env.unwrapped.model
    obj_pos = data.xpos[model.body('Object').id]
    n_contacts = data.ncon
    target_pos = getattr(env, 'target_pos', np.array([0.3, 0.0, 0.2]))
    return float({formula})
"""
        metric_list.append(f'("{name}", compute_{name})')

    return f""""""
Evaluation Functions for: {task_name}
Task Type: {task_type}
Auto-generated
"""

import numpy as np
from typing import Dict, Tuple, Any

# ============ SUCCESS CRITERIA ============
{success_fns}

# ============ METRICS ============
{metric_fns}

# ============ EVALUATION SUITE ============

class EvalSuite:
    def __init__(self, env):
        self.env = env
        self.success_checks = [{', '.join(success_list)}]
        self.metric_fns = [{', '.join(metric_list)}]

    def check_success(self) -> Tuple[bool, Dict[str, bool]]:
        results = {{}}
        for name, fn in self.success_checks:
            try:
                results[name] = fn(self.env)
            except:
                results[name] = False
        return all(results.values()), results

    def compute_metrics(self) -> Dict[str, float]:
        metrics = {{}}
        for name, fn in self.metric_fns:
            try:
                metrics[name] = fn(self.env)
            except:
                metrics[name] = float('nan')
        return metrics

    def evaluate_step(self) -> Dict[str, Any]:
        success, milestone_progress = self.check_success()
        metrics = self.compute_metrics()
        return {{'success': success, 'metrics': metrics, 'milestone_progress': milestone_progress}}
"""


def main():
    for task_name, run_groups in task_run_mapping.items():
        print(f"\nGenerating code for {task_name}...")

        analysis_path = data_dir / task_name / "analysis.json"
        if not analysis_path.exists():
            print(f"  No analysis.json found")
            continue

        with open(analysis_path) as f:
            analysis = json.load(f)

        task_type = analysis.get("task_type", "pick")
        milestones = analysis.get("milestones", [])

        code_files = {
            "config.py": generate_config(task_name, task_type, milestones, analysis),
            "baseline.py": generate_baseline(task_name, task_type),
            "residual.py": generate_residual(task_name, task_type),
            "reward.py": generate_reward(task_name, task_type, milestones),
            "train.py": generate_train(task_name, task_type),
            "eval.py": generate_eval(task_name, task_type, analysis),
        }

        # Copy to all matching runs
        for group in run_groups:
            group_dir = runs_dir / group
            if not group_dir.exists():
                continue

            for run_dir in group_dir.iterdir():
                if run_dir.is_dir():
                    code_dir = run_dir / "code"
                    code_dir.mkdir(exist_ok=True)

                    for filename, content in code_files.items():
                        (code_dir / filename).write_text(content, encoding="utf-8")

                    print(f"  -> {group}/{run_dir.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
