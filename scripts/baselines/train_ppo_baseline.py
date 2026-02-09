"""
Train PPO on the original AdroitHandRelocate-v1 environment.
No modifications - just the original dense reward.

This validates the base environment works before adding video guidance.
"""
import os
import argparse
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


class MetricsCallback(BaseCallback):
    """Track and plot training metrics."""

    def __init__(self, plot_dir: str, plot_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.plot_dir = plot_dir
        self.plot_freq = plot_freq
        os.makedirs(plot_dir, exist_ok=True)

        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.timesteps_log = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps_log.append(self.num_timesteps)
                self.successes.append(1 if info.get("success", False) else 0)

        if self.n_calls % 2000 == 0 and self.verbose > 0 and self.episode_rewards:
            recent = self.episode_rewards[-100:]
            succ = self.successes[-100:] if self.successes else [0]
            print(f"Step {self.num_timesteps:,}: reward={np.mean(recent):.1f}, success={np.mean(succ):.1%}")

        if self.n_calls % self.plot_freq == 0 and len(self.episode_rewards) > 10:
            self._save_plots()

        return True

    def _save_plots(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Original Relocate Training - {self.num_timesteps:,} steps', fontsize=14)

        # Rewards
        ax = axes[0, 0]
        ax.plot(self.timesteps_log, self.episode_rewards, alpha=0.3, color='blue')
        if len(self.episode_rewards) > 50:
            w = 50
            smoothed = np.convolve(self.episode_rewards, np.ones(w)/w, mode='valid')
            ax.plot(self.timesteps_log[w-1:], smoothed, color='blue', linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Episode Rewards')
        ax.grid(True, alpha=0.3)

        # Success rate
        ax = axes[0, 1]
        if len(self.successes) > 100:
            w = 100
            rates = [np.mean(self.successes[i-w:i]) for i in range(w, len(self.successes))]
            ax.plot(self.timesteps_log[w:], rates, color='green', linewidth=2)
            ax.fill_between(self.timesteps_log[w:], 0, rates, alpha=0.3, color='green')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate (rolling 100 eps)')
        ax.grid(True, alpha=0.3)

        # Episode length
        ax = axes[1, 0]
        ax.plot(self.timesteps_log, self.episode_lengths, alpha=0.3, color='orange')
        if len(self.episode_lengths) > 50:
            w = 50
            smoothed = np.convolve(self.episode_lengths, np.ones(w)/w, mode='valid')
            ax.plot(self.timesteps_log[w-1:], smoothed, color='orange', linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)

        # Histogram
        ax = axes[1, 1]
        recent = self.episode_rewards[-200:] if len(self.episode_rewards) > 200 else self.episode_rewards
        ax.hist(recent, bins=30, color='purple', alpha=0.7)
        ax.axvline(np.mean(recent), color='red', linestyle='--', label=f'Mean: {np.mean(recent):.1f}')
        ax.set_xlabel('Reward')
        ax.set_title('Recent Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training.png'), dpi=150)
        plt.close()


def make_env():
    """Create the original relocate environment."""
    def _init():
        env = gym.make("AdroitHandRelocate-v1")
        env = Monitor(env)
        return env
    return _init


def train(args):
    print("=" * 60)
    print("Training PPO on Original AdroitHandRelocate-v1")
    print("=" * 60)
    print(f"\nThis uses the original dense reward from the paper:")
    print("  - get_to_ball: -0.1 * dist(palm, ball)")
    print("  - ball_off_table: +1.0 if z > 0.04")
    print("  - make_hand_go_to_target: -0.5 * dist(palm, target)")
    print("  - make_ball_go_to_target: -0.5 * dist(ball, target)")
    print("  - ball_close_to_target: +10 if < 0.1m, +20 if < 0.05m")
    print()

    # Create envs
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env()])

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Model
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(args.output_dir, "tensorboard"),
        device=args.device,
    )

    # Callbacks
    callbacks = [
        MetricsCallback(os.path.join(args.output_dir, "plots"), plot_freq=10000, verbose=1),
        CheckpointCallback(save_freq=25000, save_path=os.path.join(args.output_dir, "checkpoints"), name_prefix="ppo"),
    ]

    print(f"Training for {args.timesteps:,} timesteps...")
    print(f"Output: {args.output_dir}")
    print()

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)

    model.save(os.path.join(args.output_dir, "model_final"))
    env.save(os.path.join(args.output_dir, "vec_normalize.pkl"))
    print(f"\nDone! Model saved to {args.output_dir}/model_final.zip")


def evaluate(args):
    """Evaluate a trained model."""
    import time

    model = PPO.load(args.model)
    env = gym.make("AdroitHandRelocate-v1", render_mode="human")

    successes = 0
    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            time.sleep(0.02)

        success = info.get("success", False)
        successes += int(success)
        print(f"Episode {ep+1}: reward={total_reward:.1f}, success={success}")

    print(f"\nSuccess rate: {successes}/{args.episodes} = {successes/args.episodes:.1%}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Train
    p = subparsers.add_parser("train")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="./runs/original/")

    # Eval
    p = subparsers.add_parser("eval")
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True) if hasattr(args, 'output_dir') else None

    if args.cmd == "train":
        train(args)
    else:
        evaluate(args)
