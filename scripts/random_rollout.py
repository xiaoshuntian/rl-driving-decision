"""Smoke test: load env from a config, run N random episodes, print stats.

Use this to confirm the env is correctly wired before launching long training.

Usage:
    python scripts/random_rollout.py --config configs/ppo_highway.yaml --episodes 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so `src.*` imports work without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.envs import make_highway_env
from src.utils import load_config, set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["experiment"]["seed"])

    env = make_highway_env(
        env_id=cfg["env"]["id"],
        config=cfg["env"]["config"],
        render_mode="human" if args.render else None,
        seed=cfg["experiment"]["seed"],
    )

    print(f"obs_space: {env.observation_space}")
    print(f"act_space: {env.action_space}")

    returns, lengths = [], []
    for ep in range(args.episodes):
        obs, _info = env.reset(seed=cfg["experiment"]["seed"] + ep)
        ep_return, ep_len, done = 0.0, 0, False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            done = terminated or truncated
        returns.append(ep_return)
        lengths.append(ep_len)
        print(f"  episode {ep + 1}: return={ep_return:.2f}, length={ep_len}")

    env.close()
    print(
        f"\nrandom-policy stats over {args.episodes} eps: "
        f"return={np.mean(returns):.2f}±{np.std(returns):.2f}, "
        f"length={np.mean(lengths):.1f}"
    )


if __name__ == "__main__":
    main()
