"""Evaluate a trained model on highway-env and report success/return stats.

Usage:
    python scripts/eval.py --config configs/ppo_highway.yaml \
        --model runs/ppo_highway_fast/final_model.zip --episodes 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import PPO, SAC

from src.envs import make_highway_env
from src.utils import load_config, set_global_seed

ALGO_LOADERS = {"ppo": PPO, "sac": SAC}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, help="ppo | sac")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["experiment"]["seed"] + 99)

    env = make_highway_env(
        env_id=cfg["env"]["id"],
        config=cfg["env"]["config"],
        render_mode="human" if args.render else None,
    )
    model = ALGO_LOADERS[args.algo].load(args.model, env=env)

    returns, lengths, crashes = [], [], 0
    for ep in range(args.episodes):
        obs, _info = env.reset(seed=cfg["experiment"]["seed"] + 1000 + ep)
        ep_return, ep_len, done = 0.0, 0, False
        crashed = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            done = terminated or truncated
            if info.get("crashed", False):
                crashed = True
        returns.append(ep_return)
        lengths.append(ep_len)
        crashes += int(crashed)
        print(
            f"  ep {ep + 1}: return={ep_return:.2f}, len={ep_len}, "
            f"crashed={crashed}"
        )

    env.close()
    print(
        f"\nresults over {args.episodes} eps: "
        f"return={np.mean(returns):.2f}±{np.std(returns):.2f}, "
        f"length={np.mean(lengths):.1f}, "
        f"crash_rate={crashes / args.episodes:.1%}"
    )


if __name__ == "__main__":
    main()
