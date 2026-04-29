"""Train an RL baseline on highway-env using a YAML config.

Usage:
    python scripts/train.py --algo ppo --config configs/ppo_highway.yaml
    python scripts/train.py --algo sac --config configs/sac_highway.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecMonitor

from src.algos import build_model
from src.envs import make_vec_highway_env
from src.utils import ensure_dir, load_config, set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, help="ppo | sac")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override config train.total_timesteps (e.g. 5000 for a smoke run)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["experiment"]["seed"])

    out_dir = ensure_dir(cfg["experiment"]["output_dir"])
    tb_dir = ensure_dir(out_dir / "tb")
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    eval_dir = ensure_dir(out_dir / "eval")

    train_env = VecMonitor(
        make_vec_highway_env(
            env_id=cfg["env"]["id"],
            n_envs=cfg["env"]["n_envs"],
            config=cfg["env"]["config"],
            seed=cfg["experiment"]["seed"],
        )
    )
    eval_env = VecMonitor(
        make_vec_highway_env(
            env_id=cfg["env"]["id"],
            n_envs=1,
            config=cfg["env"]["config"],
            seed=cfg["experiment"]["seed"] + 10_000,
        )
    )

    model = build_model(
        algo_name=args.algo,
        env=train_env,
        algo_kwargs=cfg["algo"],
        tensorboard_log=str(tb_dir),
        seed=cfg["experiment"]["seed"],
    )

    train_cfg = cfg["train"]
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(eval_dir),
            log_path=str(eval_dir),
            eval_freq=train_cfg["eval_freq"],
            n_eval_episodes=train_cfg["n_eval_episodes"],
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=train_cfg["save_freq"],
            save_path=str(ckpt_dir),
            name_prefix=cfg["experiment"]["name"],
        ),
    ]

    total_timesteps = args.total_timesteps or train_cfg["total_timesteps"]
    print(f"[train] {args.algo} on {cfg['env']['id']} for {total_timesteps} steps")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    final_path = out_dir / "final_model.zip"
    model.save(final_path)
    print(f"[train] saved final model -> {final_path}")


if __name__ == "__main__":
    main()
