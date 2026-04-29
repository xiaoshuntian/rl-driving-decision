"""Plot training reward curve from SB3 evaluations.npz or VecMonitor csv.

Saves PNG to assets/<run_name>_reward.png.

Usage:
    python scripts/plot_reward.py --run-dir runs/ppo_highway_fast --title "PPO on highway-fast-v0"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")


def load_eval_npz(eval_dir: Path):
    npz_path = eval_dir / "evaluations.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]  # shape: (n_evals, n_episodes)
    mean = results.mean(axis=1)
    std = results.std(axis=1)
    return timesteps, mean, std


def load_monitor_csv(run_dir: Path):
    rows = []
    for csv in run_dir.glob("**/*.monitor.csv"):
        df = pd.read_csv(csv, skiprows=1)
        rows.append(df)
    if not rows:
        return None
    df = pd.concat(rows).sort_values("t").reset_index(drop=True)
    df["timesteps"] = df["l"].cumsum()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--title", default="Training reward")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else ROOT / "assets" / f"{run_dir.name}_reward.png"

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)

    eval_data = load_eval_npz(run_dir / "eval")
    if eval_data is not None:
        ts, mean, std = eval_data
        ax.plot(ts, mean, color="#1f77b4", lw=2, label="eval mean (10 eps)")
        ax.fill_between(ts, mean - std, mean + std, color="#1f77b4", alpha=0.2,
                        label="±1 std")

    mon = load_monitor_csv(run_dir)
    if mon is not None and len(mon) > 0:
        ax.scatter(mon["timesteps"], mon["r"], s=8, color="#888", alpha=0.45,
                   label="train ep return", zorder=1)
        if len(mon) >= 5:
            sm = _smooth(mon["r"].to_numpy(), k=min(20, len(mon) // 4 or 1))
            ts_sm = mon["timesteps"].to_numpy()[len(mon) - len(sm):]
            ax.plot(ts_sm, sm, color="#ff7f0e", lw=1.6, alpha=0.8,
                    label="train smoothed")

    ax.set_xlabel("timesteps")
    ax.set_ylabel("episode return")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
