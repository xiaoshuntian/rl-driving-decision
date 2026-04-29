"""Parse SB3's stdout training log and plot ep_rew_mean / ep_len_mean.

Useful when EvalCallback didn't fire or VecMonitor wasn't configured to write a CSV.
The plot script defaults to reading rollout/ rows from a text log.

Usage:
    python scripts/plot_from_log.py --log /tmp/train_full.log \
        --title "PPO on highway-fast-v0" --out assets/ppo_highway_fast_reward.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


KEYS = ["ep_rew_mean", "ep_len_mean", "total_timesteps", "iterations", "fps"]
KEY_RE = {k: re.compile(rf"\|\s+{re.escape(k)}\s*\|\s+([\d.\-]+)\s*\|") for k in KEYS}


def parse_log(path: Path):
    """Return list of dicts, one per training iteration block."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Split on lines of dashes that separate SB3's pretty tables
    blocks = re.split(r"-{3,}", text)
    rows = []
    for blk in blocks:
        row = {}
        for k, rgx in KEY_RE.items():
            m = rgx.search(blk)
            if m:
                row[k] = float(m.group(1))
        if "ep_rew_mean" in row and "total_timesteps" in row:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--title", default="Training reward")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = parse_log(Path(args.log))
    if not rows:
        raise SystemExit(f"No iteration rows parsed from {args.log}")

    ts = [r["total_timesteps"] for r in rows]
    rew = [r["ep_rew_mean"] for r in rows]
    elen = [r.get("ep_len_mean", float("nan")) for r in rows]

    out_path = Path(args.out) if args.out else ROOT / "assets" / "ppo_reward.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)

    ax1.plot(ts, rew, marker="o", color="#1f77b4", lw=2)
    ax1.set_xlabel("total timesteps")
    ax1.set_ylabel("ep_rew_mean")
    ax1.set_title(f"{args.title} — episode return")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=rew[0], color="gray", linestyle=":", lw=1, alpha=0.6,
                label=f"start = {rew[0]:.1f}")
    ax1.axhline(y=rew[-1], color="green", linestyle=":", lw=1, alpha=0.6,
                label=f"end = {rew[-1]:.1f}")
    ax1.legend(loc="lower right", fontsize=9)

    ax2.plot(ts, elen, marker="o", color="#ff7f0e", lw=2)
    ax2.set_xlabel("total timesteps")
    ax2.set_ylabel("ep_len_mean")
    ax2.set_title(f"{args.title} — episode length (max=40)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 42)
    ax2.axhline(y=40, color="red", linestyle="--", lw=1, alpha=0.4,
                label="episode cap (40 steps)")
    ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}  ({len(rows)} iterations)")
    for r in rows:
        print(
            f"  ts={int(r['total_timesteps']):>6}  "
            f"rew={r['ep_rew_mean']:.2f}  "
            f"len={r.get('ep_len_mean', float('nan')):.1f}"
        )


if __name__ == "__main__":
    main()
