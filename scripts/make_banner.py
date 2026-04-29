"""Generate the GitHub social-preview banner (1280x640).

Pure matplotlib — no PIL/cairo dependency. Saves to assets/social_preview.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "social_preview.png"

plt.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    # Background: deep navy gradient
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    grad = np.repeat(grad, 256, axis=1)
    ax.imshow(
        grad,
        extent=(0, 12.8, 0, 6.4),
        aspect="auto",
        cmap="Blues_r",
        origin="lower",
        alpha=0.92,
        zorder=0,
    )

    # Highway lanes (right side accent)
    for i in range(4):
        y = 1.0 + i * 0.55
        ax.add_patch(
            patches.Rectangle(
                (7.0, y), 5.6, 0.45,
                facecolor="#1a3a52", edgecolor="none", alpha=0.55, zorder=1,
            )
        )
        # dashed lane markers
        for x in np.arange(7.1, 12.6, 0.45):
            ax.plot(
                [x, x + 0.22],
                [y + 0.45, y + 0.45],
                color="white",
                lw=1.2,
                alpha=0.55,
                zorder=2,
            )

    # Ego vehicle (red) and surrounding (blue)
    cars = [
        (8.6, 2.10, "#ff4d4f"),  # ego
        (10.2, 1.55, "#4ea0ff"),
        (9.4, 2.65, "#4ea0ff"),
        (11.4, 3.20, "#4ea0ff"),
        (7.7, 1.55, "#4ea0ff"),
    ]
    for x, y, c in cars:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                0.55, 0.28,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=c, edgecolor="white", linewidth=1.2,
                zorder=3,
            )
        )

    # Title block (left)
    ax.text(
        0.5, 5.55,
        "rl-driving-decision",
        fontsize=42, fontweight="bold", color="white",
        family="DejaVu Sans",
    )
    ax.text(
        0.5, 4.85,
        "强化学习 · 模仿学习 · 扩散策略",
        fontsize=22, color="#ffffff", alpha=0.95,
    )
    ax.text(
        0.5, 4.30,
        "在 highway-env 中的自动驾驶决策基线",
        fontsize=18, color="#cfe6ff",
    )

    # Tag pills
    tags = ["PPO", "SAC", "BC", "Diffusion Policy", "PyTorch", "SB3"]
    x_pos = 0.5
    for tag in tags:
        w = 0.18 * len(tag) + 0.45
        ax.add_patch(
            patches.FancyBboxPatch(
                (x_pos, 3.30),
                w, 0.55,
                boxstyle="round,pad=0.02,rounding_size=0.18",
                facecolor="#0f4d80", edgecolor="#4ea0ff", linewidth=1.2,
                zorder=2,
            )
        )
        ax.text(
            x_pos + w / 2, 3.575, tag,
            fontsize=14, color="white",
            ha="center", va="center", family="DejaVu Sans",
        )
        x_pos += w + 0.18

    # Footer
    ax.text(
        0.5, 0.55,
        "github.com/xiaoshuntian/rl-driving-decision",
        fontsize=14, color="#7fb6ff", family="DejaVu Sans",
    )
    ax.text(
        0.5, 0.20,
        "Tongji University · Transportation Engineering · 2026",
        fontsize=11, color="#a0c4ed", family="DejaVu Sans",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT, dpi=100, facecolor="#0a2240")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
