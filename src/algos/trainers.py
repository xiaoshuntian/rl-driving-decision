"""Algorithm builders.

Wraps Stable-Baselines3 PPO/SAC for our two configs. Behavior Cloning and
Diffusion Policy will be added later as separate modules; the registry
pattern lets them slot in without changing the train script.
"""
from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

ALGO_REGISTRY: dict[str, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "sac": SAC,
    # "bc": BehaviorCloning,            # TODO Phase 2
    # "diffusion": DiffusionPolicy,     # TODO Phase 3
}


def build_model(
    algo_name: str,
    env: VecEnv,
    algo_kwargs: dict[str, Any],
    tensorboard_log: str | None = None,
    seed: int | None = None,
) -> BaseAlgorithm:
    """Construct an SB3 model from a registry name + kwargs."""
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(
            f"Unknown algo '{algo_name}'. Registered: {list(ALGO_REGISTRY)}"
        )
    cls = ALGO_REGISTRY[algo_name]
    return cls(env=env, tensorboard_log=tensorboard_log, seed=seed, **algo_kwargs)
