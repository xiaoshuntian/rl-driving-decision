"""highway-env factory functions.

Wraps `gymnasium.make` to apply a config dict and returns either a single env
(for evaluation / inspection) or a vectorized env (for training).
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import highway_env  # noqa: F401  (registers envs with gymnasium)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


def make_highway_env(
    env_id: str,
    config: Optional[dict[str, Any]] = None,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> gym.Env:
    """Create a single highway-env environment with the given config dict."""
    env = gym.make(env_id, render_mode=render_mode)
    if config:
        env.unwrapped.configure(config)
        env.reset(seed=seed)
    return env


def make_vec_highway_env(
    env_id: str,
    n_envs: int = 1,
    config: Optional[dict[str, Any]] = None,
    seed: Optional[int] = None,
    use_subproc: bool = False,
) -> VecEnv:
    """Create a vectorized highway-env environment for SB3 training.

    Subproc only matters when n_envs > 1 and the env is heavy enough
    that GIL contention dominates. For highway-env it's usually fine
    to leave use_subproc=False.
    """

    def _factory(rank: int):
        def _thunk():
            env = make_highway_env(
                env_id, config=config, seed=None if seed is None else seed + rank
            )
            return env

        return _thunk

    factories = [_factory(i) for i in range(n_envs)]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(factories)
    return DummyVecEnv(factories)
