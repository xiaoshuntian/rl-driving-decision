"""Smoke tests — lightweight checks that run in CI without GPU or full training.

Goals:
- Catch import errors in src/* and config-loading bugs.
- Verify the algo registry exposes the names we advertise in the README.
- Run *a single env step* with a random policy to confirm gymnasium / highway-env
  registration is intact on a fresh runner.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Avoid OpenMP duplicate-lib crash when torch + numpy MKL collide.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def test_src_modules_import():
    from src import envs, algos, utils  # noqa: F401
    from src.envs import make_highway_env, make_vec_highway_env  # noqa: F401
    from src.algos import build_model, ALGO_REGISTRY  # noqa: F401
    from src.utils import ensure_dir, load_config, set_global_seed  # noqa: F401


def test_algo_registry_has_ppo_and_sac():
    from src.algos import ALGO_REGISTRY

    assert "ppo" in ALGO_REGISTRY
    assert "sac" in ALGO_REGISTRY


def test_configs_load():
    from src.utils import load_config

    for name in ("ppo_highway.yaml", "sac_highway.yaml"):
        cfg = load_config(ROOT / "configs" / name)
        assert "experiment" in cfg
        assert "env" in cfg
        assert "algo" in cfg
        assert "train" in cfg


def test_env_factory_runs_one_step():
    from src.envs import make_highway_env
    from src.utils import load_config

    cfg = load_config(ROOT / "configs" / "ppo_highway.yaml")
    env = make_highway_env(
        env_id=cfg["env"]["id"],
        config=cfg["env"]["config"],
        seed=cfg["experiment"]["seed"],
    )
    obs, info = env.reset(seed=cfg["experiment"]["seed"])
    assert obs is not None
    action = env.action_space.sample()
    step_out = env.step(action)
    assert len(step_out) == 5  # gymnasium API: obs, reward, terminated, truncated, info
    env.close()
