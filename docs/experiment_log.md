# Experiment Log

Keep one entry per training run. Helps when interviewers ask "what did you try and what didn't work".

Template:

```
## YYYY-MM-DD — short title
- **Hypothesis**: what I expected
- **Setup**: env, algo, hyperparams (or pointer to config file + git SHA)
- **Result**: numbers + 1-line takeaway
- **Next step**: what to change next
```

---

## 2026-04-29 — PPO baseline on highway-fast-v0 (20k steps)

- **Hypothesis**: PPO with default highway-env reward should drop the crash rate well below the random-policy baseline (~50%) within ~10 iterations and learn to ride out the full 40-step episode cap.
- **Setup**: `configs/ppo_highway.yaml`, seed 42, `n_envs=4`, `n_steps=512` (rollout = 2048), 10 PPO epochs, lr=5e-4, γ=0.95, ent_coef=0.01. CPU only, Windows 11. Total 20 480 env steps = 10 PPO iterations. Wall-clock ~111 minutes.
- **Result**:

  | iter | total_timesteps | ep_rew_mean | ep_len_mean |
  |------|----------------:|------------:|------------:|
  | 1 |  2 048 |  9.7 | 12.8 |
  | 2 |  4 096 | 12.8 | 17.0 |
  | 3 |  6 144 | 15.0 | 20.1 |
  | 4 |  8 192 | 18.5 | 24.9 |
  | 5 | 10 240 | 21.3 | 29.3 |
  | 6 | 12 288 | 21.5 | 29.7 |
  | 7 | 14 336 | 24.0 | 33.2 |
  | 8 | 16 384 | 25.0 | 34.7 |
  | 9 | 18 432 | 26.9 | 37.5 |
  | 10 | 20 480 | **27.4** | **38.4 / 40** |

  Deterministic eval, 20 episodes: return = **27.15 ± 4.66**, mean length 38.5 / 40, **crash rate 5.0%** (1/20).

  Take-away: clean monotonic learning curve, no instability. Episode length saturates near the 40-step cap, indicating the 40-step horizon may now be the primary ceiling — for a stronger Phase 1.6 we should bump `duration` in the env config to let the agent demonstrate longer horizons.

- **Next step**:
  1. Run SAC on the continuous-action variant (`configs/sac_highway.yaml`) for a side-by-side comparison.
  2. Repeat PPO at 200k steps with `duration: 80` to study true convergence (Phase 1.6).
  3. Phase 2: collect a small BC dataset from the trained PPO and train a BC policy as warm-start; measure how many steps PPO needs starting from BC.
