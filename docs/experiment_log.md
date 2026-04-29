# Experiment Log

Keep one entry per training run. Helps when interviewers ask "what did you try and what didn't work".

Template:

## YYYY-MM-DD — short title

- **Hypothesis**: what I expected
- **Setup**: env, algo, hyperparams (or pointer to config file + git SHA)
- **Result**: numbers + 1-line takeaway
- **Next step**: what to change next

---

## 2026-05-?? — PPO baseline on highway-fast-v0

- **Hypothesis**: PPO with default highway-env reward should reach ~25 mean return after 200k steps.
- **Setup**: `configs/ppo_highway.yaml`, seed 42, 4 parallel envs.
- **Result**: TBD
- **Next step**: TBD
