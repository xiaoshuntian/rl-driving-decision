# rl-driving-decision

> 自动驾驶决策的强化学习基线 — 在 highway-env 中分层对比 PPO、SAC、行为克隆（BC）与扩散策略（Diffusion Policy），后续扩展到 CARLA / nuPlan。

![banner](assets/social_preview.png)

[English README](README.md) · 中文说明

**作者**：肖顺天（同济大学交通工程，2027 届）　|　**当前阶段**：Phase 1（PPO/SAC baselines）

---

## 项目动机

2026 年自动驾驶决策规划类岗位的招聘要求里反复出现同一个三件套：

- 熟练 **PyTorch** 与主流强化学习框架（Stable-Baselines3 / CleanRL / TianShou）
- 跑得通 **闭环仿真**（highway-env / CARLA / nuPlan）
- 能 **复现论文**（Diffusion Policy、DriveDPO、Hydra-MDP …）

这个仓库就是把上述三点最小可演示地串起来，分三个阶段逐层加码：

| 阶段 | 时间 | 目标 | 方法 | 仿真环境 |
|------|------|------|------|----------|
| Phase 1 | 2026.05 | RL 基线跑通、评估、对比 | PPO、SAC | highway-env |
| Phase 2 | 2026.06–07 | 加入模仿学习 + 升级仿真器 | + 行为克隆、IL warm-start | highway-env / CARLA 简化场景 |
| Phase 3 | 2026.08+ | 复现 Diffusion Policy 并对比 | + Diffusion Policy | highway-env / nuPlan 子集 |

---

## 仓库结构

```
rl-driving-decision/
├── configs/                 # YAML 配置（每个 算法 × 环境 一份）
│   ├── ppo_highway.yaml
│   └── sac_highway.yaml
├── src/
│   ├── envs/                # highway-env 工厂 + VecEnv 封装
│   ├── algos/               # 算法注册表（PPO/SAC，BC/Diffusion 待加）
│   └── utils/               # 配置加载、随机种子
├── scripts/
│   ├── random_rollout.py    # 冒烟测试 — 随机策略，不训练
│   ├── train.py             # 按配置训练
│   ├── eval.py              # 加载已保存模型评估
│   ├── plot_reward.py       # 从训练日志画奖励曲线
│   └── make_banner.py       # 生成社交预览图
├── docs/experiment_log.md   # 每次训练一条记录
├── assets/                  # 图表 / 预览图
├── requirements.txt
└── README.md / README.zh-CN.md
```

---

## 安装

测试环境：Python 3.11 / 3.13，Windows 11 + Ubuntu 22.04，CPU 与 CUDA 均可。

```bash
# 1. clone
git clone https://github.com/xiaoshuntian/rl-driving-decision.git
cd rl-driving-decision

# 2. 创建虚拟环境（推荐）
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

> **Windows + OpenMP 报错处理**：如果你看到 `OMP: Error #15: Initializing libiomp5md.dll`，在环境变量里设置 `KMP_DUPLICATE_LIB_OK=TRUE` 即可。这是 torch / numpy 在 Windows 下已知的兼容问题，对当前规模的训练完全无害。

---

## 快速上手

### 1. 冒烟测试 — 确认环境正常加载

```bash
python scripts/random_rollout.py --config configs/ppo_highway.yaml --episodes 5
```

预期输出：随机策略下 5 个 episode 的累计奖励（因频繁碰撞通常为负或较低）。

### 2. 极短训练（约 30 秒，纯流程检测）

```bash
python scripts/train.py --algo ppo --config configs/ppo_highway.yaml --total-timesteps 5000
```

`--total-timesteps` 覆盖配置文件里默认的 200k 步，方便先确认管线无误。Tensorboard 日志写入 `runs/ppo_highway_fast/tb`。

### 3. 完整 PPO 训练（CPU 大约 30–90 分钟看步数）

```bash
python scripts/train.py --algo ppo --config configs/ppo_highway.yaml
```

```bash
# 另开一个终端启动 Tensorboard
tensorboard --logdir runs/ppo_highway_fast/tb
```

### 4. 评估

```bash
python scripts/eval.py --algo ppo --config configs/ppo_highway.yaml \
    --model runs/ppo_highway_fast/final_model.zip --episodes 20
```

输出包括平均回报、平均 episode 长度、碰撞率三项指标。

### 5. SAC（连续动作版）

```bash
python scripts/train.py --algo sac --config configs/sac_highway.yaml
```

### 6. 画训练曲线

训练结束后执行：

```bash
python scripts/plot_reward.py --run-dir runs/ppo_highway_fast --title "PPO on highway-fast-v0"
```

PNG 文件输出到 `assets/<run_name>_reward.png`。

---

## 实验结果

> 每完成一次训练就在 `docs/experiment_log.md` 加一条记录，并把曲线图嵌到这里。

### Phase 1 · PPO baseline on highway-fast-v0

![PPO 训练曲线](assets/ppo_highway_fast_reward.png)

| 指标 | 数值 |
|------|------|
| 训练步数 | 20 480 |
| 起始 ep_rew_mean | 9.7 |
| **最终 ep_rew_mean** | **27.4** |
| 起始 ep_len_mean | 12.8 |
| **最终 ep_len_mean** | **38.4 / 40** |
| 训练耗时（CPU） | ~111 分钟 |

确定性评估（20 集，加载最终模型）：

| 指标 | 数值 |
|------|------|
| 平均 return | **27.15 ± 4.66** |
| 平均 episode 长度 | 38.5 / 40 |
| **碰撞率** | **5.0 %**（20 集中 1 次） |

> 训练在不到 11 个 PPO 迭代内（~20k 步）就把碰撞率从随机策略的 ~50% 压到 5%，episode 长度跑满 40 步上限说明智能体已经学会在车流里稳定保持车道与跟车。后续 Phase 2 会用 Behavior Cloning 做 warm-start，看能否把所需训练步数再压一半。

完整实验记录见 [`docs/experiment_log.md`](docs/experiment_log.md)。

---

## 设计思路与代码组织

整个项目用 **配置驱动 + 注册表模式** 解耦：

- **环境**：`src/envs/highway_wrappers.py` 只暴露两个工厂函数（单环境 / 向量环境），所有 highway-env 配置走 YAML 字典传入，方便 sweep。
- **算法**：`src/algos/trainers.py` 用一个 `ALGO_REGISTRY` 字典管理；新加 BC、Diffusion Policy 不需要改训练脚本，注册即用。
- **训练入口**：`scripts/train.py` 只做三件事 — 加载 YAML、构造模型、调 `model.learn()` + 回调（`EvalCallback` + `CheckpointCallback`）。

这种结构在面试时方便回答"为什么是这样写"："YAML 描述实验意图，代码描述算法机制，两者解耦后 reproducibility 和 sweep 都简单。"

---

## 路线图

- [x] **Phase 0** — 仓库骨架，配置驱动训练管线
- [x] **Phase 1** — PPO 基线训练完成、评估、画图（20k 步，5% 碰撞率）
- [ ] **Phase 1.5** — SAC 基线 + 与 PPO 并列对比
- [ ] **Phase 1.6** — 200k 步完整 PPO 训练，研究收敛行为
- [ ] **Phase 2** — Behavior Cloning 模块 + IL warm-start；CARLA leaderboard-2.0 简化场景
- [ ] **Phase 3** — 复现 Diffusion Policy（Chi et al., 2023）并在同一基准上正面对比
- [ ] **Phase 4** — 把全部实验整理成 6–8 页技术报告放在 `docs/`

---

## 复现 README 中的曲线

```bash
python scripts/train.py --algo ppo --config configs/ppo_highway.yaml \
    --total-timesteps 20000 > /tmp/train.log 2>&1
python scripts/plot_from_log.py --log /tmp/train.log \
    --title "PPO on highway-fast-v0" --out assets/ppo_highway_fast_reward.png
python scripts/eval.py --algo ppo --config configs/ppo_highway.yaml \
    --model runs/ppo_highway_fast/final_model.zip --episodes 20
```

---

## 阅读清单

正在配合代码精读的论文：

- Schulman et al., *Proximal Policy Optimization Algorithms* (2017)
- Haarnoja et al., *Soft Actor-Critic* (2018)
- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (2023)
- Liu et al., *DriveDPO* (2024)
- Renz et al., *Hydra-MDP*（NVIDIA, 2024）

## 许可证

MIT — 详见 `LICENSE`。
