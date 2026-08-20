# Hybrid-Guided-Learning — Go2 Single-Handstand Task

Genesis implementation of an agile Unitree Go2 balance task, trained with
[`rl_games`](https://github.com/Denys88/rl_games) PPO:

| Task | Env class | Description |
|---|---|---|
| **Single handstand** | `SingleHandstand` | Balance on *one* front leg while the other front leg lifts toward the target height; rear legs in the air. |

The task runs at 250 Hz control and physics and is trainable in two action
spaces: joint-position targets behind a PD loop, or direct joint torques.
(`SingleHandstand` derives from a `Handstand` base class that is not exposed
as a task.) The env implementation lives in
[hybrid_guided_learning/envs/go2/handstand.py](hybrid_guided_learning/envs/go2/handstand.py).

---

## 1. Prerequisites

- Linux (developed on Ubuntu 24.04)
- NVIDIA GPU with CUDA 12+ (the Genesis rigid solver's GPU kernels)
- Conda (Miniconda or Anaconda)

---

## 2. Installation

### 2.1 Clone

```bash
git clone git@github.com:YilangLiu/Hybrid-Guided-Learning.git
cd Hybrid-Guided-Learning
```

> **Note on `externals/`**
> Genesis and rl_games are heavyweight checkouts kept out of this repo (see
> `.gitignore`). The steps below clone them at the exact commits this project
> was developed against.

### 2.2 Create the conda environment

```bash
conda create -n hybrid_guided_learning python=3.11 -y
conda activate hybrid_guided_learning
```

### 2.3 Install PyTorch (CUDA 12.x / 13.x)

```bash
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
```

For other CUDA versions see the [PyTorch install matrix](https://pytorch.org/get-started/locally/).

### 2.4 Clone & install Genesis

```bash
mkdir -p externals && cd externals
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
(cd Genesis && git checkout 723a20eca31433db37a440bbd7df2d181cb4758d)
pip install -e Genesis
cd ..
```

The first install builds the Genesis runtime and can take several minutes.

### 2.5 Clone & install rl_games

```bash
cd externals
git clone https://github.com/Denys88/rl_games.git
(cd rl_games && git checkout 46f286ac456555e82a4dc3488b3641860db06111)
pip install -e rl_games
cd ..
```

### 2.6 Remaining Python deps

```bash
pip install pyyaml tensordict tensorboard
```

W&B is optional but recommended:

```bash
pip install wandb && wandb login
```

### 2.7 Verify

```bash
python -c "import torch, genesis, rl_games; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
python -c "from hybrid_guided_learning.envs.go2 import SingleHandstand; print('env class OK')"
```

---

## 3. Config layout

Each run composes three YAMLs:

```
cfgs/train/<task>.yaml            ← entrypoint: names the env + agent config
  ├─ env:   cfgs/env/<task>.yaml     ← reward shaping, joint init, kp/kd, control mode
  └─ agent: cfgs/agent/ppo_handstand.yaml   ← PPO hyperparameters
```

`ppo_handstand.yaml` is shared by both runs below — they differ only
in their env config and in the `env_name:` that selects the env class. Env-class
registration with rl_games happens in
[hybrid_guided_learning/rl_games_integration/register.py](hybrid_guided_learning/rl_games_integration/register.py).

Available entrypoints:

| `cfgs/train/…` | Env class | Actions |
|---|---|---|
| `go2_single_handstand_ppo.yaml` | `SingleHandstand` | position (PD, incremental targets) |
| `go2_single_handstand_ppo_torque.yaml` | `SingleHandstand` | direct joint torques |

The torque variant sets `env_cfg.control_mode: torque`, bypassing the PD loop so
actions in [-1, 1] map to ±`torque_limit_scale` × the MJCF actuator limits. It
also re-tunes two reward weights (`action_rate: -3.0`, `orientation: 1.5`) —
without PD filtering the policy otherwise amplifies observation noise into
torque chatter and settles into a shallow-pose attractor. Rationale is in the
config's comments.

---

## 4. Train

```bash
# Single-leg handstand (position control)
python scripts/train.py --config cfgs/train/go2_single_handstand_ppo.yaml

# Single-leg handstand, torque control
python scripts/train.py --config cfgs/train/go2_single_handstand_ppo_torque.yaml
```

With W&B logging (bridged through rl_games' TensorBoard writer):

```bash
python scripts/train.py \
  --config cfgs/train/go2_single_handstand_ppo.yaml \
  --wandb --wandb_project hybrid_guided_learning \
  --wandb_run_name singlehandstand_ppo_$(date +%m%d_%H%M%S)
```

Checkpoints land under `runs/<exp_name>_<dd-HH-MM-SS>/nn/`:

- `<exp_name>.pth` — best-so-far (overwritten as reward improves)
- `last_<exp_name>_ep_<N>_rew_<R>.pth` — periodic snapshots

TensorBoard scalars go to `runs/<exp>_<ts>/summaries/`:

```bash
tensorboard --logdir runs/
```

CLI overrides (all optional, each wins over the YAML):

| Flag | Effect |
|---|---|
| `--num_envs N` | Parallel envs (config default: 8192) |
| `--max_epochs N` | Training length (config default: 2500) |
| `--seed N` | RNG seed (config default: 42) |
| `--gamma G` | PPO discount factor |
| `--exp_name NAME` | Run/experiment name, i.e. the `runs/` dir prefix |
