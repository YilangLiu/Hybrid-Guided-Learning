"""Train Genesis Go2 with rl_games PPO.

Usage:
    python scripts/train.py --config cfgs/train/go2_flat_ppo.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make the project root importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_config(train_cfg_path: Path, overrides: argparse.Namespace) -> dict:
    """Compose cfgs/train + cfgs/env + cfgs/agent into a single rl_games config."""
    train_cfg = load_yaml(train_cfg_path)
    cfgs_root = train_cfg_path.resolve().parent.parent
    env_cfg = load_yaml(cfgs_root / "env" / train_cfg["env"])
    agent_cfg = load_yaml(cfgs_root / "agent" / train_cfg["agent"])

    cfg = agent_cfg
    cfg["params"]["config"]["name"] = train_cfg["exp_name"]
    cfg["params"]["config"]["env_name"] = train_cfg["env_name"]
    cfg["params"]["config"]["env_config"] = env_cfg

    if overrides.seed is not None:
        cfg["params"]["seed"] = overrides.seed
    if overrides.num_envs is not None:
        cfg["params"]["config"]["num_actors"] = overrides.num_envs
    if overrides.max_epochs is not None:
        cfg["params"]["config"]["max_epochs"] = overrides.max_epochs
    if overrides.gamma is not None:
        cfg["params"]["config"]["gamma"] = overrides.gamma
    if overrides.exp_name is not None:
        cfg["params"]["config"]["name"] = overrides.exp_name
    if overrides.show_viewer:
        # Forwarded to Go2VecEnv -> Handstand(show_viewer=...).
        cfg["params"]["config"]["env_config"]["show_viewer"] = True
    if overrides.trace:
        cfg["params"]["config"]["env_config"]["trace"] = True
        if overrides.trace_path:
            cfg["params"]["config"]["env_config"]["trace_path"] = overrides.trace_path
    return cfg


def find_latest_checkpoint(exp_name: str) -> Path | None:
    """Find ``runs/<exp_name>_<ts>/nn/<exp_name>.pth`` from the most recent
    run directory matching ``exp_name``. Returns None if nothing matches.

    Used by ``--play`` to default ``--checkpoint`` so eval doesn't need the
    full path. Sort key is the run dir's mtime (matches "the run I just
    finished" intuition even if timestamp strings sort lexically wrong).
    """
    runs_root = REPO_ROOT / "runs"
    if not runs_root.is_dir():
        return None
    candidates = sorted(
        (d for d in runs_root.glob(f"{exp_name}_*") if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for d in candidates:
        ckpt = d / "nn" / f"{exp_name}.pth"
        if ckpt.is_file():
            return ckpt
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "cfgs/train/go2_flat_ppo.yaml")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override the agent yaml's PPO discount factor.")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Override the run/experiment name (run dir prefix).")
    parser.add_argument("--play", action="store_true",
                        help="Run inference instead of training. In --play mode, "
                             "--num_envs defaults to 1, --show_viewer and --trace "
                             "default ON, and --checkpoint auto-resolves to the "
                             "latest runs/<exp_name>_*/nn/<exp_name>.pth.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pth checkpoint (for --play or resume). "
                             "Optional under --play (auto-resolved).")
    parser.add_argument("--show_viewer", action="store_true",
                        help="Open the Genesis viewer (default ON under --play).")
    parser.add_argument("--trace", action="store_true",
                        help="Record per-step rollout state to a .pt file "
                             "(default ON under --play).")
    parser.add_argument("--trace_path", type=str, default=None,
                        help="Where to save the trace (default: "
                             "runs/eval_traces/trace_<ts>.pt).")
    parser.add_argument("--wandb", action="store_true",
                        help="Log training metrics to Weights & Biases. "
                             "Bridges rl_games' tensorboard writer via "
                             "`wandb.init(sync_tensorboard=True)` — no rl_games "
                             "changes needed.")
    parser.add_argument("--wandb_project", type=str, default="hybrid_guided_learning",
                        help="W&B project name (default: hybrid_guided_learning).")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name; defaults to the exp_name from the "
                             "training YAML.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (team/user); defaults to your wandb "
                             "config default if unset.")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="Optional W&B run-group label for sweeps.")
    args = parser.parse_args()

    # --play conveniences: shrink to a single visible env, default tracing
    # and the viewer on, and auto-resolve the checkpoint from the latest run
    # directory if the user didn't pass one. Each default is a no-op if the
    # flag was already set on the CLI.
    if args.play:
        if args.num_envs is None:
            args.num_envs = 1
        args.show_viewer = True
        args.trace = True
        if args.checkpoint is None:
            exp_name = load_yaml(args.config)["exp_name"]
            ckpt = find_latest_checkpoint(exp_name)
            if ckpt is None:
                raise FileNotFoundError(
                    f"--play requested but no checkpoint found under "
                    f"runs/{exp_name}_*/nn/{exp_name}.pth. Pass --checkpoint "
                    f"explicitly."
                )
            args.checkpoint = str(ckpt)
            print(f"[play] auto-resolved checkpoint: {args.checkpoint}")

    cfg = build_config(args.config, args)

    # Genesis MUST be initialised before any env construction.
    import genesis as gs
    import torch
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=cfg["params"]["seed"],
        performance_mode=True,
    )

    torch.set_default_device("cpu")

    # Side-effect import: registers GENESIS_GO2 with rl_games.
    import hybrid_guided_learning.rl_games_integration  # noqa: F401

    # Bridge rl_games' tensorboard writer to W&B. Must happen BEFORE Runner
    # creates its SummaryWriter (which happens inside A2CBase.__init__ during
    # runner.run_train), so wandb's monkey-patch is already in place.
    wandb_run = None
    if args.wandb and not args.play:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "--wandb requested but `wandb` is not installed. "
                "Install with: pip install wandb"
            ) from e
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or cfg["params"]["config"]["name"],
            group=args.wandb_group,
            config=cfg["params"],   # logs all PPO + env hyperparams
            sync_tensorboard=True,  # auto-mirror rl_games' tb scalars
        )

    from rl_games.torch_runner import Runner
    runner = Runner()
    runner.load(cfg)
    runner.reset()

    run_args = {"train": not args.play, "play": args.play}
    if args.checkpoint is not None:
        run_args["checkpoint"] = args.checkpoint
    try:
        runner.run(run_args)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
