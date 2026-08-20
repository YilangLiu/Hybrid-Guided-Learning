"""Render a trained handstand policy to an mp4 -- headless, no Genesis viewer.

`train.py --play` opens the interactive viewer, which needs a display. This
script instead attaches an offscreen Genesis camera to the env's scene and
writes the frames to a video file, so it works over SSH / on a cluster.

Usage:
    python scripts/eval_video.py --config cfgs/train/go2_single_handstand_ppo.yaml
    python scripts/eval_video.py --config cfgs/train/go2_single_handstand_ppo_torque.yaml \
        --checkpoint runs/single_handstand_.../nn/single_handstand.pth --out clip.mp4

Defaults render one frame every 5 control steps at 50 fps, which is real-time
playback for the tasks' 250 Hz control rate.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as train_mod  # config composition + checkpoint resolution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Defaults to the latest runs/<exp>_*/nn/<exp>.pth.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output mp4 (default: runs/videos/<exp>_<ts>.mp4).")
    ap.add_argument("--steps", type=int, default=1500,
                    help="Control steps to roll out (250 per simulated second).")
    ap.add_argument("--render_every", type=int, default=5,
                    help="Render one frame every N control steps.")
    ap.add_argument("--fps", type=int, default=50,
                    help="Video frame rate. 250/render_every == real time.")
    ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--cam_pos", type=float, nargs=3, default=[1.3, -1.3, 0.8])
    ap.add_argument("--cam_lookat", type=float, nargs=3, default=[0.0, 0.0, 0.38])
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--no_follow", action="store_true",
                    help="Keep the camera static instead of tracking the robot "
                         "(episodes start at xy ~ U(-0.5, 0.5), so tracking "
                         "usually frames the robot better).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample actions instead of using the policy mean.")
    args = ap.parse_args()

    exp_name = train_mod.load_yaml(args.config)["exp_name"]
    if args.checkpoint is None:
        ckpt = train_mod.find_latest_checkpoint(exp_name)
        if ckpt is None:
            raise FileNotFoundError(
                f"No checkpoint under runs/{exp_name}_*/nn/{exp_name}.pth. "
                f"Note the PPO config only writes that file after "
                f"`save_best_after` epochs -- pass --checkpoint explicitly to "
                f"use a `last_*` snapshot instead."
            )
        args.checkpoint = str(ckpt)
        print(f"[eval_video] auto-resolved checkpoint: {args.checkpoint}")

    out = args.out or str(
        REPO_ROOT / "runs" / "videos"
        / f"{exp_name}_{time.strftime('%m%d_%H%M%S')}.mp4"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # One env: the scene renders only env 0 (VisOptions.rendered_envs_idx=[0]).
    overrides = argparse.Namespace(
        seed=args.seed, num_envs=1, max_epochs=None, gamma=None,
        exp_name=None, show_viewer=False, trace=False, trace_path=None,
    )
    cfg = train_mod.build_config(args.config, overrides)
    cfg["params"]["config"]["player"] = {
        **cfg["params"]["config"].get("player", {}),
        "use_vecenv": True,
        "render": False,
    }

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

    # `Scene.add_camera` is @assert_unbuilt and the env builds its scene inside
    # __init__, so there is no hook to add a camera through. Wrap Scene.build to
    # slot the camera in just before the build, and restore it right after so
    # nothing else in the process is affected.
    holder: dict = {}
    orig_build = gs.Scene.build

    def build_with_camera(scene, *a, **kw):
        holder["cam"] = scene.add_camera(
            res=tuple(args.res),
            pos=tuple(args.cam_pos),
            lookat=tuple(args.cam_lookat),
            fov=args.fov,
            GUI=False,
        )
        return orig_build(scene, *a, **kw)

    gs.Scene.build = build_with_camera
    try:
        import hybrid_guided_learning.rl_games_integration  # noqa: F401  (registers envs)
        from rl_games.torch_runner import Runner
        runner = Runner()
        runner.load(cfg)
        player = runner.create_player()
    finally:
        gs.Scene.build = orig_build

    player.restore(args.checkpoint)
    # Only set inside player.run(); we drive the rollout ourselves, and leaving
    # it False would unsqueeze the already-batched obs.
    player.has_batch_dimension = True

    cam = holder["cam"]
    env = player.env
    if not args.no_follow:
        cam.follow_entity(
            env.env.robot,
            fixed_axis=(None, None, args.cam_pos[2]),
            smoothing=0.5,
        )

    obs = player.env_reset(env)
    cam.start_recording()

    frames = 0
    ep_return = 0.0
    ep_len = 0
    returns: list[float] = []
    lengths: list[int] = []
    t0 = time.time()
    for t in range(args.steps):
        action = player.get_action(obs, is_deterministic=not args.stochastic)
        obs, rew, done, _ = player.env_step(env, action)
        ep_return += float(rew.reshape(-1)[0])
        ep_len += 1
        if t % args.render_every == 0:
            cam.render()
            frames += 1
        if bool(done.reshape(-1)[0]):
            returns.append(ep_return)
            lengths.append(ep_len)
            ep_return, ep_len = 0.0, 0

    cam.stop_recording(save_to_filename=out, fps=args.fps)

    print(f"\n[eval_video] {exp_name}")
    print(f"  checkpoint      : {args.checkpoint}")
    print(f"  video           : {out}")
    print(f"  frames          : {frames} @ {args.fps} fps "
          f"({frames / args.fps:.1f}s of video, "
          f"{args.steps * 0.004:.1f}s simulated)")
    print(f"  wall clock      : {time.time() - t0:.1f}s")
    if returns:
        mean_r = sum(returns) / len(returns)
        mean_l = sum(lengths) / len(lengths)
        print(f"  episodes ended  : {len(returns)}")
        print(f"  return  mean    : {mean_r:.2f}  (per-episode: "
              f"{', '.join(f'{r:.1f}' for r in returns[:8])}"
              f"{' ...' if len(returns) > 8 else ''})")
        print(f"  length  mean    : {mean_l:.0f} / "
              f"{env.env.max_episode_length} steps")
    else:
        print(f"  episodes ended  : 0 -- survived all {args.steps} steps "
              f"(max episode length {env.env.max_episode_length})")


if __name__ == "__main__":
    main()
