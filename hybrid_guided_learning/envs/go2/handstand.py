"""Go2 handstand tasks in Genesis (standalone implementation).

This file is a 1:1 port of mujoco_playground's Go2 handstand task
(``externals/mujoco_playground/mujoco_playground/_src/locomotion/go2/handstand.py``)
on top of Genesis. The env builds its own ``gs.Scene`` and manages all
buffers internally, so it has no dependency on a shared locomotion base
class.

Alignment with the reference (the relevant lines in the JAX/MJX version):
  * Action:    ``motor_targets = state.data.ctrl + action * action_scale``
               (incremental encoding via a persistent ``_motor_targets``).
               Alternatively ``env_cfg.control_mode: torque`` bypasses the
               PD loop: actions in [-1, 1] map to direct joint torques
               (± ``torque_limit_scale`` × the MJCF actuator limit) via
               ``control_dofs_force``; the MJCF's passive joint damping
               and friction loss still apply.
  * Obs:       dict with ``policy`` (45D, noisy) and ``critic`` (94D,
               privileged: clean signals + joint torques + IMU world z).
  * Termination: fall (body-z's world-z < -0.25) | unwanted-contact on
               front-leg geoms | energy threshold | sim error | time-out.
               The termination *reward* uses only the "real" terminations
               (no time-out) — matches the reference's ``done`` arg into
               ``_get_reward``.
  * Reward:    ``clip(sum(rew_k * scale_k) * dt, 0, 10000)`` (the clip
               floors any net-negative step to 0, which is core to the
               reference's stability — keep it).
  * Reset:     xy ~ U(-0.5, 0.5), yaw ~ U(-π, π), qvel[:6] ~ U(-0.5, 0.5);
               qpos[7:] left at the default home pose.

Documented Genesis-vs-MuJoCo approximations:
  * Accelerometer: the reference reads ``data.sensordata[accel_sensor]``
    (body-frame linear accel); Genesis has no equivalent sensor. We pad
    the privileged obs with zeros at that slot. This is not load-bearing
    for handstand — the critic still has clean linvel/angvel/torques/height.
  * Per-geom contact: the reference uses MuJoCo's per-collision-geom
    ``*_floor_found`` sensors. Genesis exposes per-LINK net contact force,
    so we threshold its norm against ``contact_force_threshold`` (1 N).
    With Genesis MJCF loading, foot-tip contact lives on ``*_foot`` links
    (distinct from ``*_calf``), so the distinction the reference's
    ``fl_calf_*`` geoms make is still preserved.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import (
    inv_quat,
    transform_by_quat,
    transform_quat_by_quat,
)


def _resolve_xml(path: str) -> str:
    if os.path.isabs(path):
        return path
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return os.path.join(repo_root, path)


class Handstand:
    """Front-leg handstand task for Go2 in Genesis.

    Matches the mujoco_playground reference's training contract exactly
    (obs / reward / termination / action encoding). Loads the MJCF at
    ``env_cfg["xml_path"]`` — required, because the URDF collapses
    ``*_foot`` bodies into their parent calves and we need them distinct
    to mirror the reference's per-geom contact semantics.
    """

    # --- Defaults (override via reward_cfg in the YAML) ---
    DESIRED_FORWARD: List[float] = [0.0, 0.0, -1.0]
    POSE_JOINT_IDS: List[int] = [0, 3, 6, 7, 8, 9, 10, 11]
    # Front-leg shaft links (foot-tip is on FL_foot/FR_foot which IS allowed).
    UNWANTED_CONTACT_LINKS: List[str] = [
        "FL_calf", "FR_calf",
        "FL_thigh", "FR_thigh",
        "FL_hip", "FR_hip",
    ]
    OFF_GROUND_FEET: List[str] = ["RL_foot", "RR_foot"]
    Z_DES: float = 0.55

    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg=None,
        show_viewer: bool = False,
    ):
        self.num_envs = int(num_envs)
        self.num_actions = int(env_cfg["num_actions"])
        self.device = gs.device

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        # command_cfg kept for vecenv compatibility; handstand has no commands.

        # ----- Control / physics timing -----
        # The reference uses 250 Hz control & sim (ctrl_dt = sim_dt = 0.004).
        self.dt = float(env_cfg.get("ctrl_dt", 0.004))
        self.substeps = int(env_cfg.get("substeps", 1))
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.control_mode = str(env_cfg.get("control_mode", "position"))
        if self.control_mode not in ("position", "torque"):
            raise ValueError(
                f"env_cfg.control_mode must be 'position' or 'torque', "
                f"got {self.control_mode!r}"
            )
        # Position-increment gain; unused in torque mode.
        self.action_scale = (
            float(env_cfg["action_scale"]) if self.control_mode == "position" else 0.0
        )
        self.clip_actions = float(env_cfg.get("clip_actions", 100.0))

        # ----- Build the Genesis scene -----
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.substeps),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=20,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        )

        xml_path = env_cfg.get("xml_path")
        if not xml_path:
            raise ValueError(
                "Handstand requires env_cfg.xml_path -> go2_genesis.xml. "
                "The MJCF preserves *_foot links the URDF collapses."
            )
        xml_path = _resolve_xml(xml_path)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"env_cfg.xml_path -> {xml_path}")
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=xml_path,
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            ),
        )

        self.scene.build(n_envs=self.num_envs)

        # ----- Joint indexing -----
        # ``motors_dof_idx``: DOF indices ordered by env_cfg.joint_names.
        # ``actions_dof_idx``: permutation that re-orders an action vector
        # (in joint_names order) into the robot's NATIVE DOF order, which
        # is what ``control_dofs_position(..., slice(6, 18))`` expects.
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in env_cfg["joint_names"]],
            dtype=gs.tc_int, device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        if self.control_mode == "position":
            # PD gains (Kp=35, Kd=0.5 in the reference's default_config).
            self.robot.set_dofs_kp(
                [float(env_cfg["kp"])] * self.num_actions, self.motors_dof_idx
            )
            self.robot.set_dofs_kv(
                [float(env_cfg["kd"])] * self.num_actions, self.motors_dof_idx
            )
            self._torque_limit = None
        else:
            # Torque mode: no PD. Action ∈ [-1, 1] maps to
            # ± torque_limit_scale × the MJCF actuator force range.
            _, f_hi = self.robot.get_dofs_force_range(self.motors_dof_idx)
            f_hi = f_hi[0] if f_hi.ndim == 2 else f_hi
            self._torque_limit = f_hi.to(gs.tc_float) * float(
                env_cfg.get("torque_limit_scale", 1.0)
            )
            if not torch.isfinite(self._torque_limit).all():
                raise ValueError(
                    "control_mode=torque requires finite actuator force "
                    f"ranges; got {self._torque_limit.tolist()}. Load an MJCF "
                    "with motor ctrlrange/forcerange."
                )

        # Soft & hard joint position limits (used for clamping the persistent
        # motor target and for the dof_pos_limits cost).
        lo, hi = self.robot.get_dofs_limit(self.motors_dof_idx)
        self._dof_lower = lo[0] if lo.ndim == 2 else lo
        self._dof_upper = hi[0] if hi.ndim == 2 else hi
        soft_factor = float(reward_cfg.get("soft_joint_pos_limit_factor", 0.9))
        center = (self._dof_lower + self._dof_upper) * 0.5
        half_range = (self._dof_upper - self._dof_lower) * 0.5
        self._soft_lower = center - half_range * soft_factor
        self._soft_upper = center + half_range * soft_factor

        # ----- Initial / default pose -----
        self.init_base_pos = torch.tensor(
            env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device
        )
        self.init_base_quat = torch.tensor(
            env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device
        )
        # default_dof_pos is in env_cfg.joint_names order — matches the
        # ordering of policy actions / motor_targets.
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            dtype=gs.tc_float, device=gs.device,
        )
        # init_dof_robot_order matches the robot's NATIVE DOF order — this
        # is what set_qpos consumes.
        self._init_dof_robot_order = torch.tensor(
            [env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float, device=gs.device,
        )
        self.init_qpos = torch.cat(
            (self.init_base_pos, self.init_base_quat, self._init_dof_robot_order)
        )

        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device
        )

        # ----- Task-specific config (override via reward_cfg) -----
        cls = type(self)
        self.desired_forward_vec = torch.tensor(
            reward_cfg.get("desired_forward_vec", cls.DESIRED_FORWARD),
            dtype=gs.tc_float, device=gs.device,
        )
        self.pose_joint_ids = torch.tensor(
            reward_cfg.get("pose_joint_ids", cls.POSE_JOINT_IDS),
            dtype=torch.long, device=gs.device,
        )
        self.pose_target = self.default_dof_pos[self.pose_joint_ids].clone()
        self.z_des = float(reward_cfg.get("height_target", cls.Z_DES))

        link_idx_map = {link.name: i for i, link in enumerate(self.robot.links)}
        self.unwanted_contact_idx = torch.tensor(
            [link_idx_map[n] for n in reward_cfg.get(
                "unwanted_contact_links", cls.UNWANTED_CONTACT_LINKS
            )],
            dtype=torch.long, device=gs.device,
        )
        self.off_ground_feet_idx = torch.tensor(
            [link_idx_map[n] for n in reward_cfg.get(
                "off_ground_feet", cls.OFF_GROUND_FEET
            )],
            dtype=torch.long, device=gs.device,
        )
        self.contact_force_thresh = float(
            reward_cfg.get("contact_force_threshold", 1.0)
        )
        self.upvector_term_thresh = float(
            reward_cfg.get("upvector_termination_threshold", -0.25)
        )
        self.energy_term_thresh = float(
            reward_cfg.get("energy_termination_threshold", float("inf"))
        )

        # ----- Observation config -----
        self.obs_scales: Dict[str, float] = obs_cfg["obs_scales"]
        self.noise_level = float(obs_cfg.get("noise_level", 1.0))
        ns = obs_cfg.get("noise_scales", {})
        self.noise_joint_pos = float(ns.get("joint_pos", 0.01))
        self.noise_joint_vel = float(ns.get("joint_vel", 1.5))
        self.noise_gyro = float(ns.get("gyro", 0.2))
        self.noise_gravity = float(ns.get("gravity", 0.05))
        self.noise_linvel = float(ns.get("linvel", 0.1))

        # IMU site offset in body frame (from go2_genesis.xml).
        self._imu_offset_body = torch.tensor(
            [-0.02557, 0.0, 0.04232], dtype=gs.tc_float, device=gs.device,
        ).expand(self.num_envs, 3).contiguous()

        # ----- Reward config -----
        # Per the reference, rewards are scaled by dt before being summed and
        # clipped. We fold the dt into reward_scales here so the per-step loop
        # only does one multiply per term.
        self.reward_scales: Dict[str, float] = dict(reward_cfg["reward_scales"])
        for k in self.reward_scales:
            self.reward_scales[k] *= self.dt
        self.reward_functions = {}
        for name in self.reward_scales:
            fn = getattr(self, "_reward_" + name, None)
            if fn is None:
                raise ValueError(
                    f"reward_scales['{name}'] has no matching _reward_{name} method"
                )
            self.reward_functions[name] = fn

        # ----- State buffers (shape matches existing attrs read by the
        # vecenv trace logger). -----
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.global_lin_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.global_ang_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)

        # Prefixed names retained for the vecenv trace logger (it looks them
        # up by these exact attribute names).
        self._dof_torques = torch.zeros_like(self.actions)
        self._link_contact_force = torch.zeros(
            (self.num_envs, self.robot.n_links, 3),
            dtype=gs.tc_float, device=gs.device,
        )
        self._upvector_z = torch.zeros(self.num_envs, dtype=gs.tc_float, device=gs.device)

        # Persistent motor target — Genesis equivalent of MuJoCo's data.ctrl
        # (the incremental action encoding's accumulator).
        self._motor_targets = self.default_dof_pos.unsqueeze(0).expand(
            self.num_envs, self.num_actions
        ).contiguous()

        self._body_x_ref = torch.tensor(
            [1.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device
        ).expand(self.num_envs, 3).contiguous()
        self._body_z_ref = torch.tensor(
            [0.0, 0.0, 1.0], dtype=gs.tc_float, device=gs.device
        ).expand(self.num_envs, 3).contiguous()

        self.rew_buf = torch.zeros(self.num_envs, dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=gs.tc_int, device=gs.device)
        # ``done_real``: termination excluding time-out (used by the
        # termination reward — matches the reference's ``done`` arg).
        self.done_real = torch.zeros(self.num_envs, dtype=gs.tc_bool, device=gs.device)

        self.episode_sums = {
            k: torch.zeros(self.num_envs, dtype=gs.tc_float, device=gs.device)
            for k in self.reward_scales
        }

        self.extras: Dict[str, Any] = {}

        # Pre-allocate obs buffers so get_observations is safe before the
        # first step.
        self.obs_buf = torch.zeros(
            (self.num_envs, 45), dtype=gs.tc_float, device=gs.device
        )
        self.critic_obs_buf = torch.zeros(
            (self.num_envs, 94), dtype=gs.tc_float, device=gs.device
        )

        self.reset()

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def _sample_init_qpos(self, n: int) -> torch.Tensor:
        """Per-env initial qpos with reference randomization (xy ±0.5, yaw ±π)."""
        qpos = self.init_qpos.unsqueeze(0).expand(n, -1).clone()
        dxy = torch.empty(n, 2, dtype=gs.tc_float, device=gs.device).uniform_(-0.5, 0.5)
        qpos[:, 0:2] = qpos[:, 0:2] + dxy
        # Yaw quaternion (axis = world z), composed onto the base quat as
        # base ⊗ yaw — same as MuJoCo math.quat_mul(qpos[3:7], yaw_quat).
        # Genesis: transform_quat_by_quat(v, u) computes quatmul(u, v).
        yaw = torch.empty(n, dtype=gs.tc_float, device=gs.device).uniform_(-3.14, 3.14)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        zeros = torch.zeros_like(cy)
        yaw_quat = torch.stack([cy, zeros, zeros, sy], dim=-1)
        base = qpos[:, 3:7]
        qpos[:, 3:7] = transform_quat_by_quat(yaw_quat, base)
        return qpos

    def _sample_init_qvel(self, n: int) -> torch.Tensor:
        qvel = torch.zeros(n, self.robot.n_dofs, dtype=gs.tc_float, device=gs.device)
        qvel[:, 0:6] = torch.empty(
            n, 6, dtype=gs.tc_float, device=gs.device
        ).uniform_(-0.5, 0.5)
        return qvel

    def _reset_idx(self, envs_idx=None):
        """Reset selected envs in-place.

        envs_idx: ``None`` for "reset all", or a bool mask of shape (num_envs,).
        """
        if envs_idx is None:
            idx_int = None
            n = self.num_envs
            qpos = self._sample_init_qpos(n)
            qvel = self._sample_init_qvel(n)
        else:
            if not envs_idx.any():
                return
            idx_int = envs_idx.nonzero(as_tuple=False).squeeze(-1)
            n = int(idx_int.numel())
            qpos = self._sample_init_qpos(n)
            qvel = self._sample_init_qvel(n)

        self.robot.set_qpos(qpos, envs_idx=idx_int, zero_velocity=False, skip_forward=True)
        self.robot.set_dofs_velocity(qvel, envs_idx=idx_int)

        # Per-key average episode return (logged once per reset for wandb /
        # tensorboard). Normalised by episode_length_s so longer episodes
        # don't artificially inflate the magnitudes.
        ep_len_s = float(self.env_cfg["episode_length_s"])
        self.extras["episode"] = {}
        for k, v in self.episode_sums.items():
            if envs_idx is None:
                mean = v.mean()
            else:
                if n > 0:
                    mean = v[envs_idx].sum() / n
                else:
                    mean = torch.zeros((), device=gs.device, dtype=gs.tc_float)
            self.extras["episode"]["rew_" + k] = mean / ep_len_s
            if envs_idx is None:
                v.zero_()
            else:
                v.masked_fill_(envs_idx, 0.0)

        # Clear per-env buffers for the envs we just reset. Doing this here
        # (rather than in step()) keeps the invariant: after _reset_idx, the
        # reset envs look like episode-start envs.
        if envs_idx is None:
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            self._motor_targets[:] = self.default_dof_pos
        else:
            mask = envs_idx[:, None]
            self.actions.masked_fill_(mask, 0.0)
            self.last_actions.masked_fill_(mask, 0.0)
            self.last_dof_vel.masked_fill_(mask, 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)
            self._motor_targets[envs_idx] = self.default_dof_pos

    def reset(self):
        self._reset_idx(None)
        self._refresh_state()
        self._update_observation()
        return self.get_observations()

    # ------------------------------------------------------------------ #
    # External state checkpoint / manual reset (lets an external driver
    # snapshot the start-of-rollout state and do group-coupled resets
    # rather than auto-reset).
    # ------------------------------------------------------------------ #
    def _as_int_idx(self, envs_idx):
        """Normalize idx to LongTensor of env ints (or None for all)."""
        if envs_idx is None:
            return None
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            return envs_idx.nonzero(as_tuple=False).squeeze(-1)
        return envs_idx

    def get_full_state(self, envs_idx=None):
        """Snapshot everything a future ``set_full_state`` needs to resume
        exactly: sim state (qpos, qvel) AND the persistent buffers that
        ``step`` reads/writes between calls (motor target accumulator,
        last action, last dof vel, episode-length counter).
        """
        idx = self._as_int_idx(envs_idx)
        qpos = self.robot.get_qpos(envs_idx=idx)
        qvel = self.robot.get_dofs_velocity(envs_idx=idx)
        sl = slice(None) if idx is None else idx
        return {
            "qpos": qpos.clone(),
            "qvel": qvel.clone(),
            "motor_targets": self._motor_targets[sl].clone(),
            "last_actions": self.last_actions[sl].clone(),
            "last_dof_vel": self.last_dof_vel[sl].clone(),
            "episode_length_buf": self.episode_length_buf[sl].clone(),
        }

    def set_full_state(self, state, envs_idx=None):
        """Restore env state from a ``get_full_state`` snapshot. After
        this call ``_refresh_state`` + ``_update_observation`` are run so
        ``get_observations()`` returns the obs that the snapshot would
        produce.
        """
        idx = self._as_int_idx(envs_idx)
        self.robot.set_qpos(
            state["qpos"], envs_idx=idx, zero_velocity=False, skip_forward=True
        )
        self.robot.set_dofs_velocity(state["qvel"], envs_idx=idx)
        sl = slice(None) if idx is None else idx
        self._motor_targets[sl] = state["motor_targets"]
        self.last_actions[sl] = state["last_actions"]
        self.last_dof_vel[sl] = state["last_dof_vel"]
        self.episode_length_buf[sl] = state["episode_length_buf"]
        # Re-derive cached state + obs so the next step starts from a
        # consistent view.
        self._refresh_state()
        self._update_observation()

    def reset_envs(self, envs_idx):
        """Public reset for a subset of envs (used when ``step`` was
        called with ``auto_reset=False`` and the caller wants to choose
        which envs to reset themselves).

        ``envs_idx`` is a bool mask of shape (num_envs,).
        """
        self._reset_idx(envs_idx)
        self._refresh_state()
        self._update_observation()

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, actions, auto_reset: bool = True):
        self.actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        # Commands go to the robot in NATIVE DOF order (slice(6, 18) = the
        # 12 actuated DOFs after the 6 free-joint DOFs).
        if self.control_mode == "torque":
            # ``_motor_targets`` stays frozen so state snapshots are
            # mode-uniform.
            torques = self.actions * self._torque_limit
            self.robot.control_dofs_force(
                torques[:, self.actions_dof_idx], slice(6, 18)
            )
        else:
            self._motor_targets = torch.clamp(
                self._motor_targets + self.actions * self.action_scale,
                min=self._dof_lower,
                max=self._dof_upper,
            )
            self.robot.control_dofs_position(
                self._motor_targets[:, self.actions_dof_idx], slice(6, 18)
            )
        self.scene.step()
        self.episode_length_buf += 1

        self._refresh_state()

        # Termination is computed BEFORE rewards so the termination reward
        # can read ``self.done_real``.
        fall = self._upvector_z < self.upvector_term_thresh
        unwanted_force = self._link_contact_force[:, self.unwanted_contact_idx].norm(dim=-1)
        contact_term = (unwanted_force > self.contact_force_thresh).any(dim=-1)
        energy = (self._dof_torques.abs() * self.dof_vel.abs()).sum(dim=1)
        energy_term = energy > self.energy_term_thresh
        sim_err = self.scene.rigid_solver.get_error_envs_mask()
        time_out = self.episode_length_buf > self.max_episode_length

        # done_real: matches the reference's ``done`` (no time-out).
        self.done_real = fall | contact_term |  sim_err
        self.reset_buf = self.done_real | time_out

        # Reward = clip(sum_k rew_k * scale_k * dt, 0, 10000); the dt was
        # folded into reward_scales in __init__.
        self.rew_buf.zero_()
        for name, fn in self.reward_functions.items():
            rew_k = fn() * self.reward_scales[name]
            self.rew_buf += rew_k
            self.episode_sums[name] += rew_k
        self.rew_buf.clamp_(min=0.0, max=10000.0)

        # Save last_actions / last_dof_vel BEFORE any reset so they reflect
        # the action that just executed.
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        # Update obs before any reset so the returned obs reflects the
        # pre-reset state; the reset boundary is handled via ``done`` and
        # the time_outs flag.
        self._update_observation()

        # Time-out signal for value bootstrapping. Real failure takes
        # precedence: an env that fell on the same step its clock ran out
        # must NOT be bootstrapped as a mere truncation.
        self.extras["time_outs"] = (time_out & ~self.done_real).to(gs.tc_float)

        # With ``auto_reset=False`` the caller owns the reset policy (e.g.
        # an external driver doing group-coupled resets); dead envs stay
        # dead until ``reset_envs`` / ``set_full_state``.
        if auto_reset:
            self._reset_idx(self.reset_buf)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict(
            {"policy": self.obs_buf, "critic": self.critic_obs_buf},
            batch_size=[self.num_envs],
        )

    # ------------------------------------------------------------------ #
    # State refresh + observation construction
    # ------------------------------------------------------------------ #
    def _refresh_state(self):
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        inv_base_quat = inv_quat(self.base_quat)
        self.global_lin_vel = self.robot.get_vel()
        self.global_ang_vel = self.robot.get_ang()
        self.base_lin_vel = transform_by_quat(self.global_lin_vel, inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.global_ang_vel, inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self._dof_torques = self.robot.get_dofs_control_force(self.motors_dof_idx)
        self._link_contact_force = self.robot.get_links_net_contact_force()
        body_z_world = transform_by_quat(self._body_z_ref, self.base_quat)
        self._upvector_z = body_z_world[:, 2]

    def _update_observation(self):
        gyro = self.base_ang_vel              # body-frame angular vel (= MJ gyro sensor).
        gravity = self.projected_gravity      # body-frame gravity direction.
        joint_angles = self.dof_pos
        joint_vel = self.dof_vel
        linvel = self.base_lin_vel            # body-frame linear velocity.
        last_act = self.last_actions

        if self.noise_level > 0:
            def noisy(x, scale):
                u = 2.0 * torch.rand_like(x) - 1.0
                return x + u * (self.noise_level * scale)
            n_gyro = noisy(gyro, self.noise_gyro)
            n_gravity = noisy(gravity, self.noise_gravity)
            n_jpos = noisy(joint_angles, self.noise_joint_pos)
            n_jvel = noisy(joint_vel, self.noise_joint_vel)
            n_linvel = noisy(linvel, self.noise_linvel)
        else:
            n_gyro, n_gravity = gyro, gravity
            n_jpos, n_jvel, n_linvel = joint_angles, joint_vel, linvel

        # Policy obs (45D) — order matches the reference's _get_obs hstack.
        # The reference does not apply obs_scales; we multiply by them so the
        # existing YAML's obs_scales (lin_vel=2.0, ang_vel=0.25, dof_vel=0.05,
        # dof_pos=1.0) can be tuned independently. Set them all to 1.0 in the
        # YAML to recover the reference's raw-value policy obs exactly.
        self.obs_buf = torch.cat(
            (
                n_linvel * self.obs_scales["lin_vel"],                                  # 3
                n_gyro * self.obs_scales["ang_vel"],                                    # 3
                n_gravity,                                                              # 3
                (n_jpos - self.default_dof_pos) * self.obs_scales["dof_pos"],           # 12
                n_jvel * self.obs_scales["dof_vel"],                                    # 12
                last_act,                                                               # 12
            ),
            dim=-1,
        )

        # Privileged (critic) obs (94D) = [policy, clean gyro, accel,
        # clean local linvel, clean global angvel, clean joint pos, clean
        # joint vel, joint torques, IMU world z].
        accelerometer = torch.zeros_like(linvel)
        imu_world = self.base_pos + transform_by_quat(self._imu_offset_body, self.base_quat)
        torso_height = imu_world[:, 2:3]
        self.critic_obs_buf = torch.cat(
            (
                self.obs_buf,           # 45
                gyro,                   # 3
                accelerometer,          # 3 (zeros — Genesis has no accel sensor)
                linvel,                 # 3
                self.global_ang_vel,    # 3 (world-frame, matches get_global_angvel)
                joint_angles,           # 12
                joint_vel,              # 12
                self._dof_torques,      # 12
                torso_height,           # 1
            ),
            dim=-1,
        )

    # ------------------------------------------------------------------ #
    # Reward functions — names match reward_cfg.reward_scales keys.
    # Each returns shape (num_envs,) BEFORE multiplication by scale*dt.
    # ------------------------------------------------------------------ #
    def _reward_height(self):
        imu_world = self.base_pos + transform_by_quat(
            self._imu_offset_body, self.base_quat
        )
        height = torch.clamp(imu_world[:, 2], max=self.z_des)
        return torch.exp(-(self.z_des - height))

    # Backward-compat alias: the existing YAML uses ``handstand_height``.
    def _reward_handstand_height(self):
        return self._reward_height()

    def _reward_orientation(self):
        forward = transform_by_quat(self._body_x_ref, self.base_quat)
        cos_dist = (forward * self.desired_forward_vec).sum(dim=-1)
        normalized = 0.5 * cos_dist + 0.5
        return normalized.square()

    def _reward_contact(self):
        f = self._link_contact_force[:, self.off_ground_feet_idx].norm(dim=-1)
        return (f > self.contact_force_thresh).any(dim=-1).to(gs.tc_float)

    def _reward_action_rate(self):
        return (self.actions - self.last_actions).square().sum(dim=1)

    def _reward_torques(self):
        return self._dof_torques.square().sum(dim=1)

    def _reward_termination(self):
        # Matches the reference: termination cost fires only on "real"
        # terminations (fall/contact/energy), NOT on time-outs.
        return self.done_real.to(gs.tc_float)

    def _reward_dof_pos_limits(self):
        below = torch.clamp(self.dof_pos - self._soft_lower, max=0.0)
        above = torch.clamp(self.dof_pos - self._soft_upper, min=0.0)
        return (-below + above).sum(dim=1)

    def _reward_dof_acc(self):
        acc = (self.dof_vel - self.last_dof_vel) / self.dt
        return acc.square().sum(dim=1)

    def _reward_pose(self):
        diff = self.dof_pos[:, self.pose_joint_ids] - self.pose_target
        return diff.square().sum(dim=1)

    def _reward_stay_still(self):
        # Reference uses qvel[:6] which is world-frame for the free joint:
        # qvel[:3] = world linear vel, qvel[3:6] = world angular vel,
        # qvel[5] = world yaw rate.
        return (
            self.global_lin_vel[:, :2].square().sum(dim=1)
            + self.global_ang_vel[:, 2].square()
        )

    def _reward_energy(self):
        return (self.dof_vel.abs() * self._dof_torques.abs()).sum(dim=1)


class SingleHandstand(Handstand):
    """Single-front-leg handstand task for Go2.

    Balances on one front leg while lifting the other; rear legs stay in
    the air. Mirrors mujoco_playground's ``SingleHandstand``:
      * ``env_cfg.support_leg`` selects which front leg supports the body
        ("FR" or "FL"; default "FR"). The other front leg is the "lifted"
        leg.
      * The lifted front foot is added to ``off_ground_feet`` — the rear
        feet AND the lifted front foot all contribute to the contact cost.
      * Pose regularisation switches from "both front hips + all rear" to
        "lifted-leg full pose + all rear" so the support leg is free to
        bend while the lifted leg is held near its default joint angles.
      * Two extra rewards: ``lifted_foot_height`` (encourages the lifted
        foot's world z to reach ``z_des``) and ``com_over_support``
        (encourages the robot CoM's xy to sit over the support foot's xy).

    Joint-name index map (from ``cfgs/env/go2_single_handstand.yaml``):
        0-2: FR (hip, thigh, calf)    6-8:  RR
        3-5: FL                        9-11: RL
    """

    Z_DES: float = 0.50

    # The lifted-foot-height / com-over-support reward shaping matches the
    # MJX reference: foot_z / z_des clipped to [0, 1], exp(-d^2 / 0.01).
    COM_OVER_SUPPORT_SIGMA_SQ: float = 0.01

    # support leg -> (lifted leg, lifted-leg joint ids in joint_names order).
    # Joint-name index map: 0-2 FR, 3-5 FL, 6-8 RR, 9-11 RL.
    SUPPORT_LEGS = {"FR": ("FL", [3, 4, 5]), "FL": ("FR", [0, 1, 2])}
    DEFAULT_SUPPORT = "FR"
    # Joints of the two non-balancing legs (regularized toward home pose
    # alongside the lifted leg's).
    FREE_LEG_POSE_IDS: List[int] = [6, 7, 8, 9, 10, 11]

    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg=None,
        show_viewer: bool = False,
    ):
        cls = type(self)
        support = env_cfg.get("support_leg", cls.DEFAULT_SUPPORT)
        try:
            lifted, lifted_joint_ids = cls.SUPPORT_LEGS[support]
        except KeyError:
            raise ValueError(
                f"env_cfg.support_leg must be one of "
                f"{sorted(cls.SUPPORT_LEGS)}, got '{support}'"
            ) from None

        self._support_name = support
        self._lifted_name = lifted

        # Inject support-dependent defaults so the parent's __init__ picks
        # them up via reward_cfg.get(...). Explicit YAML values still win.
        reward_cfg = dict(reward_cfg)
        reward_cfg.setdefault(
            "pose_joint_ids", lifted_joint_ids + list(cls.FREE_LEG_POSE_IDS)
        )
        reward_cfg.setdefault(
            "off_ground_feet",
            [f"{lifted}_foot"] + list(cls.OFF_GROUND_FEET),
        )

        super().__init__(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )

        link_idx_map = {link.name: i for i, link in enumerate(self.robot.links)}
        self._support_foot_idx = link_idx_map[f"{support}_foot"]
        self._lifted_foot_idx = link_idx_map[f"{lifted}_foot"]

        # Cache link masses for the mass-weighted CoM (link masses are
        # constant across steps). Shape (n_envs, n_links) for broadcast.
        link_mass = self.robot.get_links_inertial_mass()
        if link_mass.dim() == 1:
            link_mass = link_mass.unsqueeze(0).expand(
                self.num_envs, -1
            ).contiguous()
        self._link_mass = link_mass
        self._total_mass = link_mass.sum(dim=-1, keepdim=True)

    # ------------------------------------------------------------------ #
    # Extra reward terms. Picked up automatically by Handstand.__init__'s
    # auto-wiring loop as long as the YAML's reward_scales lists them.
    # ------------------------------------------------------------------ #
    def _reward_lifted_foot_height(self):
        foot_z = self.robot.get_links_pos()[:, self._lifted_foot_idx, 2]
        return torch.clamp(foot_z / self.z_des, 0.0, 1.0)

    def _reward_com_over_support(self):
        # ref="link_com" so each link contributes its own CoM, not its
        # frame origin — closest match to MuJoCo's data.subtree_com.
        link_com = self.robot.get_links_pos(ref="link_com")
        com_xy = (
            (link_com[..., :2] * self._link_mass.unsqueeze(-1)).sum(dim=1)
            / self._total_mass
        )
        foot_xy = self.robot.get_links_pos()[:, self._support_foot_idx, :2]
        dist_sq = (com_xy - foot_xy).square().sum(dim=-1)
        return torch.exp(-dist_sq / self.COM_OVER_SUPPORT_SIGMA_SQ)
