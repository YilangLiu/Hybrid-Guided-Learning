"""rl_games IVecEnv wrapper around Genesis Go2-style envs.

Supports both single-tensor observations and asymmetric {policy, critic}
observations (what Handstand / SingleHandstand return). When the inner env
returns a 'critic' obs, this wrapper exposes a `state_space` so rl_games'
`central_value_config` can wire up the privileged critic.
"""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_games.common.ivecenv import IVecEnv


class Go2VecEnv(IVecEnv):
    """Adapts a Handstand-style Genesis env to rl_games' contract.

    The first positional arg is the env *class* — passed via a closure from
    `register.py`. `kwargs` carries the `env_config:` block from the YAML
    (env_cfg / obs_cfg / reward_cfg / command_cfg).
    """

    def __init__(self, env_class, config_name: str, num_actors: int, **kwargs):
        env_cfg = kwargs["env_cfg"]
        obs_cfg = kwargs["obs_cfg"]
        reward_cfg = kwargs["reward_cfg"]
        command_cfg = kwargs["command_cfg"]
        show_viewer = kwargs.get("show_viewer", False)

        # Optional per-step trajectory logging (used by `train.py --trace`).
        self._trace = bool(kwargs.get("trace", False))
        self._trace_path = kwargs.get("trace_path", None)
        self._trace_log: list = []
        if self._trace:
            import atexit
            atexit.register(self._save_trace)

        self.env = env_class(
            num_envs=num_actors,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )
        self.num_envs = num_actors

        # Probe observations once. We support both:
        #   - plain TensorDict({"policy": ...})  -> symmetric (walking)
        #   - TensorDict({"policy": ..., "critic": ...}) -> asymmetric
        obs_td = self.env.get_observations()
        policy_obs = obs_td["policy"]
        self.has_critic_obs = "critic" in obs_td.keys()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(int(policy_obs.shape[-1]),), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(int(env_cfg["num_actions"]),), dtype=np.float32,
        )

        if self.has_critic_obs:
            critic_dim = int(obs_td["critic"].shape[-1])
            self.state_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(critic_dim,), dtype=np.float32,
            )

    # --- IVecEnv API -----------------------------------------------------

    def step(self, actions: torch.Tensor):
        obs_td, rew, done, info = self.env.step(actions)
        if self._trace:
            self._record_step(actions, obs_td, rew, done)
        return self._pack_obs(obs_td), rew, done.to(torch.float32), info

    def _record_step(self, actions, obs_td, rew, done):
        """Append one step of fully diagnosable state to the in-memory log.

        We snapshot what's needed to reconstruct the rollout offline:
          - the policy obs the actor saw,
          - the action it issued,
          - the resulting reward / done,
          - the env's internal kinematics (base pose, joint pos/vel),
          - the motor target (Handstand's incremental target) so we can spot
            policies that pin joints against the limits.
        """
        env = self.env
        entry = {
            "policy_obs": obs_td["policy"].detach().cpu(),
            "actions": actions.detach().cpu(),
            "reward": rew.detach().cpu(),
            "done": done.detach().cpu(),
            "base_pos": env.base_pos.detach().cpu(),
            "base_quat": env.base_quat.detach().cpu(),
            "base_lin_vel": env.base_lin_vel.detach().cpu(),
            "base_ang_vel": env.base_ang_vel.detach().cpu(),
            "projected_gravity": env.projected_gravity.detach().cpu(),
            "dof_pos": env.dof_pos.detach().cpu(),
            "dof_vel": env.dof_vel.detach().cpu(),
        }
        # Handstand-specific bonuses (cached in step()).
        for attr in ("_motor_targets", "_dof_torques", "_upvector_z",
                     "_link_contact_force"):
            if hasattr(env, attr):
                entry[attr.lstrip("_")] = getattr(env, attr).detach().cpu()
        # Critic obs only present in the asymmetric tasks.
        if "critic" in obs_td.keys():
            entry["critic_obs"] = obs_td["critic"].detach().cpu()
        self._trace_log.append(entry)

    def _save_trace(self):
        if not self._trace_log:
            return
        import time, os
        path = self._trace_path
        if not path:
            os.makedirs("runs/eval_traces", exist_ok=True)
            path = f"runs/eval_traces/trace_{int(time.time())}.pt"
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {"steps": self._trace_log, "num_envs": self.num_envs},
            path,
        )
        print(f"[Go2VecEnv] Saved {len(self._trace_log)} steps -> {path}")

    def reset(self):
        return self._pack_obs(self.env.reset())

    def _pack_obs(self, obs_td):
        """Format observations the way rl_games' `obs_to_tensors` expects.

        When central_value_config is enabled, rl_games reads `obs['obs']`
        for the actor and `obs['states']` for the critic. With no central
        value, it just wraps the bare tensor.
        """
        if self.has_critic_obs:
            return {"obs": obs_td["policy"], "states": obs_td["critic"]}
        return obs_td["policy"]

    def get_number_of_agents(self) -> int:
        return 1

    def get_env_info(self) -> dict:
        info = {
            "action_space": self.action_space,
            "observation_space": self.observation_space,
        }
        if self.has_critic_obs:
            info["state_space"] = self.state_space
            info["use_global_observations"] = True
        return info

    def seed(self, seed):  # Genesis seed is handled in gs.init()
        pass

    def close(self):
        pass
