"""Student-only Nezha WF-CTS inference, with the training observation schema."""

import json
from typing import Sequence, Tuple

import torch

from rsl_rl.utils.cts_observations import CTSObservationHistory, build_proprioception

NUM_ACTIONS = 16


class NezhaCTSPolicyRuntime:
    """Keep history outside TorchScript and reuse Nezha's hybrid torque law.

    Observation preprocessing is read from the exported model. In particular,
    wheel positions are zeroed but measured wheel velocities are retained.
    """

    def __init__(
        self,
        policy_path: str,
        default_dof_pos: Sequence[float],
        p_gains: Sequence[float],
        d_gains: Sequence[float],
        action_scale: float = 0.25,
        wheel_velocity_scale: float = 10.0,
        torque_limits: Sequence[float] = None,
        device: str = "cpu",
        history_steps: int = None,
        control_dt: float = 0.02,
        joint_names: Sequence[str] = None,
        policy=None,
    ):
        self.device = torch.device(device)
        self.policy = (
            torch.jit.load(policy_path, map_location=self.device)
            if policy is None else policy.to(self.device)
        ).eval()
        if not hasattr(self.policy, "get_architecture") or self.policy.get_architecture() != "nezha_moe_cts":
            raise ValueError("Expected a nezha_moe_cts student export")
        self.architecture = "nezha_moe_cts"
        self.observation_config = json.loads(self.policy.get_observation_config_json())
        self.control_config = json.loads(self.policy.get_control_config_json())
        self.clip_actions = float(self.control_config.get("clip_actions", 100.0))
        if self.clip_actions <= 0:
            raise ValueError("clip_actions must be positive")
        if "wheel_indices" in self.control_config:
            if self.control_config["wheel_indices"] != self.observation_config["wheel_indices"]:
                raise ValueError("Control and observation metadata use different wheel indices")
        supplied_control = {
            "default_dof_pos": default_dof_pos,
            "p_gains": p_gains,
            "d_gains": d_gains,
            "action_scale": action_scale,
            "wheel_velocity_scale": wheel_velocity_scale,
            "control_dt": control_dt,
        }
        if torque_limits is not None:
            supplied_control["torque_limits"] = torque_limits
        for name, supplied in supplied_control.items():
            if name in self.control_config:
                saved = torch.as_tensor(self.control_config[name], dtype=torch.float32)
                actual = torch.as_tensor(supplied, dtype=torch.float32)
                if saved.shape != actual.shape or not torch.allclose(saved, actual, atol=1e-6, rtol=1e-5):
                    raise ValueError("Deployment {} differs from training metadata; update the deployment YAML".format(name))
        if joint_names is not None and "joint_names" in self.control_config:
            if list(joint_names) != self.control_config["joint_names"]:
                raise ValueError("Deployment joint order differs from training metadata")
        model_history_steps = int(self.observation_config["history_length"])
        if history_steps is not None and int(history_steps) != model_history_steps:
            raise ValueError("history_steps conflicts with the exported observation schema")
        self.observations = CTSObservationHistory(
            1, config=self.observation_config, device=device
        )
        observation_width = sum(term["dim"] for term in self.observation_config["proprioceptive_terms"])
        if model_history_steps * observation_width != int(self.policy.get_num_actor_obs()):
            raise ValueError("Exported history schema does not match policy input dimensions")
        for name, values in (
            ("default_dof_pos", default_dof_pos),
            ("p_gains", p_gains),
            ("d_gains", d_gains),
        ):
            if len(values) != NUM_ACTIONS:
                raise ValueError("{} must contain 16 values".format(name))
        if torque_limits is not None and len(torque_limits) != NUM_ACTIONS:
            raise ValueError("torque_limits must contain 16 values")
        self.default_dof_pos = torch.tensor(default_dof_pos, dtype=torch.float32, device=self.device)
        self.p_gains = torch.tensor(p_gains, dtype=torch.float32, device=self.device)
        self.d_gains = torch.tensor(d_gains, dtype=torch.float32, device=self.device)
        limits = torque_limits if torque_limits is not None else self.control_config.get("torque_limits", [float("inf")] * NUM_ACTIONS)
        self.torque_limits = torch.tensor(limits, dtype=torch.float32, device=self.device)
        self.action_scale = float(action_scale)
        self.wheel_velocity_scale = float(wheel_velocity_scale)
        self.wheel_ids = torch.tensor(self.observation_config["wheel_indices"], dtype=torch.long, device=self.device)
        self.previous_actions = torch.zeros(NUM_ACTIONS, device=self.device)
        self.num_experts = int(self.policy.get_num_experts())
        self.expert_names = tuple("expert_{}".format(i) for i in range(self.num_experts))
        self.last_expert_probabilities = torch.zeros(self.num_experts, device=self.device)

    def reset(self):
        self.previous_actions.zero_()
        self.last_expert_probabilities.zero_()
        self.observations.reset()

    def expert_probabilities(self) -> torch.Tensor:
        return self.last_expert_probabilities.clone()

    def compute_torques(self, actions, dof_pos, dof_vel):
        """Recompute leg-position PD and wheel-velocity control each physics step."""
        position = torch.as_tensor(dof_pos, dtype=torch.float32, device=self.device)
        velocity = torch.as_tensor(dof_vel, dtype=torch.float32, device=self.device)
        position_target = self.default_dof_pos + actions * self.action_scale
        velocity_target = torch.zeros_like(actions)
        position_target[self.wheel_ids] = position[self.wheel_ids]
        velocity_target[self.wheel_ids] = actions[self.wheel_ids] * self.wheel_velocity_scale
        torques = self.p_gains * (position_target - position) + self.d_gains * (velocity_target - velocity)
        return torch.clamp(torques, -self.torque_limits, self.torque_limits)

    def build_one_step(self, base_ang_vel, projected_gravity, commands, dof_pos, dof_vel):
        values = (base_ang_vel, projected_gravity, commands, dof_pos, dof_vel)
        tensors = [torch.as_tensor(v, dtype=torch.float32, device=self.device) for v in values]
        for tensor, width in zip(tensors, (3, 3, 3, 16, 16)):
            if tensor.shape != (width,):
                raise ValueError("Runtime input has shape {}; expected [{}]".format(tuple(tensor.shape), width))
        angular, gravity, command, position, velocity = tensors
        return build_proprioception(
            {
                "base_ang_vel": angular.unsqueeze(0),
                "projected_gravity": gravity.unsqueeze(0),
                "velocity_commands": command.unsqueeze(0),
                "joint_pos": (position - self.default_dof_pos).unsqueeze(0),
                "joint_vel": velocity.unsqueeze(0),
                "actions": self.previous_actions.unsqueeze(0),
            },
            config=self.observation_config,
            add_noise=False,
        )

    @torch.inference_mode()
    def step(self, base_ang_vel, projected_gravity, commands, dof_pos, dof_vel) -> Tuple[torch.Tensor, torch.Tensor]:
        frame = self.build_one_step(base_ang_vel, projected_gravity, commands, dof_pos, dof_vel)
        history = self.observations.update(frame)
        actions = self.policy(history).squeeze(0).clamp(-self.clip_actions, self.clip_actions)
        if actions.shape != (NUM_ACTIONS,) or not torch.isfinite(actions).all():
            raise RuntimeError("CTS policy must return 16 finite actions")
        probabilities = self.policy.get_expert_probabilities(history).squeeze(0)
        if probabilities.shape != (self.num_experts,) or not torch.isfinite(probabilities).all():
            raise RuntimeError("CTS policy returned invalid expert weights")
        self.last_expert_probabilities.copy_(probabilities)
        torques = self.compute_torques(actions, dof_pos, dof_vel)
        self.previous_actions.copy_(actions)
        return actions, torques
