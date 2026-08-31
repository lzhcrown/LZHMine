"""Runtime-neutral Nezha policy adapter.

Connect ``NezhaPolicyRuntime.step`` to the robot SDK transport.  This module
contains no Isaac Gym, MuJoCo or vendor communication dependency, which keeps
the exact observation/control contract reusable on the real robot.
"""

from typing import Dict, List, Sequence, Tuple

import torch


NUM_ACTIONS = 16
NUM_ONE_STEP_OBS = 65
LEG_JOINT_IDS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
WHEEL_JOINT_IDS = [3, 7, 11, 15]
MODE_NAMES = ("wheel", "leg", "hybrid")


class NezhaObservationHistory:
    """Build the newest-first 2x65 observation consumed by the policy."""

    def __init__(
        self,
        default_dof_pos: Sequence[float],
        history_steps: int = 2,
        scales: Dict[str, float] = None,
        device: str = "cpu",
    ):
        if len(default_dof_pos) != NUM_ACTIONS:
            raise ValueError("default_dof_pos must contain 16 values")
        self.device = torch.device(device)
        self.default_dof_pos = torch.tensor(
            default_dof_pos, dtype=torch.float32, device=self.device
        )
        self.history_steps = history_steps
        self.scales = scales or {
            "lin_vel": 2.0,
            "ang_vel": 1.0,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        }
        self.history = torch.zeros(
            history_steps, NUM_ONE_STEP_OBS, dtype=torch.float32, device=self.device
        )

    def reset(self, first_observation: torch.Tensor = None):
        self.history.zero_()
        if first_observation is not None:
            if first_observation.shape != (NUM_ONE_STEP_OBS,):
                raise ValueError("first_observation must have shape [65]")
            self.history[:] = first_observation

    def build_one_step(
        self,
        base_ang_vel: Sequence[float],
        projected_gravity: Sequence[float],
        commands: Sequence[float],
        dof_pos: Sequence[float],
        dof_vel: Sequence[float],
        previous_actions: Sequence[float],
    ) -> torch.Tensor:
        tensors = [
            torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for value in (
                base_ang_vel,
                projected_gravity,
                commands,
                dof_pos,
                dof_vel,
                previous_actions,
            )
        ]
        ang_vel, gravity, command, position, velocity, actions = tensors
        expected_shapes = [(3,), (3,), (3,), (16,), (16,), (16,)]
        for tensor, shape in zip(tensors, expected_shapes):
            if tensor.shape != shape:
                raise ValueError(f"Runtime input has shape {tensor.shape}, expected {shape}")
        leg_ids = torch.tensor(LEG_JOINT_IDS, device=self.device)
        position_error = (position - self.default_dof_pos) * self.scales["dof_pos"]
        return torch.cat(
            (
                ang_vel * self.scales["ang_vel"],
                gravity,
                command
                * torch.tensor(
                    [
                        self.scales["lin_vel"],
                        self.scales["lin_vel"],
                        self.scales["ang_vel"],
                    ],
                    device=self.device,
                ),
                position_error.index_select(0, leg_ids),
                velocity * self.scales["dof_vel"],
                (position * self.scales["dof_pos"]).index_select(0, leg_ids),
                actions,
            )
        )

    def push(self, one_step_observation: torch.Tensor) -> torch.Tensor:
        if one_step_observation.shape != (NUM_ONE_STEP_OBS,):
            raise ValueError("one_step_observation must have shape [65]")
        self.history[1:] = self.history[:-1].clone()
        self.history[0] = one_step_observation
        return self.history.reshape(1, -1)


class NezhaPolicyRuntime:
    """TorchScript inference plus Nezha hybrid PD/wheel control targets."""

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
        history_steps: int = 2,
        observation_scales: Dict[str, float] = None,
    ):
        for name, values in (
            ("default_dof_pos", default_dof_pos),
            ("p_gains", p_gains),
            ("d_gains", d_gains),
        ):
            if len(values) != NUM_ACTIONS:
                raise ValueError(f"{name} must contain 16 values")
        self.device = torch.device(device)
        self.policy = torch.jit.load(policy_path, map_location=self.device).eval()
        if not hasattr(self.policy, "get_mode_probabilities"):
            raise RuntimeError(
                "The policy is not an integrated gated-modal export. "
                "Re-export it with play_nezha_mine.py."
            )
        self.observations = NezhaObservationHistory(
            default_dof_pos,
            history_steps=history_steps,
            scales=observation_scales,
            device=device,
        )
        self.default_dof_pos = torch.tensor(
            default_dof_pos, dtype=torch.float32, device=self.device
        )
        self.p_gains = torch.tensor(p_gains, dtype=torch.float32, device=self.device)
        self.d_gains = torch.tensor(d_gains, dtype=torch.float32, device=self.device)
        limits = torque_limits if torque_limits is not None else [float("inf")] * 16
        self.torque_limits = torch.tensor(limits, dtype=torch.float32, device=self.device)
        self.action_scale = action_scale
        self.wheel_velocity_scale = wheel_velocity_scale
        self.previous_actions = torch.zeros(NUM_ACTIONS, device=self.device)
        self.last_mode_probabilities = torch.zeros(3, device=self.device)

    def reset(self):
        self.previous_actions.zero_()
        self.last_mode_probabilities.zero_()
        self.observations.reset()

    def mode_probabilities(self) -> torch.Tensor:
        """Return the last [wheel, leg, hybrid] gate probabilities."""
        return self.last_mode_probabilities.clone()

    @torch.inference_mode()
    def step(
        self,
        base_ang_vel: Sequence[float],
        projected_gravity: Sequence[float],
        commands: Sequence[float],
        dof_pos: Sequence[float],
        dof_vel: Sequence[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        one_step = self.observations.build_one_step(
            base_ang_vel,
            projected_gravity,
            commands,
            dof_pos,
            dof_vel,
            self.previous_actions,
        )
        history = self.observations.push(one_step)
        actions = self.policy(history).squeeze(0).clamp(-100.0, 100.0)
        probabilities = self.policy.get_mode_probabilities(history).squeeze(0)
        if probabilities.shape != (3,):
            raise RuntimeError(
                f"Policy returned gate probabilities {probabilities.shape}; expected [3]"
            )
        self.last_mode_probabilities.copy_(probabilities)
        if actions.shape != (NUM_ACTIONS,):
            raise RuntimeError(f"Policy returned {actions.shape}; expected [16]")

        position = torch.as_tensor(dof_pos, dtype=torch.float32, device=self.device)
        velocity = torch.as_tensor(dof_vel, dtype=torch.float32, device=self.device)
        position_target = self.default_dof_pos + actions * self.action_scale
        velocity_target = torch.zeros_like(actions)
        wheel_ids = torch.tensor(WHEEL_JOINT_IDS, device=self.device)
        position_target[wheel_ids] = position[wheel_ids]
        velocity_target[wheel_ids] = (
            actions[wheel_ids] * self.wheel_velocity_scale
        )
        torques = self.p_gains * (position_target - position) + self.d_gains * (
            velocity_target - velocity
        )
        torques = torch.clamp(torques, -self.torque_limits, self.torque_limits)
        self.previous_actions.copy_(actions)
        return actions, torques
