"""Checkpoint comparisons must preserve state-dependent environment resets."""

from types import SimpleNamespace

import torch

from rsl_rl.runners.on_policy_runner_cts import MoECTSOnPolicyRunner
from rsl_rl.utils.wf_curriculum import WFCurriculum, default_wf_command_config
from test_cts_initialization import all_moe_config
from test_cts_runner import FakeVectorEnv


class StatefulWFCurriculumEnv(FakeVectorEnv):
    """Use Nezha's reset order and real WF decisions, without Isaac physics.

    Terrain promotion precedes clearing the old episode's displacement and
    command statistics. Reseeding only the second reset cannot undo changes
    introduced by the runner's first reset.
    """

    num_envs = 512
    max_episode_length = 1250

    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.dt = 0.02
        self.max_episode_length_s = 25.0
        settings = default_wf_command_config()
        settings["range_curriculum"] = "nezha_reward"
        self.cfg = SimpleNamespace(
            commands=SimpleNamespace(curriculum=True, wf_settings=settings),
            terrain=SimpleNamespace(
                mesh_type="trimesh", terrain_length=9.0,
                sub_terrain_border_width=0.5,
            ),
        )
        self.command_ranges = {
            "lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-0.6, 0.6],
            "ang_vel_yaw": [-1.0, 1.0],
        }
        self.terrain = SimpleNamespace(column_types=["flat"])
        self.terrain_types = torch.zeros(self.num_envs, dtype=torch.long)
        self.terrain_levels = torch.full((self.num_envs,), 5, dtype=torch.long)
        self.root_states = torch.zeros(self.num_envs, 13)
        self.env_origins = torch.zeros(self.num_envs, 3)
        self.commands = torch.zeros(self.num_envs, 4)
        self.reward_scales = {}
        self.wf_curriculum = WFCurriculum(self)

    def reset(self):
        ids = torch.arange(self.num_envs)
        move_up, move_down = self.wf_curriculum.terrain_move_masks(ids)
        self.terrain_levels.add_(move_up.long() - move_down.long()).clamp_(min=0)
        self.wf_curriculum.update_reward_curriculum()
        self.root_states[:, :2] = 2.0 * torch.rand(self.num_envs, 2) - 1.0
        # Like Nezha.reset_idx, sample commands before resetting episode length.
        self.wf_curriculum.reset(ids)
        super().reset()
        # BaseTask.reset also performs a zero-action control step. Its position
        # statistic is enough here to exercise the next reset's terrain decision.
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self.wf_curriculum.step()
        return self.observations, self.privileged


def test_four_arms_have_identical_stateful_curriculum_initialization(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(12)
        source = MoECTSOnPolicyRunner(StatefulWFCurriculumEnv(), all_moe_config())
        source.current_learning_iteration = 20000
        source.env.common_step_counter = 20000 * source.num_steps_per_env
        source.env.command_ranges["lin_vel_x"] = [-1.4, 1.4]
        source.env.wf_curriculum.update_command_ranges()
        checkpoint = tmp_path / "g7.pt"
        source.save(checkpoint)

        snapshots = []
        for residual, temporal in ((False, False), (True, False),
                                   (False, True), (True, True)):
            torch.manual_seed(42)
            env = StatefulWFCurriculumEnv()
            runner = MoECTSOnPolicyRunner(env, all_moe_config(residual, temporal))
            runner.initialize_from(checkpoint)
            assert env.reset_calls == 2
            assert env.common_step_counter == source.env.common_step_counter + 1
            assert env.command_ranges == source.env.command_ranges
            snapshots.append({
                "terrain_levels": env.terrain_levels.clone(),
                "root_states": env.root_states.clone(),
                "commands": env.commands.clone(),
                "max_move_distance": env.wf_curriculum.max_move_distance.clone(),
                "commands_xy_accumulation": env.wf_curriculum.commands_xy_accumulation.clone(),
                "rng_state": torch.get_rng_state(),
            })

        # Ensure the test actually exercises the state-dependent level update.
        assert (snapshots[0]["terrain_levels"] < 5).any()
        for state in snapshots[1:]:
            for name, expected in snapshots[0].items():
                torch.testing.assert_close(state[name], expected, rtol=0, atol=0,
                                           msg="Comparison arms differ in " + name)
    finally:
        torch.set_num_threads(previous_threads)
