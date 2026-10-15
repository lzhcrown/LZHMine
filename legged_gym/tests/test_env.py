# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

def run_env_smoke(args):
    """Manual short physics regression; ordinary pytest collection stays CPU-only.

    Example (activate the training environment first)::

        python legged_gym/tests/test_env.py --headless --rl_device cpu \
            --sim_device cpu --smoke_terrain trimesh
    """
    import isaacgym  # noqa: F401 - must precede torch
    import torch
    import legged_gym.envs  # noqa: F401 - registers the task
    from legged_gym.utils import task_registry, class_to_dict, update_class_from_dict
    from rsl_rl.utils.cts_observations import extract_current_observation

    args.task = "nezha_moe_cts"
    args.resume = False
    args.headless = True
    args.num_envs = min(args.num_envs or 16, 16)
    if args.num_envs < 1 or args.smoke_steps < 1:
        raise ValueError("num_envs and smoke_steps must be positive")
    cfg, _ = task_registry.get_cfgs(args.task)
    # Real checkpoints restore nested config instances before constructing Gym.
    saved_cfg = class_to_dict(cfg)
    cfg.env.frame_stack = 7
    cfg.env.num_envs = 3
    cfg.control.action_scale = 0.9
    update_class_from_dict(cfg, saved_cfg)
    assert cfg.env.frame_stack == saved_cfg["env"]["frame_stack"]
    assert cfg.env.num_envs == saved_cfg["env"]["num_envs"]
    assert cfg.control.action_scale == saved_cfg["control"]["action_scale"]
    cfg.terrain.mesh_type = args.smoke_terrain
    cfg.terrain.num_rows = 2
    cfg.terrain.num_cols = 2
    cfg.terrain.max_init_terrain_level = 1
    cfg.terrain.terrain_length = 4.0
    cfg.terrain.terrain_width = 4.0
    cfg.terrain.curriculum = False
    cfg.commands.curriculum = False
    cfg.noise.add_noise = False
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_joint_friction = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    try:
        obs, privileged = env.reset()
        assert obs.shape == (env.num_envs, 228)
        assert privileged.shape == (env.num_envs, 295)
        assert torch.isfinite(obs).all() and torch.isfinite(privileged).all()
        assert env.wheel_body_indices.unique().numel() == 4
        assert env.wheel_body_indices.min() >= 0
        dims = [item["dim"] for item in env.observation_config["proprioceptive_terms"]]
        torch.testing.assert_close(extract_current_observation(obs, 4, dims), env.current_obs)
        torch.testing.assert_close(
            env.observation_history.buffer,
            env.current_obs[:, None, :].expand(-1, 4, -1),
        )
        assert env.height_points.shape == (env.num_envs, 187, 3)
        torch.testing.assert_close(
            env.height_points[0, :17, 0],
            torch.tensor(cfg.terrain.measured_points_x, device=env.device),
        )
        torch.testing.assert_close(
            env.height_points[0, :17, 1],
            torch.full((17,), -0.5, device=env.device),
        )
        before_substep = [None]
        compute_torques = env._compute_torques

        def traced_torques(actions):
            before_substep[0] = env.dof_vel.clone()
            return compute_torques(actions)

        env._compute_torques = traced_torques
        action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        action[:, env.wheel_indices] = 0.3
        with torch.inference_mode():
            for _ in range(args.smoke_steps):
                result = env.step(action)
                assert len(result) == 7
                assert torch.isfinite(result[0]).all() and torch.isfinite(result[1]).all()
                alive = ~result[3].bool()
                torch.testing.assert_close(
                    env.dof_acc[alive],
                    (env.dof_vel[alive] - before_substep[0][alive]) / env.sim_params.dt,
                )
                torch.testing.assert_close(env.current_obs[:, 25:41], env.dof_vel.clamp(-100, 100) * 0.05)
                torch.testing.assert_close(
                    env.current_obs[:, 9:25][:, env.wheel_indices],
                    torch.zeros(env.num_envs, 4, device=env.device),
                )
                torch.testing.assert_close(
                    env.wheel_contact_history[alive, 0],
                    env.contact_forces[alive][:, env.wheel_body_indices],
                )
        assert env.dof_vel[:, env.wheel_indices].abs().max() > 1e-4
        before_reward = env.dof_vel.clone()
        env._reward_dof_vel()
        torch.testing.assert_close(env.dof_vel, before_reward)
        # Regression: rollout creates tensors under inference_mode; resetting
        # later from runner.load/play must still work outside that context.
        env.reset_idx(torch.tensor([0], device=env.device))
        env.compute_observations()
        torch.testing.assert_close(env.observation_history.buffer[0], env.current_obs[0].expand(4, -1))
        torch.testing.assert_close(env.current_obs[0, 3:6], env.projected_gravity[0])
        assert torch.count_nonzero(env.wheel_contact_history[0]) == 0
        env.max_episode_length = 0
        result = env.step(torch.zeros_like(action))
        assert result[5].numel() == env.num_envs
        assert result[6].shape == (env.num_envs, 295)
        assert result[3].all()
        torch.testing.assert_close(
            env.observation_history.buffer,
            env.current_obs[:, None, :].expand(-1, 4, -1),
        )
        print("CTS physics smoke passed: {} environments, {} steps, {}, "
              "dimensions/history/real wheel rates/physics acceleration/reset/terminals".format(
                  env.num_envs, args.smoke_steps + 2, args.smoke_terrain))
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    import isaacgym  # noqa: F401
    import legged_gym.envs  # noqa: F401 - initialize tasks before utils
    from legged_gym.utils import get_args
    args = get_args(extra_custom_parameters=[
        {"name": "--smoke_terrain", "type": str, "default": "trimesh",
         "choices": ["plane", "trimesh", "heightfield"]},
        {"name": "--smoke_steps", "type": int, "default": 8},
    ])
    run_env_smoke(args)
