"""Training configuration for the Nezha gated-modal policy."""

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class NezhaMINECfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 16
        frame_stack = 2
        critic_frame_stack = 3
        num_one_step_observations = 65
        num_observations = frame_stack * num_one_step_observations
        num_height_observations = 187
        # Complete privileged frame, aligned with wheel_gym_CQ:
        # actor(65) + dof_acc(16) + torques(16) + base_mass(1)
        # + base_com(3) + kp(16) + kd(16) + all-body contact forces(51)
        # + terrain heights(187) + velocity/height estimation targets(4).
        num_one_step_privileged_obs = 375
        num_privileged_obs = critic_frame_stack * num_one_step_privileged_obs
        estimation_target_indices = [371, 372, 373, 374]
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        terrain_length = 10.0
        terrain_width = 10.0
        num_rows = 10
        num_cols = 20
        max_init_terrain_level = 5
        # LZHMine's terrain generator has eight valid branches.  Keep all
        # eight probabilities so discrete obstacles are actually generated;
        # the five-entry legacy wheel_gym_CQ list would index out of range.
        terrain_proportions = [0.1, 0.1, 0.15, 0.15, 0.2, 0.1, 0.1, 0.1]
        curriculum = True
        measure_heights = True
        restitution = 0.5

    class commands(LeggedRobotCfg.commands):
        heading_command = False
        max_curriculum = 1.5

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.60]
        rot = [0.0, 0.0, 0.0, 1.0]
        default_joint_angles = {
            "FL_hip_joint": -0.10,
            "FL_thigh_joint": 0.925,
            "FL_calf_joint": -1.85,
            "FL_foot_joint": 0.0,
            "FR_hip_joint": 0.10,
            "FR_thigh_joint": 0.925,
            "FR_calf_joint": -1.85,
            "FR_foot_joint": 0.0,
            "RL_hip_joint": -0.10,
            "RL_thigh_joint": 0.925,
            "RL_calf_joint": -1.85,
            "RL_foot_joint": 0.0,
            "RR_hip_joint": 0.10,
            "RR_thigh_joint": 0.925,
            "RR_calf_joint": -1.85,
            "RR_foot_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "hip_joint": 150.0,
            "thigh_joint": 150.0,
            "calf_joint": 300.0,
            "foot_joint": 0.0,
        }
        damping = {
            "hip_joint": 4.0,
            "thigh_joint": 4.0,
            "calf_joint": 4.0,
            "foot_joint": 1.2,
        }
        action_scale = 0.25
        vel_scale = 10.0
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = (
            LEGGED_GYM_ROOT_DIR
            + "/resources/robots/nezha2_description1126/urdf/nezha2_description.urdf"
        )
        name = "nezha"
        foot_name = "foot"
        wheel_name = ["foot_joint"]
        feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        wheel_dof_names = [
            "FL_foot_joint",
            "FR_foot_joint",
            "RL_foot_joint",
            "RR_foot_joint",
        ]
        penalize_contacts_on = ["hip", "thigh", "calf"]
        terminate_after_contacts_on = ["trunk", "hip"]
        self_collisions = 1
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [-5.0, 5.0]
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_restitution = False
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_kp = False
        kp_range = [0.9, 1.1]
        randomize_kd = False
        kd_range = [0.9, 1.1]
        randomize_initial_joint_pos = False
        initial_joint_pos_range = [0.9, 1.1]
        disturbance = False
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 400  # policy steps, approximately 8 s at 50 Hz
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.5
        delay = False

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        base_height_target = 0.50
        max_contact_force = 400.0

        class scales:
            termination = -0.8
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            lin_vel_z = -4.0
            ang_vel_xy = -0.2
            orientation = -2.0
            base_height = -1.0
            feet_air_time = 4.0
            feet_contact_uniform = -0.1
            hip_default = -8.0
            stand_still = -3.0
            collision = -0.1
            feet_stumble = -0.5
            action_rate = -0.01
            torques = -2.5e-6
            dof_vel = -2.0e-3
            dof_acc = -2.5e-7
            dof_pos_limits = -0.9
            run_still = -0.1
            smoothness = -0.05
            joint_power = -2.0e-4
            feet_contact_forces = -1.0e-3
            wheel_contact_vel_y = -1.0

    class normalization(LeggedRobotCfg.normalization):
        contact_force_range = [0.0, 200.0]

        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 1.0
            height_measurements = 5.0
            torques = 1.0
            force_measurements = 0.01

    class viewer(LeggedRobotCfg.viewer):
        pos = [3.0, -3.0, 2.0]
        lookat = [0.0, 0.0, 0.5]


class NezhaMINECfgPPO(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = "MINEOnPolicyRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        enc_hidden_dims = [256, 128, 64]
        tar_hidden_dims = [256, 128, 64]
        latent_dim = 16
        num_modes = 3
        gate_hidden_dim = 64
        num_prototypes = 32
        temperature = 3.0
        estimator_learning_rate = 1e-3
        estimator_max_grad_norm = 10.0
        mode_loss_coef = 0.5
        # Fixed semantic indices: 0=wheel, 1=leg, 2=wheel-leg hybrid.
        # Static/uncertain samples are excluded from semantic supervision.
        mode_semantic_cfg = {
            "enabled": True,
            "loss_coef": 0.10,
            "mode_loss_coef": 0.50,
            "command_slice": [6, 9],
            "dof_vel_slice": [21, 37],
            "action_slice": [49, 65],
            "wheel_dof_indices": [3, 7, 11, 15],
            "leg_dof_indices": [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
            "wheel_activity_low": 0.05,
            "wheel_activity_high": 0.20,
            "leg_activity_low": 0.03,
            "leg_activity_high": 0.12,
            "static_command_threshold": 0.05,
            "leg_action_delta_weight": 0.70,
            "leg_velocity_weight": 0.30,
        }
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "MINEActorCritic"
        algorithm_class_name = "MINEPPO"
        num_steps_per_env = 24
        max_iterations = 100000
        save_interval = 200
        experiment_name = "nezha_mine"
        run_name = "gated_modal_dual_encoder"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
