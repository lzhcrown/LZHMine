"""Robot, terrain, control and reward configuration for Nezha."""

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.terrains.wf import default_wf_sub_terrains
from rsl_rl.utils.wf_curriculum import default_wf_command_config


class NezhaRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 16
        episode_length_s = 25
        fail_to_terminal_time_s = 0.6

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        generator_profile = "legbot_wf"
        terrain_length = 9.0
        terrain_width = 9.0
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25.0
        sub_terrain_border_width = 0.5
        difficulty_range = [0.0, 1.0]
        seed = None
        num_rows = 10
        num_cols = 20
        max_init_terrain_level = 5
        # legbot_lab WF-CTS-MOE daff9fb: TERRAIN_CFG, not DEFAULT_TERRAIN_CFG.
        wf_sub_terrains = default_wf_sub_terrains()
        terrain_proportions = [spec["proportion"] for spec in wf_sub_terrains.values()]
        curriculum = True
        measure_heights = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        yaw_init_range = 3.14
        spawn_xy_range = 0.5
        spawn_z_range = [0.0, 0.05]

    class commands(LeggedRobotCfg.commands):
        curriculum_profile = "legbot_wf"
        wf_settings = default_wf_command_config()
        # Keep WF sampling; restore Nezha's reward-gated range progression.
        wf_settings["range_curriculum"] = "nezha_reward"
        curriculum = True
        resampling_time = 5.0
        heading_command = False
        max_curriculum = 1.5
        max_curriculum_yaw = 1.5

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.60]
        rot = [0.0, 0.0, 0.0, 1.0]
        rand_init_dof = False
        rand_init_dof_range = 0.0
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.6,
            "FL_foot_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.6,
            "FR_foot_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.8,
            "RL_calf_joint": -1.6,
            "RL_foot_joint": 0.0,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.6,
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
            + "/resources/robots/urdf2.1/urdf/nezha3_description.urdf"
        )
        name = "nezha3"
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
        # The archived baseline spells the trunk token as "thunk", which
        # matches no body. Its effective termination set therefore contains
        # the four hip bodies only.
        terminate_after_contacts_on = ["hip"]
        self_collisions = 1
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        rand_interval_s = 4
        randomize_payload_mass = True
        payload_mass_range = [-5.0, 5.0]
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_restitution = False
        restitution_range = [0.0, 1.0]
        randomize_rigids_after_start = False
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
        max_push_ang_vel = 0.5
        delay = False
        action_noise = 0.0
        action_delay = 0.0

        randomize_inertia = False
        randomize_inertia_range = [0.8, 1.2]
        randomize_coulomb_friction = False
        joint_stick_friction_range = [0.1, 0.2]
        joint_coulomb_friction_range = [0.0, 0.0]

        randomize_joint_friction = True
        randomize_joint_friction_each_joint = False
        default_joint_friction = [
            0.025, 0.05, 0.05, 0.05,
            0.025, 0.05, 0.05, 0.05,
            0.025, 0.05, 0.05, 0.05,
            0.025, 0.05, 0.05, 0.05,
        ]
        joint_friction_range = [0.95, 1.05]
        randomize_joint_damping = False
        default_joint_damping = [0.0] * 16
        randomize_joint_armature = False
        default_joint_armature = [0.0] * 16

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 1.0
        base_height_target = 0.50
        max_contact_force = 400.0
        clip_single_reward = 10.0

        class scales:
            termination = -0.8
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            # WF curriculum: -2 -> -1 over iterations 0..1500.
            lin_vel_z = -2.0
            ang_vel_xy = -0.2
            orientation = -2.0
            # WF curriculum: -1 -> -10 over iterations 0..5000.
            base_height = -1.0
            feet_contact_uniform = -0.1
            hip_default = -8.0
            stand_still = -3.0
            collision = -0.1
            feet_stumble = -0.5
            action_rate = -0.01
            torques = -2.5e-6
            dof_vel = -2.0e-3
            dof_acc = -2.5e-7
            # Apr10 gated-three-class training snapshot uses 1.0.  The
            # wheel_gym_CQ working tree was later changed to 4.0.
            feet_air_time = 1.0
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

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.1
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2
