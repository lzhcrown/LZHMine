"""WF MoE-CTS observations and training, with Nezha's Isaac Gym dynamics."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.nezha.nezha_robot_config import NezhaRobotCfg


class NezhaMoECTSCfg(NezhaRobotCfg):
    class env(NezhaRobotCfg.env):
        frame_stack = 4
        critic_frame_stack = 1
        history_order = "oldest_first"
        history_layout = "term_major"
        num_one_step_observations = 57
        num_observations = 228
        num_one_step_privileged_obs = 295
        num_privileged_obs = 295
        num_height_observations = 187
        # Set a schema dict to customize fields/scales; loaded checkpoints
        # restore their schema here before the simulator is constructed.
        observation_config = None

    class normalization(NezhaRobotCfg.normalization):
        # Observation terms implement WF clipping BEFORE scaling. A final
        # global clip would incorrectly reduce scaled linear velocities.
        clip_observations = float("inf")

        class obs_scales(NezhaRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 1e-4
            torques = 0.01
            force_measurements = 0.001
            height_measurements = 2.5


class NezhaMoECTSCfgPPO(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = "MoECTSOnPolicyRunner"

    class policy:
        init_noise_std = 1.0
        # Shared defaults preserve the original baseline. Experiment-specific
        # overrides live in experiments/exp01_baseline and exp02_actor_moe.
        actor_type = "mlp"
        actor_expert_num = 0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        teacher_encoder_hidden_dims = [512, 256]
        student_encoder_hidden_dims = [512, 256, 256]
        latent_dim = 32
        expert_num = 8
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3
        student_encoder_learning_rate = 1e-3
        load_balance_coef = 0.01
        teacher_env_ratio = 0.75
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCriticMoECTS"
        algorithm_class_name = "MoECTS"
        num_steps_per_env = 24
        max_iterations = 100000
        save_interval = 1000
        experiment_name = "nezha_moe_cts"
        run_name = "wf_terrain"
        # TensorBoard always records; W&B is an optional CLI opt-in.
        wandb_enabled = False
        wandb_project = "LZHMine"
        wandb_entity = None
        wandb_group = "nezha_moe_cts"
        wandb_tags = ["nezha", "moe-cts", "isaac-gym"]
        wandb_mode = "online"
        wandb_notes = "WF MoE concurrent teacher-student training on Nezha"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
