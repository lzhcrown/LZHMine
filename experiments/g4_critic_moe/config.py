"""Architecture-only overrides; all other CTS settings are shared."""

KEY = "g4_critic_moe"
TASK_NAME = "nezha_cts_g4_critic_moe"
POLICY = {
    "student_encoder_type": "mlp",
    "expert_num": 0,
    "actor_type": "mlp",
    "actor_expert_num": 0,
    "critic_type": "moe",
    "critic_expert_num": 8,
}
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/g4_critic_moe",
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", "g4_critic_moe", "isaac-gym"],
    "wandb_notes": "CTS ablation g4_critic_moe: student=mlp, actor=mlp, critic=moe",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/g4_critic_moe/policy.pt"

