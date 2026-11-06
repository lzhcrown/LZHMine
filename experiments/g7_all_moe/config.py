"""Architecture-only overrides; all other CTS settings are shared."""

KEY = "g7_all_moe"
TASK_NAME = "nezha_cts_g7_all_moe"
POLICY = {
    "student_encoder_type": "moe",
    "expert_num": 8,
    "actor_type": "moe",
    "actor_expert_num": 8,
    "critic_type": "moe",
    "critic_expert_num": 8,
}
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/g7_all_moe",
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", "g7_all_moe", "isaac-gym"],
    "wandb_notes": "CTS ablation g7_all_moe: student=moe, actor=moe, critic=moe",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/g7_all_moe/policy.pt"

