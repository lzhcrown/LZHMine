"""Architecture-only overrides; all other CTS settings are shared."""

KEY = "g1_student_moe"
TASK_NAME = "nezha_cts_g1_student_moe"
POLICY = {
    "student_encoder_type": "moe",
    "expert_num": 8,
    "actor_type": "mlp",
    "actor_expert_num": 0,
    "critic_type": "mlp",
    "critic_expert_num": 0,
}
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/g1_student_moe",
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", "g1_student_moe", "isaac-gym"],
    "wandb_notes": "CTS ablation g1_student_moe: student=moe, actor=mlp, critic=mlp",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/g1_student_moe/policy.pt"

