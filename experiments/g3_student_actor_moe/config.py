"""Architecture-only overrides; all other CTS settings are shared."""

KEY = "g3_student_actor_moe"
TASK_NAME = "nezha_cts_g3_student_actor_moe"
POLICY = {
    "student_encoder_type": "moe",
    "expert_num": 8,
    "actor_type": "moe",
    "actor_expert_num": 8,
    "critic_type": "mlp",
    "critic_expert_num": 0,
}
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/g3_student_actor_moe",
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", "g3_student_actor_moe", "isaac-gym"],
    "wandb_notes": "CTS ablation g3_student_actor_moe: student=moe, actor=moe, critic=mlp",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/g3_student_actor_moe/policy.pt"

