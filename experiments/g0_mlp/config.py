"""Architecture-only overrides; all other CTS settings are shared."""

KEY = "g0_mlp"
TASK_NAME = "nezha_cts_g0_mlp"
POLICY = {
    "student_encoder_type": "mlp",
    "expert_num": 0,
    "actor_type": "mlp",
    "actor_expert_num": 0,
    "critic_type": "mlp",
    "critic_expert_num": 0,
}
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/g0_mlp",
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", "g0_mlp", "isaac-gym"],
    "wandb_notes": "CTS ablation g0_mlp: student=mlp, actor=mlp, critic=mlp",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/g0_mlp/policy.pt"

