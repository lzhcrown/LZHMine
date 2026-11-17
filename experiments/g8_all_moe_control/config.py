"""G7-derived residual/temporal factorial; all unrelated settings stay shared."""

KEY = "g8_all_moe_control"
TASK_NAME = "nezha_cts_" + KEY
POLICY = {
    "student_encoder_type": "moe",
    "expert_num": 8,
    "actor_type": "moe",
    "actor_expert_num": 8,
    "critic_type": "moe",
    "critic_expert_num": 8,
    "residual_enabled": False,
    "residual_hidden_dims": [128, 64],
    "residual_action_scales": None,
}
# Initial coefficient for the factorial; tune only in a separate sensitivity experiment.
ALGORITHM = {"temporal_loss_coef": 0.0}
RUNNER = {
    "experiment_key": KEY,
    "comparison_reference": "g7_all_moe",
    "comparison_factors": {"residual": False, "temporal": False},
    "experiment_name": "nezha_moe_cts/" + KEY,
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", "residual_factorial", KEY],
    "wandb_notes": ("G7 residual factorial: residual=False, temporal=False; "
                    "initialize all arms from the same checkpoint and use equal interaction budgets"),
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"
