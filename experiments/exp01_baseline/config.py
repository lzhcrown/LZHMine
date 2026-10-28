"""Baseline overrides; shared observations, environment and PPO stay unchanged."""

KEY = "exp01_baseline"
TASK_NAME = "nezha_moe_cts"
POLICY = {"actor_type": "mlp", "actor_expert_num": 0}
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/exp01_baseline",
    "run_name": "baseline",
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "moe-cts", "baseline", "isaac-gym"],
    "wandb_notes": "Experiment 1: CTS student MoE, original MLP actor and critic",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/exp01_baseline/policy.pt"
