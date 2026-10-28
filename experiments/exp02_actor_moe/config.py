"""Actor-only MoE overrides; all other training settings use the baseline."""

KEY = "exp02_actor_moe"
TASK_NAME = "nezha_moe_cts_actor_moe"
POLICY = {"actor_type": "moe", "actor_expert_num": 8}
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/exp02_actor_moe",
    "run_name": "actor_moe",
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "moe-cts", "actor-moe", "isaac-gym"],
    "wandb_notes": "Experiment 2: CTS student MoE, eight-expert MoE actor, MLP critic",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/exp02_actor_moe/policy.pt"
