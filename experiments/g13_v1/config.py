"""Train a student and privileged teacher with frozen pretrained terrain features."""

from experiments.g13_v2.config import POLICY as V2_POLICY, configure_environment
from experiments.registry import SKILL_V1_EXPERIMENT


KEY = SKILL_V1_EXPERIMENT
TASK_NAME = "nezha_cts_" + KEY
POLICY = dict(V2_POLICY, teacher_map_attention=False, teacher_frozen_terrain=True)
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/" + KEY,
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", KEY],
    "wandb_notes": "Student and teacher use frozen pretrained terrain features. Teacher also receives privileged state and skill ID.",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"
