"""Train a geometry-conditioned student with a jointly trained map-attention teacher."""

from experiments.common import configure_teacher_skill_observation
from experiments.g12_v5.config import POLICY as V5_POLICY, configure_environment as configure_v5
from experiments.registry import SKILL_V2_EXPERIMENT


KEY = SKILL_V2_EXPERIMENT
TASK_NAME = "nezha_cts_" + KEY
POLICY = dict(V5_POLICY, observation_dims=[3, 3, 3, 16, 16, 16],
              student_frozen_terrain=True, teacher_map_attention=True,
              critic_type="task_mlp", critic_expert_num=0, critic_task_count=2)
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/" + KEY,
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", KEY],
    "wandb_notes": "Student uses pretrained terrain features. Teacher learns map attention with privileged state and skill ID.",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"


def configure_environment(cfg):
    configure_v5(cfg)
    configure_teacher_skill_observation(cfg)
