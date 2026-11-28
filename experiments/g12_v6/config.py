"""Train a proprioceptive student with a task-conditioned teacher and two critics."""

from experiments.common import configure_teacher_skill_observation
from experiments.g12_v5.config import POLICY as V5_POLICY, configure_environment as configure_v5
from experiments.registry import PLATFORM_V6_EXPERIMENT


KEY = PLATFORM_V6_EXPERIMENT
TASK_NAME = "nezha_cts_" + KEY
POLICY = dict(
    V5_POLICY,
    observation_dims=[3, 3, 3, 16, 16, 16],
    num_critic_obs=296,
    critic_type="task_mlp",
    critic_expert_num=0,
    critic_task_count=2,
)
RUNNER = {
    "experiment_key": KEY,
    "experiment_name": "nezha_moe_cts/" + KEY,
    "run_name": KEY,
    "wandb_group": KEY,
    "wandb_tags": ["nezha", "cts", KEY],
    "wandb_notes": "G12 v5 rewards and curriculum, privileged task ID, independent terrain critics and a proprioceptive student.",
}
EXPORT_PATH = "deploy/deploy_mujoco/nezha/policy/" + KEY + "/policy.pt"


def configure_environment(cfg):
    configure_v5(cfg)
    configure_teacher_skill_observation(cfg)
