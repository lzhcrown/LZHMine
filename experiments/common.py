"""Thin experiment entry points with lazy imports for separate runtimes."""

from experiments.registry import export_path, get_experiment
from legged_gym.scripts._bootstrap import configure_runtime


def train_main(key):
    experiment = get_experiment(key)
    configure_runtime(isaac_gym=True)
    # The training module imports Isaac Gym before PyTorch.
    from legged_gym.scripts.train_nezha_moe_cts import train, training_args

    return train(training_args(), task_name=experiment.TASK_NAME)


def play_main(key):
    experiment = get_experiment(key)
    configure_runtime(isaac_gym=True)
    from legged_gym.scripts.play_nezha_moe_cts import PLAY_ARGUMENTS, get_args, play

    args = get_args(extra_custom_parameters=PLAY_ARGUMENTS)
    return play(
        args,
        task_name=experiment.TASK_NAME,
        export_path=export_path(key),
        expected_actor_type=experiment.POLICY["actor_type"],
        expected_actor_expert_num=experiment.POLICY["actor_expert_num"],
    )


def export_main(key):
    experiment = get_experiment(key)
    configure_runtime()
    from legged_gym.scripts.export_nezha_moe_cts import export, parse_args

    args = parse_args(
        default_experiment=experiment.RUNNER["experiment_name"],
        default_output=export_path(key),
    )
    return export(
        args,
        expected_actor_type=experiment.POLICY["actor_type"],
        expected_actor_expert_num=experiment.POLICY["actor_expert_num"],
    )


def simulate_main(key):
    configure_runtime()
    from deploy.deploy_mujoco.deploy_mujoco import parse_args, run

    args = parse_args()
    if args.policy is None:
        args.policy = export_path(key)
    return run(args)
