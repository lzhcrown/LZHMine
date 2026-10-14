#!/usr/bin/env python
"""Export a Nezha MINE checkpoint without starting Isaac Gym or rendering."""

import argparse
import os
import re

import torch

from rsl_rl.modules import MINEActorCritic, MINEPolicyExporter
from rsl_rl.utils.mine_checkpoint import policy_config_from_checkpoint


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
EXPERIMENT = "nezha3_mine"


def _latest_checkpoint(run_dir):
    candidates = []
    for filename in os.listdir(run_dir):
        match = re.fullmatch(r"model_(\d+)\.pt", filename)
        if match:
            candidates.append((int(match.group(1)), os.path.join(run_dir, filename)))
    if not candidates:
        raise ValueError(f"No model_<iteration>.pt checkpoints in {run_dir}")
    return max(candidates)[1]


def _build_policy(history_order, is_privileged_obs=True, use_mode_gate=True, **overrides):
    config = dict(
        num_actor_obs=130,
        num_critic_obs=1125,
        num_one_step_obs=65,
        num_one_step_privileged_obs=375,
        estimation_target_indices=[371, 372, 373, 374],
        num_actions=16,
        history_order=history_order,
        is_privileged_obs=is_privileged_obs,
        use_mode_gate=use_mode_gate,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        enc_hidden_dims=[256, 128, 64],
        tar_hidden_dims=[256, 128, 64],
        latent_dim=16,
        num_modes=3,
        gate_hidden_dim=64,
        num_prototypes=32,
        temperature=3.0,
        estimator_learning_rate=1e-3,
        estimator_max_grad_norm=10.0,
        mode_loss_coef=0.5,
        mode_semantic_cfg={"enabled": False},
    )
    config.update(overrides)
    return MINEActorCritic(**config).eval()


def export(args):
    if args.checkpoint:
        checkpoint_path = os.path.abspath(args.checkpoint)
        run_name = os.path.basename(os.path.dirname(checkpoint_path))
    else:
        if not args.run:
            raise ValueError("Pass --run <training-run-name> or --checkpoint <path>")
        run_dir = os.path.join(REPO_ROOT, "logs", EXPERIMENT, args.run)
        checkpoint_path = _latest_checkpoint(run_dir)
        run_name = args.run
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    output_path = args.output or os.path.join(
        REPO_ROOT, "logs", EXPERIMENT, "exported", run_name, "policy.pt"
    )
    output_path = os.path.abspath(output_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state_dict = checkpoint["model_state_dict"]
    target_width = state_dict["estimator.target.0.weight"].shape[1]
    config = policy_config_from_checkpoint(checkpoint)
    if args.history_order is not None:
        config["history_order"] = args.history_order
    policy = _build_policy(**config)
    policy.load_state_dict(state_dict, strict=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    MINEPolicyExporter(policy).export(output_path)

    scripted = torch.jit.load(output_path, map_location="cpu").eval()
    with torch.inference_mode():
        probe = torch.zeros(1, policy.num_actor_obs)
        actions = scripted(probe)
        gate = scripted.get_mode_probabilities(probe)
    num_modes = policy.estimator.num_modes if policy.use_mode_gate else 0
    if actions.shape != (1, policy.num_actions) or gate.shape != (1, num_modes):
        raise RuntimeError("Exported policy failed its shape validation")
    print(f"checkpoint:    {checkpoint_path}")
    print(f"target width:  {target_width}")
    print(f"mode gate:     {'on' if policy.use_mode_gate else 'off'}")
    print(f"history order: {policy.history_order}")
    print(f"policy:        {output_path}")
    print(f"validated:     input [1,{policy.num_actor_obs}], "
          f"action [1,{policy.num_actions}], gate [1,{num_modes}]")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Run directory name under logs/nezha3_mine")
    parser.add_argument("--checkpoint", help="Explicit model_<iteration>.pt path")
    parser.add_argument("--output", help="Explicit policy.pt output path")
    parser.add_argument(
        "--history-order",
        choices=("oldest_first", "newest_first"),
        default=None,
        help="Override checkpoint metadata; archived runs default to oldest_first.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
