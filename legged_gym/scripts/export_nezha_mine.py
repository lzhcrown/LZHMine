#!/usr/bin/env python
"""Export a Nezha MINE checkpoint without starting Isaac Gym or rendering."""

import argparse
import os
import re

import torch

from rsl_rl.modules import MINEActorCritic, MINEPolicyExporter


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


def _build_policy(history_order):
    return MINEActorCritic(
        num_actor_obs=130,
        num_critic_obs=1125,
        num_one_step_obs=65,
        num_one_step_privileged_obs=375,
        estimation_target_indices=[371, 372, 373, 374],
        num_actions=16,
        history_order=history_order,
        is_privileged_obs=False,
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
    ).eval()


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
    policy = _build_policy(args.history_order)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    MINEPolicyExporter(policy).export(output_path)

    scripted = torch.jit.load(output_path, map_location="cpu").eval()
    with torch.inference_mode():
        probe = torch.zeros(1, 130)
        actions = scripted(probe)
        gate = scripted.get_mode_probabilities(probe)
    if actions.shape != (1, 16) or gate.shape != (1, 3):
        raise RuntimeError("Exported policy failed its shape validation")
    print(f"checkpoint:    {checkpoint_path}")
    print(f"history order:{args.history_order}")
    print(f"policy:        {output_path}")
    print("validated:     input [1,130], action [1,16], gate [1,3]")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Run directory name under logs/nezha3_mine")
    parser.add_argument("--checkpoint", help="Explicit model_<iteration>.pt path")
    parser.add_argument("--output", help="Explicit policy.pt output path")
    parser.add_argument(
        "--history-order",
        choices=("oldest_first", "newest_first"),
        default="oldest_first",
        help="Use newest_first only for pre-alignment LZHMine checkpoints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
