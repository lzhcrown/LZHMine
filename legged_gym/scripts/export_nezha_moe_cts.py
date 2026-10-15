#!/usr/bin/env python
"""Export a WF-CTS student without importing Isaac Gym or requiring a GPU."""

import argparse
import os
import re

import torch

from rsl_rl.utils.cts_checkpoint import load_cts_checkpoint, policy_from_checkpoint
from rsl_rl.utils.cts_export import CTSStudentExporter


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
EXPERIMENT = "nezha_moe_cts"


def resolve_checkpoint(checkpoint=None, run=None, experiment=EXPERIMENT):
    if checkpoint:
        path = os.path.abspath(checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    root = os.path.join(REPO_ROOT, "logs", experiment)
    if run not in (None, "-1", "latest"):
        run_dirs = [run if os.path.isabs(run) else os.path.join(root, run)]
    else:
        if not os.path.isdir(root):
            raise FileNotFoundError("No CTS runs in {}".format(root))
        run_dirs = [os.path.join(root, name) for name in os.listdir(root)
                    if os.path.isdir(os.path.join(root, name)) and name not in ("exported", "play")]
        run_dirs.sort(key=os.path.getmtime, reverse=True)
    for directory in run_dirs:
        candidates = []
        for filename in os.listdir(directory):
            match = re.fullmatch(r"model_(\d+)\.pt", filename)
            if match:
                candidates.append((int(match.group(1)), os.path.join(directory, filename)))
        if candidates:
            return max(candidates)[1]
    raise ValueError("No model_<iteration>.pt checkpoints in {}".format(root))


def export(args):
    path = resolve_checkpoint(args.checkpoint, args.run, getattr(args, "experiment", EXPERIMENT))
    checkpoint = load_cts_checkpoint(path, map_location="cpu")
    policy = policy_from_checkpoint(checkpoint, device="cpu").eval()
    run_name = os.path.basename(os.path.dirname(path))
    output = os.path.abspath(args.output or os.path.join(
        REPO_ROOT, "logs", getattr(args, "experiment", EXPERIMENT), "exported", run_name, "policy.pt"
    ))
    CTSStudentExporter(policy, checkpoint["observation_config"], checkpoint.get("control_config")).export(output)
    scripted = torch.jit.load(output, map_location="cpu").eval()
    with torch.inference_mode():
        probe = torch.randn(3, policy.num_actor_obs)
        expected = policy.act_inference(probe)
        actual = scripted(probe)
        probabilities = scripted.get_expert_probabilities(probe)
    if actual.shape != (3, policy.num_actions) or not torch.allclose(actual, expected, atol=1e-6, rtol=1e-5):
        raise RuntimeError("Exported student differs from the checkpoint policy")
    if probabilities.shape != (3, scripted.get_num_experts()):
        raise RuntimeError("Exported expert probabilities have invalid dimensions")
    print("checkpoint: {}".format(path))
    print("student:    {}".format(output))
    print("validated:  history [N,{}], actions [N,{}], experts [N,{}]".format(
        policy.num_actor_obs, policy.num_actions, scripted.get_num_experts()
    ))
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Explicit model_<iteration>.pt path")
    parser.add_argument("--run", help="Run name, absolute run directory, or latest (default)")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--output", help="Exported policy.pt path")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
