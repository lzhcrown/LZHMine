#!/usr/bin/env python
"""Summarize a fixed iteration window without loading checkpoints or CUDA."""

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(path, start=1, end=None):
    rows = []
    previous = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            iteration = row["iteration"]
            if iteration <= previous:
                raise ValueError("Duplicate/out-of-order iterations; use a single run's log")
            previous = iteration
            if iteration >= start and (end is None or iteration <= end):
                rows.append(row)
    if not rows:
        raise ValueError("No iterations in the requested window")
    if rows[0]["iteration"] != start or (end is not None and rows[-1]["iteration"] != end):
        raise ValueError("Requested window is not complete")
    if len(rows) != rows[-1]["iteration"] - start + 1:
        raise ValueError("Missing iterations in the requested window")
    batches = [batch for row in rows for batch in row["minibatches"]]
    if not batches:
        raise ValueError("No mini-batch diagnostics")
    for row in rows:
        values = list(row["metrics"].values())
        values += [value for batch in row["minibatches"] for value in batch.values()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite diagnostics")

    def metric(name):
        return [row["metrics"][name] for row in rows]

    report = {
        "start": start,
        "end": rows[-1]["iteration"],
        "iterations": len(rows),
        "minibatches": len(batches),
        "lr_floor_iteration_fraction": statistics.mean(
            value <= 1e-5 * (1 + 1e-6) for value in metric("Loss/learning_rate")
        ),
        "lr_floor_minibatch_fraction": statistics.mean(batch["lr_at_floor"] for batch in batches),
        "kl_minibatch_mean": statistics.mean(batch["kl"] for batch in batches),
        "kl_minibatch_max": max(batch["kl"] for batch in batches),
        "pre_update_kl_max": max(metric("Diagnostics/pre_update_kl")),
        "pre_update_value_error_max": max(metric("Diagnostics/pre_update_value_max_abs_error")),
        "post_update_kl_mean": statistics.mean(metric("Diagnostics/post_update_kl")),
        "latent_cos_median_across_iterations": statistics.median(metric("Latent/latent_cos_to_mean_median")),
    }
    for group in ("reset", "non_reset"):
        count_key = "Diagnostics/pre_update_" + group + "_count"
        kl_key = "Diagnostics/pre_update_kl_" + group
        report[group + "_samples"] = sum(row["metrics"][count_key] for row in rows)
        valid = [row["metrics"][kl_key] for row in rows if row["metrics"][count_key] > 0]
        report["pre_update_kl_" + group + "_max"] = max(valid) if valid else None
    report["integrity_pass"] = (
        report["reset_samples"] > 0 and report["non_reset_samples"] > 0
        and report["pre_update_kl_max"] < 1e-3
        and report["pre_update_kl_reset_max"] < 1e-3
        and report["pre_update_kl_non_reset_max"] < 1e-3
        and report["pre_update_value_error_max"] < 1e-4
    )
    # This is only the LR target, not an automatic stage-1 pass decision.
    report["lr_target_pass"] = report["lr_floor_iteration_fraction"] < 0.10
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory or training_diagnostics.jsonl")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--check-integrity", action="store_true")
    args = parser.parse_args()
    path = args.run / "training_diagnostics.jsonl" if args.run.is_dir() else args.run
    try:
        report = summarize(path, args.start, args.end)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.check_integrity and not report["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
