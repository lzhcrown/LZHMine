#!/usr/bin/env python
"""Summarize a fixed iteration window without loading checkpoints or CUDA."""

import argparse
import json
import math
from pathlib import Path
import statistics


def _quantile(values, fraction):
    """Linear-interpolated quantile, including a single-observation window."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values, prefix):
    return {
        prefix + "_mean": statistics.mean(values) if values else None,
        prefix + "_p95": _quantile(values, 0.95) if values else None,
        prefix + "_p99": _quantile(values, 0.99) if values else None,
        prefix + "_max": max(values) if values else None,
        prefix + "_gt_0_05_count": sum(value > 0.05 for value in values),
        prefix + "_gt_0_1_count": sum(value > 0.1 for value in values),
    }


def _validate_phase_metrics(metrics, prefix):
    required = [prefix + "_kl", prefix + "_kl_sample_max"]
    for group in ("reset", "non_reset"):
        count_key = prefix + "_" + group + "_count"
        if count_key not in metrics:
            raise ValueError("Incomplete B diagnostics: missing " + count_key)
        count = metrics[count_key]
        if count < 0 or int(count) != count:
            raise ValueError("Invalid sample count: " + count_key)
        if count != metrics["Diagnostics/pre_update_" + group + "_count"]:
            raise ValueError("B phase sample count differs from rollout: " + count_key)
        if count > 0:
            required.append(prefix + "_kl_" + group)
    for name in required:
        if name not in metrics:
            raise ValueError("Incomplete B diagnostics: missing " + name)


def summarize(path, start=1, end=None):
    if start < 1 or (end is not None and end < start):
        raise ValueError("Invalid iteration window")
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
    phase_b_rows = []
    for row in rows:
        if not row["minibatches"]:
            raise ValueError(f"No mini-batch diagnostics at iteration {row['iteration']}")
        values = list(row["metrics"].values())
        values += [value for batch in row["minibatches"] for value in batch.values()]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError("Nonfinite diagnostics")
        metrics = row["metrics"]
        phase_b = any(name.startswith(("Diagnostics/post_ppo_", "Diagnostics/estimator_phase_"))
                      for name in metrics)
        phase_b_rows.append(phase_b)
        update_keys = ("Diagnostics/update_ppo_count", "Diagnostics/update_estimator_count")
        if phase_b or any(name in metrics for name in update_keys):
            for name in update_keys:
                if name not in metrics:
                    raise ValueError("Incomplete update-count diagnostics: missing " + name)
                if metrics[name] != len(row["minibatches"]):
                    raise ValueError(f"{name} differs from mini-batch count at iteration {row['iteration']}")
        if phase_b:
            for prefix in ("Diagnostics/post_ppo", "Diagnostics/estimator_phase"):
                _validate_phase_metrics(metrics, prefix)
    if any(phase_b_rows) and not all(phase_b_rows):
        raise ValueError("Mixed A/B diagnostics in the requested window")
    batches = [batch for row in rows for batch in row["minibatches"]]

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
        "pre_update_kl_max": max(metric("Diagnostics/pre_update_kl")),
        "pre_update_value_error_max": max(metric("Diagnostics/pre_update_value_max_abs_error")),
        "latent_cos_median_across_iterations": statistics.median(metric("Latent/latent_cos_to_mean_median")),
    }
    report.update(_distribution([batch["kl"] for batch in batches], "kl_minibatch"))
    report.update(_distribution(metric("Diagnostics/post_update_kl"), "post_update_kl"))
    if all(phase_b_rows):
        for phase in ("post_ppo", "estimator_phase"):
            # These are separately measured distributions, not the difference
            # between rollout-relative KL values before and after estimation.
            report.update(_distribution(metric("Diagnostics/" + phase + "_kl"), phase + "_kl"))
            report[phase + "_kl_sample_max"] = max(metric("Diagnostics/" + phase + "_kl_sample_max"))
            for group in ("reset", "non_reset"):
                count_key = "Diagnostics/" + phase + "_" + group + "_count"
                kl_key = "Diagnostics/" + phase + "_kl_" + group
                report[phase + "_" + group + "_samples"] = sum(metric(count_key))
                values = [row["metrics"][kl_key] for row in rows if row["metrics"][count_key] > 0]
                report.update(_distribution(values, phase + "_kl_" + group))
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
