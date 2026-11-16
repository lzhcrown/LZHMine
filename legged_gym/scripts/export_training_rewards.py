"""Export every recorded CTS mean reward from TensorBoard logs to CSV.

Run with the repository's Python environment; Isaac Gym is not required.
Iterations retain TensorBoard's zero-based step. Reward values are exported
without smoothing or downsampling. If a run logs a metric more than once at
the same step, the event with the latest wall time is used.
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader


REWARD_TAGS = {
    "Train/teacher_reward": "teacher_mean_reward",
    "Train/student_reward": "student_mean_reward",
}
VALUE_COLUMNS = list(REWARD_TAGS.values())
ROW_COLUMNS = ["iteration", *VALUE_COLUMNS, "wall_time_unix", "wall_time_utc"]
SUMMARY_COLUMNS = [
    "experiment", "run", "rows", "first_iteration", "last_iteration",
    "teacher_points", "student_points", "last_teacher_mean_reward",
    "last_student_mean_reward", "csv_path",
]


def read_rewards(event_paths):
    points = {}
    counts = {column: 0 for column in VALUE_COLUMNS}
    duplicates = 0
    for path in sorted(event_paths):
        for event in LegacyEventFileLoader(str(path)).Load():
            for value in event.summary.value:
                column = REWARD_TAGS.get(value.tag)
                if column is None:
                    continue
                if not value.HasField("simple_value"):
                    raise ValueError("Expected a scalar reward in {}: {}".format(path, value.tag))
                step_values = points.setdefault(event.step, {})
                previous = step_values.get(column)
                counts[column] += 1
                if previous is not None:
                    duplicates += 1
                if previous is None or event.wall_time >= previous[0]:
                    step_values[column] = (event.wall_time, value.simple_value)
    rows = []
    for step, values in sorted(points.items()):
        wall_time = max(item[0] for item in values.values())
        row = {
            "iteration": step,
            "wall_time_unix": wall_time,
            "wall_time_utc": datetime.fromtimestamp(wall_time, timezone.utc).isoformat(),
        }
        row.update({column: values[column][1] for column in values})
        rows.append(row)
    return rows, counts, duplicates


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export(log_dir, output_dir):
    runs = {}
    for path in sorted(log_dir.rglob("events.out.tfevents.*")):
        if path.is_file():
            runs.setdefault(path.parent, []).append(path)
    if not runs:
        raise ValueError("No TensorBoard event files found in {}".format(log_dir))
    if any(output_dir == run for run in runs):
        raise ValueError("Choose an output directory separate from the event directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    combined_path = output_dir / "mean_rewards_all.csv"
    with combined_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment", "run", *ROW_COLUMNS])
        writer.writeheader()
        for run_dir, event_paths in sorted(runs.items()):
            experiment, run = run_dir.parent.name, run_dir.name
            print("Reading {}/{} ...".format(experiment, run), flush=True)
            rows, counts, duplicates = read_rewards(event_paths)
            if not rows:
                print("  No CTS reward metrics recorded; skipping.", flush=True)
                continue
            relative = run_dir.relative_to(log_dir)
            run_path = output_dir / relative / "mean_rewards.csv"
            write_csv(run_path, ROW_COLUMNS, rows)
            writer.writerows(dict(experiment=experiment, run=run, **row) for row in rows)
            summary = {
                "experiment": experiment,
                "run": run,
                "rows": len(rows),
                "first_iteration": rows[0]["iteration"],
                "last_iteration": rows[-1]["iteration"],
                "teacher_points": sum("teacher_mean_reward" in row for row in rows),
                "student_points": sum("student_mean_reward" in row for row in rows),
                "last_teacher_mean_reward": rows[-1].get("teacher_mean_reward", ""),
                "last_student_mean_reward": rows[-1].get("student_mean_reward", ""),
                "csv_path": str(run_path.relative_to(output_dir)),
            }
            summaries.append(summary)
            print("  {} rows, iterations {}..{}, recorded points {}, duplicate points {}".format(
                len(rows), rows[0]["iteration"], rows[-1]["iteration"], counts, duplicates,
            ), flush=True)
    write_csv(output_dir / "mean_rewards_summary.csv", SUMMARY_COLUMNS, summaries)
    print("Exported {} runs and {} rows to {}".format(
        len(summaries), sum(row["rows"] for row in summaries), output_dir,
    ), flush=True)


def main():
    default_logs = Path(__file__).resolve().parents[2] / "logs" / "nezha_moe_cts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=default_logs)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    log_dir = args.log_dir.resolve()
    if not log_dir.is_dir():
        parser.error("Log directory does not exist: {}".format(log_dir))
    output_dir = (args.output_dir or log_dir / "reward_csv").resolve()
    export(log_dir, output_dir)


if __name__ == "__main__":
    main()
