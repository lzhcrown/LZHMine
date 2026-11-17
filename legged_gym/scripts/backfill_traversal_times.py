#!/usr/bin/env python
"""Add per-attempt passage times to completed captures without changing source CSVs."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from ._bootstrap import configure_runtime
else:
    from _bootstrap import configure_runtime
configure_runtime()

from legged_gym.evaluation.timing import TIMING_FIELDS, traversal_record


def source_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def write_csv(path, rows):
    # Exclusive creation protects any existing analysis/user edits.
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def backfill(directory):
    directory = Path(directory).resolve()
    sources = ([directory / 'episodes.csv'] if (directory / 'episodes.csv').is_file()
               else sorted(directory.glob('*/episodes.csv')))
    if not sources:
        raise FileNotFoundError('No episodes.csv found in ' + str(directory))
    all_rows, groups, outputs = [], {}, []
    for source in sources:
        metadata_path = source.with_name('metadata.json')
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('status') != 'complete':
            raise ValueError('Capture is still running or incomplete: ' + str(source.parent))
        before_hash = source_hash(source)
        with source.open(newline='') as handle:
            episodes = list(csv.DictReader(handle))
        if len(episodes) != metadata['recorded_attempts']:
            raise ValueError('Attempt count differs from metadata: ' + str(source))
        records = [traversal_record(row, metadata.get('control_dt_s')) for row in episodes]
        if any(not row.get('trial_id') for row in records):
            raise ValueError('Every attempt must have a trial_id: ' + str(source))
        invalid = [row for row in records if row['status'] == 'success' and row['timing_status'] != 'complete']
        if invalid:
            raise ValueError('Successful attempts have missing/invalid timing: ' + str(source))
        for row, episode in zip(records, episodes):
            if row['timing_status'] == 'complete' and episode.get('elapsed_s'):
                if not math.isclose(row['traversal_time_s'], float(episode['elapsed_s']), rel_tol=1e-7, abs_tol=1e-8):
                    raise ValueError('Entry/exit clocks disagree with measured duration: ' + row['trial_id'])
        if source_hash(source) != before_hash:
            raise ValueError('Source changed while reading: ' + str(source))
        outputs.append((source.with_name('traversal_time.csv'), records))
        groups[source.parent.name] = dict(attempts=len(records),
            successes=sum(row['timing_status'] == 'complete' for row in records),
            source_episodes_sha256=before_hash, source_metadata_sha256=source_hash(metadata_path),
            time_resolution_s=metadata.get('control_dt_s'))
        all_rows.extend(records)
    if len({row['trial_id'] for row in all_rows}) != len(all_rows):
        raise ValueError('Duplicate trial IDs across captures')
    if len(sources) > 1:
        outputs.append((directory / 'traversal_times_all.csv', all_rows))
    report_path = directory / 'traversal_time_backfill.json'
    for path in [path for path, _ in outputs] + [report_path]:
        if path.exists():
            raise FileExistsError('Refusing to overwrite: ' + str(path))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), groups=groups,
        attempts=len(all_rows), statuses=dict(Counter(row['status'] for row in all_rows)),
        successful_times=sum(row['timing_status'] == 'complete' for row in all_rows),
        definitions=dict(traversal_time_s='successful exit_time_s minus first-wheel entry_time_s',
                         total_time_to_success_s='successful trial_elapsed_s, including approach'),
        outputs=[str(path) for path, _ in outputs])
    for path, rows in outputs:
        write_csv(path, rows)
    with report_path.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='One completed model capture or a batch root.')
    args = parser.parse_args()
    report = backfill(args.directory)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
