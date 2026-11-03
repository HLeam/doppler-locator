"""Reproduce development checks; not an independent benchmark or labeler.

Run after installing the project, with a local --input and an empty --output.
The detector itself never receives these approximate label intervals.
"""

import argparse
import csv
import json
from pathlib import Path

from doppler_locator.cli import main as detect_main

LABELS = [("development-1", 1230.0, 1290.0), ("development-2", 2100.0, 2175.0)]


def summarize(directory: Path, labels: list[tuple[str, float, float]]) -> dict:
    with (directory / "candidates.csv").open() as stream:
        events = list(csv.DictReader(stream))
    matches = {}
    for name, start, end in labels:
        matches[name] = [
            event["event_id"]
            for event in events
            if min(float(event["end_s"]), end) - max(float(event["start_s"]), start) >= 10
        ]
    matched = {event_id for ids in matches.values() for event_id in ids}
    return {
        "candidate_count": len(events),
        "development_temporal_matches": matches,
        "unmatched_candidates_not_known_false_positives": [
            event["event_id"] for event in events if event["event_id"] not in matched
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--full-hour", action="store_true", help="also scan all audio, using default parameters"
    )
    args = parser.parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output must be a new or empty directory")
    runs = [
        ("development-1", "20:00", "22:00", [LABELS[0]]),
        ("development-2", "34:30", "36:45", [LABELS[1]]),
        ("separate-10-13min", "10:00", "13:00", []),
    ]
    if args.full_hour:
        runs.append(("full-hour", None, None, LABELS))
    report = {
        "warning": "Development labels only; temporal overlap is not proof of curve identity. Inspect images.",
        "match_rule": "At least 10 seconds of candidate overlap with an approximate supplied interval",
        "unlabeled_warning": "Separate interval and unmatched candidates have no independent ground truth",
        "runs": {},
    }
    for name, start, end, labels in runs:
        destination = args.output / name
        command = ["--input", str(args.input), "--output", str(destination)]
        if start is not None:
            command += ["--start", start, "--end", end]
        detect_main(command)
        report["runs"][name] = summarize(destination, labels)
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
