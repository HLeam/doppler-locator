"""Reproduce development checks; not an independent benchmark or labeler.

Run after installing the project, with a local --input and an empty --output.
The detector itself never receives these approximate label intervals.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from doppler_locator.cli import main as detect_main

LABELS = [("development-1", 1230.0, 1290.0), ("development-2", 2100.0, 2175.0)]


def summarize(directory: Path, labels: list[tuple[str, float, float]]) -> dict:
    with (directory / "candidates.csv").open() as stream:
        events = list(csv.DictReader(stream))
    points_by_event: dict[str, list[dict]] = {}
    with (directory / "ridges.csv").open() as stream:
        for point in csv.DictReader(stream):
            points_by_event.setdefault(point["event_id"], []).append(
                {
                    "time_s": float(point["time_s"]),
                    "frequency_hz": float(point["frequency_hz"]),
                }
            )

    def details(event: dict, overlap_s: float | None = None) -> dict:
        result = {
            "event_id": event["event_id"],
            "category": event["category"],
            "start_s": float(event["start_s"]),
            "end_s": float(event["end_s"]),
            "start_hz": float(event["start_hz"]),
            "end_hz": float(event["end_hz"]),
            "observed_support_s": float(event["observed_support_s"]),
            "coverage": float(event["coverage"]),
            "unobserved_intervals": json.loads(event["unobserved_intervals"]),
            "ambiguity_flags": json.loads(event["ambiguity_flags"]),
            "linear_rmse_hz": float(event["linear_rmse_hz"]),
            "curve_rmse_hz": float(event["curve_rmse_hz"]),
            "curve_gain_over_linear": float(event["curve_gain_over_linear"]),
            "curve_improvement_hz": float(event["curve_improvement_hz"]),
            "observed_frequency_path": points_by_event.get(event["event_id"], []),
        }
        if overlap_s is not None:
            result["label_overlap_s"] = overlap_s
        return result

    matches = {}
    for name, start, end in labels:
        matches[name] = []
        for event in events:
            overlap = min(float(event["end_s"]), end) - max(float(event["start_s"]), start)
            if overlap >= 10:
                matches[name].append(details(event, overlap))
    matched = {item["event_id"] for items in matches.values() for item in items}
    return {
        "candidate_count": len(events),
        "category_counts": dict(Counter(event["category"] for event in events)),
        "development_overlap_matches_not_truth_labels": matches,
        "unmatched_candidates_not_known_false_positives": [
            details(event) for event in events if event["event_id"] not in matched
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
