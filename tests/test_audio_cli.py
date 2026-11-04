import argparse
import csv
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from doppler_locator.audio import spectrogram
from doppler_locator.cli import main, seconds


def test_exact_8024_native_rate_and_absolute_time(make_audio):
    path = make_audio("constant", duration=3, fs=48000)
    s = spectrogram(path, start=1, end=2.8, n_fft=8024, fmin=300, fmax=600)
    assert s.n_fft == 8024
    assert s.hop == 4012
    assert s.frames_per_bin == 6
    assert np.diff(s.frequencies) == pytest.approx(48000 / 8024)
    assert s.times[0] == pytest.approx(2 * 6 * 4012 / 48000 + (2.5 * 4012 + 8024 / 2) / 48000)
    assert s.times[-1] < 2.8
    assert abs(s.frequencies[np.argmax(s.power_db.mean(axis=0))] - 420) < 6


def test_overlapping_crops_share_recording_aligned_spectral_evidence(make_audio):
    path = make_audio("curve", duration=8, fs=4000)
    whole = spectrogram(path, n_fft=800, fmin=300, fmax=600)
    crop = spectrogram(path, start=1.13, end=6.47, n_fft=800, fmin=300, fmax=600)
    selected = (whole.times >= crop.times[0]) & (whole.times <= crop.times[-1])
    np.testing.assert_array_equal(crop.times, whole.times[selected])
    np.testing.assert_allclose(crop.power_db, whole.power_db[selected], atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": -1},
        {"end": 999},
        {"end": 0.01},
        {"start": 2, "end": 1},
        {"fmin": 500, "fmax": 300},
        {"fmax": 3000},
        {"fmin": 400, "fmax": 401},
        {"n_fft": 8},
        {"hop": 9999},
        {"time_step": 0},
        {"chunk_seconds": 0},
        {"start": float("nan")},
        {"end": float("inf")},
        {"channel": "right"},
    ],
)
def test_invalid_analysis_settings(make_audio, kwargs):
    with pytest.raises(ValueError):
        spectrogram(make_audio(duration=3), **kwargs)


def test_nonfinite_audio_rejected(tmp_path):
    path = tmp_path / "nonfinite.wav"
    sf.write(path, np.full(4000, np.nan), 4000, subtype="FLOAT")
    with pytest.raises(ValueError, match="non-finite"):
        spectrogram(path, n_fft=800)


@pytest.mark.parametrize("value", ["-1", "1:60", "1:2:3:4", "abc"])
def test_invalid_time_syntax(value):
    with pytest.raises(argparse.ArgumentTypeError):
        seconds(value)


def test_cli_output_contract_and_refuses_overwrite(make_audio, tmp_path):
    path = make_audio()
    output = tmp_path / "results"
    command = [
        sys.executable,
        "-m",
        "doppler_locator",
        "--input",
        str(path),
        "--output",
        str(output),
        "--n-fft",
        "800",
        "--review-fft",
        "1024",
        "--max-plots",
        "1",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "1 candidate(s)" in result.stdout
    with (output / "candidates.csv").open() as stream:
        events = list(csv.DictReader(stream))
    assert len(events) == 1
    assert events[0]["category"] == "strict_curved_event"
    assert 0 <= float(events[0]["heuristic_score"]) <= 1
    assert float(events[0]["observed_support_s"]) > 0
    assert events[0]["unobserved_intervals"] == "[]"
    assert "source_identity_unconfirmed" in events[0]["ambiguity_flags"]
    assert float(events[0]["start_s"]) < 35 < float(events[0]["end_s"])
    with (output / "ridges.csv").open() as stream:
        points = list(csv.DictReader(stream))
    assert len(points) == int(events[0]["ridge_points"])
    assert set(point["category"] for point in points) == {"strict_curved_event"}
    assert set(point["event_id"] for point in points) == {"E0001"}
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["candidate_count"] == 1
    assert metadata["category_counts"] == {"strict_curved_event": 1}
    assert metadata["input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["spectrogram"]["n_fft"] == 800
    assert metadata["visual_review_spectrogram"]["n_fft"] == 1024
    assert metadata["category_counts"] == {"strict_curved_event": 1}
    assert str(tmp_path) not in (output / "run.json").read_text()
    assert metadata["plotted_event_ids"] == ["E0001"]
    for name in ("overview.png", "E0001.png", "visual-review-fft-1024.png"):
        with Image.open(output / name) as image:
            image.verify()
    summarize = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "evaluate_reference.py"))["summarize"]
    report = summarize(output, [("synthetic", 0, 70)])
    assert report["category_counts"] == {"strict_curved_event": 1}
    detail = report["development_overlap_matches_not_truth_labels"]["synthetic"][0]
    assert detail["curve_rmse_hz"] > 0
    assert detail["unobserved_intervals"] == []
    assert len(detail["observed_frequency_path"]) == int(events[0]["observed_point_count"])
    before = (output / "candidates.csv").read_bytes()
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode == 2
    assert "refusing to overwrite" in again.stderr
    assert (output / "candidates.csv").read_bytes() == before


def test_zero_candidate_outputs_and_selected_interval(make_audio, tmp_path):
    path = make_audio("constant", duration=10)
    output = tmp_path / "negative"
    assert (
        main(
            [
                "--input",
                str(path),
                "--output",
                str(output),
                "--n-fft",
                "800",
                "--start",
                "0:01",
                "--end",
                "0:09",
                "--max-plots",
                "0",
            ]
        )
        == 0
    )
    with (output / "candidates.csv").open() as stream:
        assert list(csv.DictReader(stream)) == []
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["analysis"]["start_s"] == 1
    assert metadata["analysis"]["end_s"] == 9
    assert metadata["candidate_count"] == 0
    assert (output / "overview.png").is_file()


def test_cli_exports_descending_tone_discovery_category(make_audio, tmp_path):
    path = make_audio("linear", duration=70, noise=0.03)
    output = tmp_path / "discovery"
    assert main(
        [
            "--input",
            str(path),
            "--output",
            str(output),
            "--n-fft",
            "800",
            "--max-plots",
            "1",
        ]
    ) == 0
    with (output / "candidates.csv").open() as stream:
        events = list(csv.DictReader(stream))
    assert events
    assert {event["category"] for event in events} == {"unconfirmed_descending_tone"}
    assert all("strict_curvature_gate_not_met" in event["ambiguity_flags"] for event in events)
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["category_counts"] == {"unconfirmed_descending_tone": len(events)}


@pytest.mark.parametrize("kind", ["genuine_gap", "sequential_handoff"])
def test_cli_outputs_separate_observed_segments(make_audio, tmp_path, kind):
    path = make_audio(kind, noise=0.01)
    output = tmp_path / kind
    assert main(
        ["--input", str(path), "--output", str(output), "--n-fft", "800", "--max-plots", "0"]
    ) == 0
    with (output / "candidates.csv").open() as stream:
        events = list(csv.DictReader(stream))
    with (output / "ridges.csv").open() as stream:
        points = list(csv.DictReader(stream))
    by_event = {}
    for point in points:
        by_event.setdefault(point["event_id"], []).append(float(point["time_s"]))
    assert events
    assert all(event["category"] == "unconfirmed_descending_tone" for event in events)
    assert any(max(times) < 24 for times in by_event.values())
    assert any(min(times) > 24 for times in by_event.values())
    assert all(not (min(times) < 24 and max(times) > 24) for times in by_event.values())


@pytest.mark.parametrize("kind", ["continuous_sweep", "seamless_handoff"])
def test_cli_exports_unverified_tracker_link_provenance(make_audio, tmp_path, kind):
    path = make_audio(kind, noise=0.03)
    output = tmp_path / kind
    assert main(
        ["--input", str(path), "--output", str(output), "--n-fft", "800", "--max-plots", "1"]
    ) == 0
    with (output / "candidates.csv").open() as stream:
        events = list(csv.DictReader(stream))
    with (output / "ridges.csv").open() as stream:
        points = list(csv.DictReader(stream))
    points_by_event = {}
    for point in points:
        points_by_event.setdefault(point["event_id"], []).append(point)
    crossing = [
        event
        for event in events
        if float(event["start_s"]) < 35 < float(event["end_s"])
    ]
    assert crossing
    for event in crossing:
        linked_points = points_by_event[event["event_id"]]
        assert linked_points[0]["association_method"] == "ridge_start"
        assert linked_points[0]["source_continuity_unverified"] == "False"
        assert all(point["association_method"] == "minimum_cost_peak_link" for point in linked_points[1:])
        assert all(point["source_continuity_unverified"] == "True" for point in linked_points[1:])
        assert int(event["unverified_source_continuity_link_count"]) == len(linked_points) - 1
        assert "tracker_links_source_continuity_unverified" in event["ambiguity_flags"]


@pytest.mark.parametrize("value,expected", [("20:30", 1230), ("01:00:00", 3600), ("3.5", 3.5)])
def test_time_parser(value, expected):
    assert seconds(value) == expected


def test_missing_input_cli(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--input", str(tmp_path / "absent.wav"), "--output", str(tmp_path / "out")])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
