import csv
import json

import numpy as np
import pytest
import soundfile as sf

from doppler_locator.physical_cli import main


def physical_audio(path):
    fs = 4000
    receive = np.arange(70 * fs) / fs
    emission = np.arange(-40, 120, 1 / fs)
    distance = np.sqrt(700**2 + (100 * (emission - 35)) ** 2)
    # Independent forward propagation and emitted phase, not the fitter's inverse.
    received_emission = np.interp(receive, emission + distance / 343, emission)
    tone = 0.2 * np.sin(2 * np.pi * 140 * received_emission)
    rng = np.random.default_rng(1)
    tone += 0.01 * rng.normal(size=len(tone))
    sf.write(path, np.column_stack((tone, -tone)), fs, subtype="FLOAT")
    return path


def test_blind_cli_retains_points_fits_and_distinct_review_fft(tmp_path):
    audio = physical_audio(tmp_path / "input.wav")
    output = tmp_path / "out"
    assert (
        main(
            [
                "--input",
                str(audio),
                "--output",
                str(output),
                "--n-fft",
                "1366",
                "--hop",
                "334",
                "--profile-points",
                "5",
            ]
        )
        == 0
    )
    result = json.loads((output / "physical-fits.json").read_text())
    assert result["metadata"]["detection_fft"] == 1366
    assert result["metadata"]["review_fft"] == 32768
    assert result["tracks"]
    assert result["independent_source_count"] is None
    with (output / "ridges.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for track in result["tracks"]:
        obs = track["observations"]
        source = [r for r in rows if r["event_id"] == obs["event_id"]]
        np.testing.assert_array_equal(
            [p["time_s"] for p in obs["points"]], [float(r["time_s"]) for r in source]
        )
        np.testing.assert_array_equal(
            [p["frequency_hz"] for p in obs["points"]], [float(r["frequency_hz"]) for r in source]
        )
        assert all(p["source_continuity_unverified"] for p in obs["points"])
        assert len(track["fit"]["residual_hz"]) == len(source)
        assert (output / f"{obs['event_id']}-physical-review.png").is_file()
    assert (output / "audit-000000.0.png").is_file()
    with pytest.raises(SystemExit):
        main(["--input", str(audio), "--output", str(output)])


def test_seeded_cli_is_not_automatic_recovery_and_keeps_gaps(tmp_path):
    audio = physical_audio(tmp_path / "input.wav")
    csv_path = tmp_path / "points.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["event_id", "time_s", "frequency_hz", "competing_hz"])
        for time in [10, 11, 12, 14, 15, 16]:
            writer.writerow(["manual", time, 170 - time, 190])
    output = tmp_path / "seeded"
    main(
        [
            "--input",
            str(audio),
            "--output",
            str(output),
            "--n-fft",
            "1366",
            "--hop",
            "334",
            "--points",
            str(csv_path),
            "--points-step-s",
            "1",
            "--profile-points",
            "5",
        ]
    )
    result = json.loads((output / "physical-fits.json").read_text())
    assert result["tracks"][0]["observations"]["category"] == "manually_selected_diagnostic"
    assert result["tracks"][0]["support"]["gaps"]
    assert result["tracks"][0]["observations"]["points"][0]["competing_hz"] == "190"
    assert json.loads((output / "coverage.json").read_text())["automatic_candidate_tracks"] == 0
    assert result["tracks"][0]["fit"]["status"] == "insufficient_evidence"


@pytest.mark.parametrize(
    ("header", "row", "message"),
    [
        ("event_id,time_s,frequency_hz", "manual,,170", "requires event_id,time_s,frequency_hz values"),
        ("event_id,time_s,frequency_hz", "manual,10,", "requires event_id,time_s,frequency_hz values"),
        ("event_id,time_s,frequency_hz", "manual,not-a-time,170", "must be numeric"),
        ("event_id,time_s,frequency_hz", "manual,10,nan", "must be finite"),
        ("event_id,time_s", "manual,10", "requires event_id,time_s,frequency_hz"),
    ],
)
def test_seeded_cli_rejects_malformed_points_with_actionable_error(tmp_path, capsys, header, row, message):
    audio = physical_audio(tmp_path / "input.wav")
    csv_path = tmp_path / "points.csv"
    csv_path.write_text(f"{header}\n{row}\n")
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--input",
                str(audio),
                "--output",
                str(tmp_path / "out"),
                "--n-fft",
                "1366",
                "--hop",
                "334",
                "--points",
                str(csv_path),
                "--points-step-s",
                "1",
                "--profile-points",
                "5",
            ]
        )
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_zero_candidate_cli_has_valid_empty_exports(tmp_path):
    audio = tmp_path / "silent.wav"
    sf.write(audio, np.zeros(4000 * 12), 4000)
    output = tmp_path / "empty"
    main(["--input", str(audio), "--output", str(output), "--n-fft", "1366", "--hop", "334"])
    result = json.loads((output / "physical-fits.json").read_text())
    assert result["tracks"] == []
    assert result["harmonic_relations"] == []
    assert (output / "physical-points.csv").read_text().count("\n") == 1
