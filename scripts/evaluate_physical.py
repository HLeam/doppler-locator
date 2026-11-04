"""Deterministic independent emission-time controls; no real media needed.

Run: python scripts/evaluate_physical.py --output <new directory>
The truth generator never calls received_frequency or an inverse propagation
formula: it maps a fine emission-time grid FORWARD, then interpolates phase.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from doppler_locator.audio import spectrogram
from doppler_locator.detector import DetectorConfig, detect
from doppler_locator.physical import fit_observations, received_frequency
from doppler_locator.physical_output import candidate_track, harmonic_relations


def forward(emission, f0, v, b, t0, c=343):
    position = v * (emission - t0)
    distance = np.sqrt(b * b + position * position)
    receive = emission + distance / c
    frequency = f0 / (1 + v * position / distance / c)
    return receive, frequency


def compact(result):
    return {
        key: value
        for key, value in result.items()
        if key not in {"searches", "predicted_hz", "residual_hz", "linear_predicted_hz"}
    }


def evaluate(output):
    output.mkdir(parents=True, exist_ok=False)
    trajectory = []
    for phase, noise in [(0, 0), (0.17, 0.3), (0.39, 1.0)]:
        for name, lo, hi, b in [
            ("full", -200, 200, 1600),
            ("large_delay", -400, 400, 15000),
            ("truncated", -5, 5, 1600),
            ("far_tail", 100, 130, 1600),
        ]:
            v, c, f0, t0 = 80, 343, 140, 1000
            emission = np.linspace(t0 + lo + phase, t0 + hi + phase, 201)
            t, f = forward(emission, f0, v, b, t0, c)
            check = received_frequency(t, f0, v / c, b / v, t0 + b / c)
            h = 0.001
            before, _ = forward(emission - h, f0, v, b, t0, c)
            after, _ = forward(emission + h, f0, v, b, t0, c)
            phase_derivative = f0 * 2 * h / (after - before)
            assert np.max(abs(check - f)) < 1e-9
            assert np.max(abs(phase_derivative - f)) < 1e-4
            observed = f + np.random.default_rng(23).normal(0, noise, len(f))
            result = fit_observations(t, observed)
            trajectory.append(
                {
                    "case": name,
                    "phase_s": phase,
                    "noise_sigma_hz": noise,
                    "truth": {
                        "f0_hz": f0,
                        "speed_m_s": v,
                        "slant_distance_m": b,
                        "emission_cpa_s": t0,
                        "receive_cpa_s": t0 + b / c,
                    },
                    "forward_inverse_max_error_hz": float(np.max(abs(check - f))),
                    "phase_derivative_max_error_hz": float(np.max(abs(phase_derivative - f))),
                    "observed_times_s": t.tolist(),
                    "observed_hz": observed.tolist(),
                    "fit": compact(result),
                }
            )
    (output / "trajectory-controls.json").write_text(json.dumps(trajectory, indent=2, allow_nan=False) + "\n")
    audio_results = []
    fs, duration = 48000, 80
    receive_grid = np.arange(duration * fs) / fs
    emission = np.arange(-40, 120, 1 / 4000)
    mapped_time, mapped_freq = forward(emission, 140, 100, 700, 40)
    for kind, phase, noise in [
        ("physical", 0, 0.01),
        ("physical", 0.7, 0.08),
        ("physical", 1.9, 0.2),
        ("harmonic", 0.7, 0.08),
        ("physical_gap", 0.7, 0.08),
        ("crossing_constant", 0.7, 0.08),
        ("independent_lines", 0.7, 0.03),
        ("dual_descending", 0.7, 0.03),
        ("linear", 0.7, 0.03),
        ("constant", 0.7, 0.03),
    ]:
        if kind in {"physical", "harmonic", "physical_gap", "crossing_constant"}:
            # Emitted phase -> receive phase using ONLY the forward propagation grid.
            emitted_at_receive = np.interp(receive_grid, mapped_time, emission)
            samples = 0.15 * np.sin(2 * np.pi * 140 * emitted_at_receive + phase)
            if kind == "harmonic":
                samples += 0.10 * np.sin(2 * np.pi * 280 * emitted_at_receive + 0.3)
            if kind == "crossing_constant":
                samples += 0.20 * np.sin(2 * np.pi * 130 * receive_grid)
            if kind == "physical_gap":
                samples[(receive_grid >= 35) & (receive_grid < 38)] = 0
        else:
            f1 = np.full_like(receive_grid, 160) if kind == "constant" else 190 - receive_grid
            samples = 0.15 * np.sin(2 * np.pi * np.cumsum(f1) / fs + phase)
            if kind in {"independent_lines", "dual_descending"}:
                f2 = 110 + receive_grid if kind == "independent_lines" else 175 - 0.7 * receive_grid
                samples += 0.15 * np.sin(2 * np.pi * np.cumsum(f2) / fs)
        samples += noise * np.random.default_rng(41).normal(size=len(samples))
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            path = Path(temporary) / "control.wav"
            sf.write(path, np.column_stack((samples, -samples)), fs, subtype="FLOAT")
            spec = spectrogram(path, fmin=20, fmax=800, n_fft=16384, hop=4012)
        candidates = detect(spec, DetectorConfig(min_duration_s=20, min_drop_hz=30))
        rows = []
        for candidate in candidates:
            fit = fit_observations(candidate.times, candidate.frequencies)
            truth = np.interp(candidate.times, mapped_time, mapped_freq)
            error = float(np.percentile(abs(candidate.frequencies - truth), 95))
            rows.append(
                {
                    "observations": candidate_track(candidate),
                    "fit": compact(fit),
                    "p95_reference_track_error_hz": error
                    if kind in {"physical", "harmonic", "physical_gap", "crossing_constant"}
                    else None,
                    "source_identity_confirmed": False,
                }
            )
        audio_results.append(
            {
                "case": kind,
                "phase_radians": phase,
                "sample_noise_sigma": noise,
                "detection_fft": 16384,
                "hop": 4012,
                "sample_rate": fs,
                "candidates": rows,
                "harmonic_relations": harmonic_relations([candidate_track(c) for c in candidates]),
            }
        )
        print(kind, phase, noise, len(candidates), flush=True)
    (output / "audio-controls.json").write_text(json.dumps(audio_results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    evaluate(parser.parse_args().output)
