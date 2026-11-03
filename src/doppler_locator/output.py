"""Portable CSV/JSON outputs and headless annotated plots."""

import csv
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .audio import Spectrogram  # noqa: E402
from .detector import Candidate, DetectorConfig  # noqa: E402

COLUMNS = [
    "event_id",
    "start_s",
    "end_s",
    "duration_s",
    "start_hz",
    "end_hz",
    "drop_hz",
    "heuristic_score",
    "coverage",
    "median_contrast_db",
    "monotonicity",
    "normalized_fit_error",
    "curve_gain_over_linear",
    "ridge_points",
]


def plot_spectrogram(
    spec: Spectrogram,
    candidates: list[Candidate],
    path: Path,
    start: float | None = None,
    end: float | None = None,
    title: str = "Candidate descending curves (not source identification)",
) -> None:
    mask = np.ones(len(spec.times), dtype=bool)
    if start is not None:
        mask &= spec.times >= start
    if end is not None:
        mask &= spec.times <= end
    times, db = spec.times[mask], spec.power_db[mask]
    # Bound plot size. Use linear-power averaging, not a decimating sample.
    group = max(1, int(np.ceil(len(times) / 2000)))
    if group > 1:
        groups = [slice(i, min(i + group, len(times))) for i in range(0, len(times), group)]
        times = np.array([times[g].mean() for g in groups])
        db = np.array([10 * np.log10(np.mean(10 ** (db[g] / 10), axis=0)) for g in groups])
    fig, ax = plt.subplots(figsize=(13, 5))
    dt = spec.time_step * group
    df = spec.frequencies[1] - spec.frequencies[0]
    time_edges = np.r_[times[0] - dt / 2, (times[:-1] + times[1:]) / 2, times[-1] + dt / 2]
    freq_edges = np.r_[spec.frequencies - df / 2, spec.frequencies[-1] + df / 2]
    low, high = np.percentile(db, [10, 99.5])
    image = ax.pcolormesh(time_edges, freq_edges, db.T, cmap="magma", vmin=low, vmax=max(high, low + 1))
    for candidate in candidates:
        ax.plot(candidate.times, candidate.frequencies, ".", color="cyan", markersize=2)
        if len(candidates) <= 20:
            ax.text(
                candidate.times[0],
                candidate.frequencies[0] + df * 2,
                f"{candidate.event_id} score={candidate.score:.2f}",
                color="cyan",
                fontsize=8,
            )
        if start is not None:
            ax.plot(
                candidate.times, candidate.fitted, "--", color="lime", linewidth=1, label="empirical shape"
            )
    ax.set(
        xlim=(time_edges[0], time_edges[-1]),
        ylim=(freq_edges[0], freq_edges[-1]),
        xlabel="Time from recording start (minutes:seconds)",
        ylabel="Frequency (Hz)",
        title=f"{title}\nHann FFT={spec.n_fft}; Δf={df:.2f} Hz; averaged Δt={spec.time_step:.3f} s",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value) // 60}:{int(value) % 60:02d}"))
    fig.colorbar(image, ax=ax, label="Relative digital power (dB; not calibrated SPL)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_outputs(
    directory: Path,
    spec: Spectrogram,
    candidates: list[Candidate],
    config: DetectorConfig,
    metadata: dict,
    max_plots: int = 20,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(candidate.summary() for candidate in candidates)
    with (directory / "ridges.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["event_id", "time_s", "frequency_hz", "local_contrast_db", "empirical_shape_hz"])
        for candidate in candidates:
            writer.writerows(
                (candidate.event_id, *values)
                for values in zip(
                    candidate.times, candidate.frequencies, candidate.contrasts, candidate.fitted, strict=True
                )
            )
    metadata.update(
        {
            "detector": asdict(config),
            "spectrogram": {
                "sample_rate": spec.sample_rate,
                "n_fft": spec.n_fft,
                "hop": spec.hop,
                "frames_per_bin": spec.frames_per_bin,
                "actual_time_step_s": spec.time_step,
                "frequency_step_hz": float(spec.frequencies[1] - spec.frequencies[0]),
            },
            "candidate_count": len(candidates),
            "score_warning": "Heuristic ranking score, not a calibrated probability or source identification",
            "versions": {
                name: version(name)
                for name in ("doppler-locator", "numpy", "scipy", "soundfile", "matplotlib")
            },
        }
    )
    selected = sorted(candidates, key=lambda candidate: -candidate.score)[:max_plots]
    metadata["plotted_event_ids"] = [candidate.event_id for candidate in selected]
    (directory / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    plot_spectrogram(spec, candidates, directory / "overview.png")
    for candidate in selected:
        plot_spectrogram(
            spec,
            [candidate],
            directory / f"{candidate.event_id}.png",
            candidate.times[0] - 10,
            candidate.times[-1] + 10,
            title=f"{candidate.event_id}: heuristic score {candidate.score:.3f} (not probability)",
        )
