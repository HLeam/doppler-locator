"""Portable CSV/JSON outputs and headless annotated plots."""

import csv
import json
import textwrap
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .audio import Spectrogram  # noqa: E402
from .detector import DESCENDING_DISCOVERY, STRICT_CURVE, Candidate, DetectorConfig  # noqa: E402

COLUMNS = [
    "event_id",
    "category",
    "start_s",
    "end_s",
    "duration_s",
    "observed_support_s",
    "observed_point_count",
    "maximum_observation_gap_s",
    "unobserved_intervals",
    "ambiguity_flags",
    "unverified_source_continuity_link_count",
    "start_hz",
    "end_hz",
    "drop_hz",
    "heuristic_score",
    "coverage",
    "median_contrast_db",
    "monotonicity",
    "normalized_fit_error",
    "curve_gain_over_linear",
    "curve_improvement_hz",
    "linear_rmse_hz",
    "curve_rmse_hz",
    "descending_movement_fraction",
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
    annotation_row = 0
    for candidate in candidates:
        color = "cyan" if candidate.category == STRICT_CURVE else "#ffb454"
        ax.plot(candidate.times, candidate.frequencies, ".", color=color, markersize=2)
        if candidate.connection_unverified is not None:
            for index in np.flatnonzero(candidate.connection_unverified[1:]) + 1:
                ax.plot(
                    candidate.times[index - 1 : index + 1],
                    candidate.frequencies[index - 1 : index + 1],
                    ":",
                    color="white",
                    alpha=0.65,
                    linewidth=0.55,
                )
        if len(candidates) <= 20:
            label = "strict curved event" if candidate.category == STRICT_CURVE else "unconfirmed descending tone"
            summary = candidate.summary()
            flags = ",".join(summary["ambiguity_flags"])
            ax.text(
                0.01,
                0.99 - annotation_row * 0.10,
                textwrap.fill(
                    f"{candidate.event_id} {label}; observed={summary['observed_support_s']:.1f}s; "
                f"gaps={len(summary['unobserved_intervals'])}; gain={candidate.curve_gain:.3f}; "
                f"improvement={summary['curve_improvement_hz']:.2f}Hz; "
                    f"unverified links={summary['unverified_source_continuity_link_count']}; {flags}",
                    width=100,
                    break_long_words=True,
                    break_on_hyphens=False,
                ),
                color=color,
                fontsize=7,
                ha="left",
                va="top",
                wrap=True,
                transform=ax.transAxes,
                bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none", "pad": 2},
            )
            annotation_row += 1
        if start is not None and candidate.category == STRICT_CURVE:
            ax.plot(
                candidate.times, candidate.fitted, "--", color="lime", linewidth=1, label="empirical shape"
            )
    has_links = any(
        candidate.connection_unverified is not None and np.any(candidate.connection_unverified)
        for candidate in candidates
    )
    association_note = "; dotted links are tracker associations with unverified source continuity" if has_links else ""
    ax.set(
        xlim=(time_edges[0], time_edges[-1]),
        ylim=(freq_edges[0], freq_edges[-1]),
        xlabel="Time from recording start (minutes:seconds)",
        ylabel="Frequency (Hz)",
        title=f"{title}\nHann FFT={spec.n_fft}; Δf={df:.2f} Hz; averaged Δt={spec.time_step:.3f} s{association_note}",
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
    review_spec: Spectrogram | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for candidate in candidates:
            row = candidate.summary()
            row["unobserved_intervals"] = json.dumps(row["unobserved_intervals"], separators=(",", ":"))
            row["ambiguity_flags"] = json.dumps(row["ambiguity_flags"], separators=(",", ":"))
            writer.writerow(row)
    with (directory / "ridges.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "event_id",
                "category",
                "point_index",
                "association_method",
                "connection_from_previous_s",
                "source_continuity_unverified",
                "time_s",
                "frequency_hz",
                "local_contrast_db",
                "comparison_fit_hz",
            ]
        )
        for candidate in candidates:
            link_flags = candidate.connection_unverified
            for index, (time, frequency, contrast, fit) in enumerate(
                zip(candidate.times, candidate.frequencies, candidate.contrasts, candidate.fitted, strict=True)
            ):
                linked = bool(link_flags[index]) if link_flags is not None else False
                dt = float(time - candidate.times[index - 1]) if index else 0.0
                writer.writerow(
                    (
                        candidate.event_id,
                        candidate.category,
                        index,
                        "minimum_cost_peak_link" if linked else "ridge_start",
                        dt,
                        linked,
                        time,
                        frequency,
                        contrast,
                        fit,
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
            "category_counts": dict(Counter(candidate.category for candidate in candidates)),
            "categories": {
                STRICT_CURVE: "Strict sigmoid-shaped descending event; not source identification",
                DESCENDING_DISCOVERY: "Observed descending tone with insufficient strict curvature evidence",
            },
            "score_warning": "Category-specific heuristic ranking, not a calibrated probability or source identification",
            "association_warning": "Each linked point records a peak association; source continuity is unverified, including for adjacent frames",
            "versions": {
                name: version(name)
                for name in ("doppler-locator", "numpy", "scipy", "soundfile", "matplotlib")
            },
        }
    )
    selected = sorted(candidates, key=lambda candidate: -candidate.score)[:max_plots]
    metadata["plotted_event_ids"] = [candidate.event_id for candidate in selected]
    if review_spec is not None:
        metadata["visual_review_spectrogram"] = {
            "n_fft": review_spec.n_fft,
            "purpose": "visual review only; candidate detection used the analysis spectrogram",
        }
    (directory / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    plot_spectrogram(spec, candidates, directory / "overview.png")
    for candidate in selected:
        plot_spectrogram(
            spec,
            [candidate],
            directory / f"{candidate.event_id}.png",
            candidate.times[0] - 10,
            candidate.times[-1] + 10,
            title=(
                f"{candidate.event_id}: {candidate.category}; observed {candidate.summary()['observed_support_s']:.1f}s; "
                f"max gap {candidate.summary()['maximum_observation_gap_s']:.2f}s; "
                f"curve gain {candidate.curve_gain:.3f}, improvement "
                f"{candidate.summary()['curve_improvement_hz']:.2f}Hz, "
                f"normalized error {candidate.fit_error:.3f}; "
                f"dotted tracker links have unverified source continuity; "
                f"ambiguity={','.join(candidate.summary()['ambiguity_flags'])}"
            ),
        )
    if review_spec is not None:
        plot_spectrogram(
            review_spec,
            candidates,
            directory / f"visual-review-fft-{review_spec.n_fft}.png",
            title=f"Visual review only; detections use analysis FFT={spec.n_fft}",
        )
