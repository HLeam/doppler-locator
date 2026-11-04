"""Observation provenance, uncertain harmonic relations and physical-fit review."""

import csv
import json
from pathlib import Path

import numpy as np

from .output import plt
from .physical import prediction_envelope, received_frequency


def candidate_track(candidate):
    return {
        "event_id": candidate.event_id,
        "category": candidate.category,
        "selection": "blind_detector",
        "time_step_s": candidate.time_step_s,
        "support": candidate.summary(),
        "points": [
            {
                "time_s": float(t),
                "frequency_hz": float(f),
                "local_contrast_db": float(c),
                "association_method": "ridge_start" if i == 0 else "minimum_cost_peak_link",
                "source_continuity_unverified": True,
            }
            for i, (t, f, c) in enumerate(zip(candidate.times, candidate.frequencies, candidate.contrasts))
        ],
    }


def load_diagnostic_points(path, time_step):
    """Explicit manual/externally selected diagnostic; never blind recovery.

    CSV may carry extra provenance columns (corridor, competing peaks, omissions).
    All columns are retained. Rows are not sorted or silently repaired.
    """
    tracks = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if not {"event_id", "time_s", "frequency_hz"} <= row.keys():
                raise ValueError("point CSV requires event_id,time_s,frequency_hz")
            event_id = row["event_id"]
            if not event_id or not all(c.isalnum() or c in "_-" for c in event_id):
                raise ValueError("diagnostic event_id must contain only letters, digits, '_' or '-'")
            point = dict(row)
            point.update(
                time_s=float(row["time_s"]),
                frequency_hz=float(row["frequency_hz"]),
                source_continuity_unverified=True,
            )
            point.setdefault("association_method", "externally_selected_unverified")
            track = tracks.setdefault(
                event_id,
                {
                    "event_id": event_id,
                    "category": "manually_selected_diagnostic",
                    "selection": "externally_selected_not_blind_recovery",
                    "time_step_s": time_step,
                    "points": [],
                },
            )
            track["points"].append(point)
    if not tracks:
        raise ValueError("point CSV is empty")
    return list(tracks.values())


def support(track):
    t = np.array([p["time_s"] for p in track["points"]])
    step = track["time_step_s"]
    delta = np.diff(t)
    return {
        "start_s": float(t[0]),
        "end_s": float(t[-1]),
        "observed_point_count": len(t),
        "time_step_s": step,
        "observed_support_s": float(step + np.minimum(delta, step).sum()),
        "gaps": [
            {"after_s": float(t[i]), "before_s": float(t[i + 1]), "unobserved_s": float(d - step)}
            for i, d in enumerate(delta)
            if d > 1.5 * step
        ],
        "warning": "support-bin duration, not independent sample support; source continuity unverified",
    }


def harmonic_relations(tracks, tolerance_hz=3.0):
    """Pairwise evidence only: match observed times, never interpolate gaps.

    A relation is a review grouping, NOT a count of independent physical events.
    Retain all tracks; integer ratio alone cannot establish a common emitter.
    """
    relations = []
    for i, a in enumerate(tracks):
        ta = np.array([p["time_s"] for p in a["points"]])
        fa = np.array([p["frequency_hz"] for p in a["points"]])
        for b in tracks[i + 1 :]:
            tb = np.array([p["time_s"] for p in b["points"]])
            fb = np.array([p["frequency_hz"] for p in b["points"]])
            at, bt = [], []
            # Two-pointer one-to-one match. Tolerance is less than a full frame;
            # omitted observations cannot be filled from a neighbor across a gap.
            j = k = 0
            dt = 0.25 * min(a["time_step_s"], b["time_step_s"])
            while j < len(ta) and k < len(tb):
                if abs(ta[j] - tb[k]) <= dt:
                    at.append(j)
                    bt.append(k)
                    j += 1
                    k += 1
                elif ta[j] < tb[k]:
                    j += 1
                else:
                    k += 1
            if len(at) < 12 or np.ptp(ta[at]) < 10:
                continue
            low, high = (fa[at], fb[bt]) if np.median(fa[at]) <= np.median(fb[bt]) else (fb[bt], fa[at])
            low_id = a["event_id"] if np.median(fa[at]) <= np.median(fb[bt]) else b["event_id"]
            high_id = b["event_id"] if low_id == a["event_id"] else a["event_id"]
            order = int(np.clip(round(float(np.median(high / low))), 1, 4))
            residual = np.abs(high / order - low)
            fraction = float(np.mean(residual <= tolerance_hz))
            if fraction >= 0.8 and np.ptp(low) > 2 * tolerance_hz:
                relations.append(
                    {
                        "low_id": low_id,
                        "high_id": high_id,
                        "order_ratio": order,
                        "label": "possible_duplicate" if order == 1 else "possible_harmonic",
                        "grouping_uncertain": True,
                        "paired_observed_points": len(at),
                        "paired_span_s": [float(ta[at[0]]), float(ta[at[-1]])],
                        "reference_residual_p90_hz": float(np.percentile(residual, 90)),
                        "fraction_within_tolerance": fraction,
                        "tolerance_hz": tolerance_hz,
                        "warning": "not common-source proof; no known fundamental or independent event count",
                    }
                )
    return relations


def write_fit_outputs(directory, tracks, results, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    relations = harmonic_relations(tracks, metadata["harmonic_tolerance_hz"])
    # JSON keeps original observations, missing intervals, and EVERY alternative.
    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "harmonic_relations": relations,
        "independent_source_count": None,
        "warning": "candidate counts are not physical-source counts; all associations unverified",
        "tracks": [
            {"observations": track, "support": support(track), "fit": result}
            for track, result in zip(tracks, results, strict=True)
        ],
    }
    (directory / "physical-fits.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with (directory / "physical-points.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "event_id",
                "category",
                "time_s",
                "observed_hz",
                "predicted_hz",
                "residual_hz",
                "association_method",
                "source_continuity_unverified",
            ]
        )
        for track, result in zip(tracks, results, strict=True):
            for point, predicted, residual in zip(
                track["points"], result["predicted_hz"], result["residual_hz"], strict=True
            ):
                writer.writerow(
                    [
                        track["event_id"],
                        track["category"],
                        point["time_s"],
                        point["frequency_hz"],
                        predicted,
                        residual,
                        point["association_method"],
                        True,
                    ]
                )
    return payload


def plot_fit(spec, track, result, path, detection_fft):
    t = np.array([p["time_s"] for p in track["points"]])
    f = np.array([p["frequency_hz"] for p in track["points"]])
    selection = (
        f"detection FFT {detection_fft}"
        if detection_fft is not None
        else "external points; no automatic detection"
    )
    margin = max(15, 0.15 * np.ptp(f))
    band = (spec.frequencies >= f.min() - margin) & (spec.frequencies <= f.max() + margin)
    mask = (spec.times >= t[0] - 10) & (spec.times <= t[-1] + 10)
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    if np.count_nonzero(mask) >= 2 and np.count_nonzero(band) >= 2:
        image = ax.pcolormesh(
            spec.times[mask],
            spec.frequencies[band],
            spec.power_db[mask][:, band].T,
            shading="auto",
            cmap="magma",
            vmin=-100,
            vmax=-65,
        )
        fig.colorbar(image, ax=ax, label="Relative digital bin power (dB; not SPL)")
    grid = np.linspace(t[0], t[-1], 400)
    lo, hi = prediction_envelope(grid, result)
    ax.fill_between(grid, lo, hi, color="lime", alpha=0.25, label="comparable-model envelope (not CI)")
    p = result["best_parameters"]
    ax.plot(
        grid,
        received_frequency(
            grid,
            p["emitted_reference_hz"],
            p["speed_ratio_v_over_c"],
            p["range_over_speed_s"],
            p["cpa_receive_s"],
        ),
        "--",
        color="lime",
        linewidth=1.1,
        label="physical prediction (also in gaps)",
    )
    ax.plot(t, result["linear_predicted_hz"], color="deepskyblue", linewidth=0.8, label="linear alternative")
    ax.scatter(t, f, s=8, c="white", label="observed peaks; association unverified")
    for gap in support(track)["gaps"]:
        ax.axvspan(gap["after_s"], gap["before_s"], color="gray", alpha=0.5)
    if result["status"] == "conditionally_constrained":
        for key, color, label in [
            ("cpa_receive_s", "cyan", "conditional model: arrival of CPA sound"),
            ("cpa_emission_s", "yellow", "conditional model: emission CPA / receive inflection"),
        ]:
            if t[0] - 10 <= p[key] <= t[-1] + 10:
                ax.axvline(p[key], color=color, linestyle=":", label=label)
    ax.set(
        xlim=(t[0] - 10, t[-1] + 10),
        ylim=(max(0, f.min() - margin), f.max() + margin),
        xlabel="Receive time from recording zero (s); gray = missing observations",
        ylabel="Frequency (Hz)",
        title=f"{track['event_id']} — {track['category']} — {result['status']}\n"
        f"Review FFT {spec.n_fft}; {selection}; "
        f"physical / linear RMSE {result['physical_rmse_hz']:.2f} / {result['linear_rmse_hz']:.2f} Hz\n"
        "No CPA marker when unconstrained; no identity or geographic inference",
    )
    ax.legend(fontsize=7, loc="best")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_audit(spec, tracks, start, end, path, detection_fft):
    """Full-time coverage without long annotation blocks hiding the raw tones."""
    mask = (spec.times >= start) & (spec.times <= end)
    if np.count_nonzero(mask) < 2:
        return
    lower, upper = float(spec.frequencies[0]), float(spec.frequencies[-1])
    bands = [(lower, 250), (250, upper)] if lower < 250 < upper else [(lower, upper)]
    fig, axes = plt.subplots(
        len(bands), 1, figsize=(14, 3.5 * len(bands)), sharex=True, layout="constrained", squeeze=False
    )
    axes = axes[:, 0]
    for ax, (low, high) in zip(axes, bands):
        band = (spec.frequencies >= low) & (spec.frequencies <= high)
        if np.count_nonzero(band) >= 2:
            image = ax.pcolormesh(
                spec.times[mask],
                spec.frequencies[band],
                spec.power_db[mask][:, band].T,
                shading="auto",
                cmap="magma",
                vmin=-100,
                vmax=-65,
            )
            fig.colorbar(image, ax=ax, label="Relative digital power (dB; not SPL)")
        for track in tracks:
            t = np.array([p["time_s"] for p in track["points"]])
            f = np.array([p["frequency_hz"] for p in track["points"]])
            selected = (t >= start) & (t <= end) & (f >= low) & (f <= high)
            if not np.any(selected):
                continue
            color = "cyan" if track["category"] == "strict_curved_event" else "#ffb454"
            ax.scatter(t[selected], f[selected], s=3, color=color)
            ax.text(
                float(np.median(t[selected])),
                float(np.median(f[selected])) + 8,
                track["event_id"],
                color="white",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
            )
        ax.set(xlim=(start, end), ylim=(low, high), ylabel="Frequency (Hz)")
    axes[-1].set_xlabel("Receive seconds from recording zero; observed points only (no gap filling)")
    selection = (
        f"detection FFT {detection_fft}"
        if detection_fft is not None
        else "external points; no automatic detection"
    )
    fig.suptitle(
        f"Coverage audit {start:g}–{end:g} s; review FFT {spec.n_fft}; {selection}\n"
        "Cyan = strict shape; orange = discovery/diagnostic; all associations unverified; not source counts"
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)
