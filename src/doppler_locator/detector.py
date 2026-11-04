"""A peak-linking baseline with empirical shape scores, not physical fits."""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from .audio import Spectrogram

STRICT_CURVE = "strict_curved_event"
DESCENDING_DISCOVERY = "unconfirmed_descending_tone"


@dataclass(frozen=True)
class DetectorConfig:
    peak_db: float = 1.5
    background_hz: float = 100.0
    max_rate_hz_s: float = 12.0
    max_gap_s: float = 2.0
    min_duration_s: float = 15.0
    max_duration_s: float = 120.0
    min_drop_hz: float = 60.0
    min_coverage: float = 0.65
    min_curve_gain: float = 0.05
    min_score: float = 0.65

    def validate(self) -> None:
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError("detector settings must be finite")
        for name in (
            "peak_db",
            "background_hz",
            "max_rate_hz_s",
            "max_gap_s",
            "min_duration_s",
            "min_drop_hz",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_duration_s < self.min_duration_s:
            raise ValueError("max_duration_s must be >= min_duration_s")
        for name in ("min_coverage", "min_curve_gain", "min_score"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass
class Ridge:
    frames: list[int] = field(default_factory=list)
    frequencies: list[float] = field(default_factory=list)
    contrasts: list[float] = field(default_factory=list)


@dataclass
class Candidate:
    times: np.ndarray
    frequencies: np.ndarray
    contrasts: np.ndarray
    fitted: np.ndarray
    score: float
    coverage: float
    drop_hz: float
    monotonicity: float
    fit_error: float
    curve_gain: float
    movement_fraction: float
    linear_rmse_hz: float
    curve_rmse_hz: float
    time_step_s: float
    category: str
    event_id: str = ""

    def summary(self) -> dict:
        deltas = np.diff(self.times)
        gaps = [
            {
                "after_s": float(self.times[i]),
                "before_s": float(self.times[i + 1]),
                "unobserved_s": float(max(0, deltas[i] - self.time_step_s)),
            }
            for i in range(len(deltas))
            if deltas[i] > 1.5 * self.time_step_s
        ]
        ambiguity = ["source_identity_unconfirmed"]
        if self.category == DESCENDING_DISCOVERY:
            ambiguity.append("strict_curvature_gate_not_met")
        if gaps:
            ambiguity.append("unobserved_intervals")
        return {
            "event_id": self.event_id,
            "category": self.category,
            "start_s": float(self.times[0]),
            "end_s": float(self.times[-1]),
            "duration_s": float(np.ptp(self.times)),
            "observed_support_s": float(self.time_step_s + np.minimum(deltas, self.time_step_s).sum()),
            "observed_point_count": len(self.times),
            "maximum_observation_gap_s": float(deltas.max()) if len(deltas) else 0.0,
            "unobserved_intervals": gaps,
            "ambiguity_flags": ambiguity,
            "start_hz": float(np.median(self.frequencies[: max(3, len(self.times) // 10)])),
            "end_hz": float(np.median(self.frequencies[-max(3, len(self.times) // 10) :])),
            "drop_hz": self.drop_hz,
            "heuristic_score": self.score,
            "coverage": self.coverage,
            "median_contrast_db": float(np.median(self.contrasts)),
            "monotonicity": self.monotonicity,
            "normalized_fit_error": self.fit_error,
            "curve_gain_over_linear": self.curve_gain,
            "curve_improvement_hz": self.linear_rmse_hz - self.curve_rmse_hz,
            "linear_rmse_hz": self.linear_rmse_hz,
            "curve_rmse_hz": self.curve_rmse_hz,
            "descending_movement_fraction": self.movement_fraction,
            "ridge_points": len(self.times),
        }


def extract_ridges(spec: Spectrogram, config: DetectorConfig) -> list[Ridge]:
    """Link salient peaks by minimum-cost assignment, one per ridge and bin.

    A four-observation velocity estimate helps through quantized plateaus. Links
    are limited by rate plus 2.25 frequency bins of quantization tolerance.
    Gaps are allowed but count against coverage. Crossing ridges can swap.
    """
    config.validate()
    df = float(spec.frequencies[1] - spec.frequencies[0])
    smooth = gaussian_filter(spec.power_db, sigma=(1.0, 0.5))
    width = max(5, round(config.background_hz / df) | 1)
    contrast = smooth - median_filter(smooth, size=(1, width), mode="nearest")
    active: list[Ridge] = []
    finished: list[Ridge] = []
    for frame, row in enumerate(contrast):
        now = spec.times[frame]
        still_active = []
        for ridge in active:
            if now - spec.times[ridge.frames[-1]] <= config.max_gap_s + spec.time_step:
                still_active.append(ridge)
            else:
                if len(ridge.frames) >= 3:
                    finished.append(ridge)
        active = still_active
        peaks, _ = find_peaks(row, height=config.peak_db, distance=2)
        frequencies = spec.frequencies[peaks].copy()
        # Parabolic interpolation reduces FFT-bin stair steps; not extra resolution.
        for j, peak in enumerate(peaks):
            a, b, c = smooth[frame, peak - 1 : peak + 2]
            denom = a - 2 * b + c
            if denom < -1e-6:
                frequencies[j] += float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5)) * df
        links = []
        for i, ridge in enumerate(active):
            dt = now - spec.times[ridge.frames[-1]]
            velocity = 0.0
            if len(ridge.frames) >= 5:
                velocity = (ridge.frequencies[-1] - ridge.frequencies[-5]) / (
                    spec.times[ridge.frames[-1]] - spec.times[ridge.frames[-5]]
                )
                velocity = float(np.clip(velocity, -config.max_rate_hz_s, config.max_rate_hz_s))
            prediction = ridge.frequencies[-1] + velocity * dt
            for j, frequency in enumerate(frequencies):
                distance = abs(frequency - ridge.frequencies[-1])
                if distance <= config.max_rate_hz_s * dt + 2.25 * df:
                    cost = abs(frequency - prediction) / df + 0.5 * dt / spec.time_step
                    links.append((cost, i, j))
        if links:
            unmatched_cost = 1e6
            costs = np.full((len(active), len(peaks) + len(active)), unmatched_cost)
            costs[:, : len(peaks)] = 1e12
            for cost, i, j in links:
                costs[i, j] = min(costs[i, j], cost)
            rows, columns = linear_sum_assignment(costs)
        else:
            rows, columns = (), ()
        used_peaks = set()
        for i, j in zip(rows, columns):
            if j >= len(peaks) or costs[i, j] >= unmatched_cost:
                continue
            active[i].frames.append(frame)
            active[i].frequencies.append(float(frequencies[j]))
            active[i].contrasts.append(float(row[peaks[j]]))
            used_peaks.add(j)
        for j, peak in enumerate(peaks):
            if j not in used_peaks:
                active.append(Ridge([frame], [float(frequencies[j])], [float(row[peak])]))
    finished.extend(ridge for ridge in active if len(ridge.frames) >= 3)
    return finished


def score_ridge(
    times: np.ndarray,
    frequencies: np.ndarray,
    contrasts: np.ndarray,
    time_step: float,
    frequency_step: float,
    config: DetectorConfig,
) -> Candidate | None:
    """Return a strict curved event or a separate descending-tone discovery.

    Scores rank survivors and are not probabilities or source identification.
    """
    if len(times) < 6:
        return None
    duration = float(np.ptp(times))
    if not config.min_duration_s <= duration <= config.max_duration_s + time_step:
        return None
    n_edge = max(3, len(times) // 10)
    drop = float(np.median(frequencies[:n_edge]) - np.median(frequencies[-n_edge:]))
    coverage = min(1.0, len(times) / (duration / time_step + 1))
    if drop < config.min_drop_hz or coverage < config.min_coverage:
        return None
    x = (times - times[0]) / duration
    y = frequencies - frequencies.mean()
    centered_x = x - x.mean()
    linear = frequencies.mean() + centered_x * (centered_x @ y) / (centered_x @ centered_x)
    linear_rmse = float(np.sqrt(np.mean((linear - frequencies) ** 2)))
    # A small fixed template bank, rather than an unstable nonlinear optimizer.
    templates = np.array(
        [
            -np.tanh((x - center) / width)
            for center in (0.3, 0.4, 0.5, 0.6, 0.7)
            for width in (0.10, 0.15, 0.22, 0.32, 0.45)
        ]
    )
    templates -= templates.mean(axis=1, keepdims=True)
    amplitude = (templates @ y) / np.sum(templates**2, axis=1)
    fits = frequencies.mean() + np.maximum(amplitude, 0)[:, None] * templates
    errors = np.sqrt(np.mean((fits - frequencies) ** 2, axis=1))
    best = int(np.argmin(errors))
    error = float(errors[best] / drop)
    raw_gain = float(1 - errors[best] / max(linear_rmse, 1e-12))
    gain = float(np.clip(raw_gain, 0, 1))
    # Compare samples about two seconds apart, tolerating half a frequency bin.
    lag = min(len(times) - 1, max(1, round(2 / time_step)))
    changes = frequencies[lag:] - frequencies[:-lag]
    monotonicity = float(np.mean(changes <= frequency_step / 2))
    movement = float(np.mean(changes <= -frequency_step / 2))
    shape = max(0.0, 1 - error / 0.16)
    strength = float(np.clip((np.median(contrasts) - config.peak_db) / 9, 0, 1))
    common = (
        times,
        frequencies,
        contrasts,
        coverage,
        drop,
        monotonicity,
        error,
        raw_gain,
        movement,
        linear_rmse,
        errors[best],
        time_step,
    )
    curve_improvement = linear_rmse - errors[best]
    if (
        error <= 0.16
        and gain >= config.min_curve_gain
        and curve_improvement >= frequency_step / 2
        and monotonicity >= 0.75
    ):
        score = float(0.25 * coverage + 0.20 * monotonicity + 0.25 * shape + 0.20 * gain + 0.10 * strength)
        if score >= config.min_score:
            return Candidate(*common[:3], fits[best], score, *common[3:], STRICT_CURVE)
    if movement < 0.10 or monotonicity < 0.75:
        return None
    score = float(0.35 * coverage + 0.30 * monotonicity + 0.25 * movement + 0.10 * strength)
    if score < config.min_score:
        return None
    return Candidate(*common[:3], linear, score, *common[3:], DESCENDING_DISCOVERY)


def detect(spec: Spectrogram, config: DetectorConfig | None = None) -> list[Candidate]:
    config = config or DetectorConfig()
    config.validate()
    ridges = extract_ridges(spec, config)
    df = float(spec.frequencies[1] - spec.frequencies[0])
    candidates = []
    # Search within long ridges too: stationary tails must not hide a sweep.
    durations = sorted(
        {config.min_duration_s, config.max_duration_s}
        | {d for d in (30.0, 45.0, 60.0, 90.0) if config.min_duration_s <= d <= config.max_duration_s}
    )
    for ridge in ridges:
        t = spec.times[ridge.frames]
        f, c = np.asarray(ridge.frequencies), np.asarray(ridge.contrasts)
        if np.ptp(t) < config.min_duration_s or np.ptp(f) < config.min_drop_hz:
            continue
        windows = {(0, len(t))} if np.ptp(t) <= config.max_duration_s else set()
        for duration in durations:
            for start in np.arange(t[0], t[-1] - duration + spec.time_step, 5.0):
                first = int(np.searchsorted(t, start))
                last = int(np.searchsorted(t, start + duration, side="right"))
                windows.add((first, last))
        for first, last in sorted(windows):
            candidate = score_ridge(t[first:last], f[first:last], c[first:last], spec.time_step, df, config)
            if candidate is not None:
                candidates.append(candidate)
    # Suppress duplicate subwindows only when they share time AND frequency.
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: (-c.score, c.times[0])):
        duplicate = False
        for other in kept:
            if candidate.category != other.category:
                continue
            lo, hi = max(candidate.times[0], other.times[0]), min(candidate.times[-1], other.times[-1])
            if hi <= lo:
                continue
            overlap = (hi - lo) / min(np.ptp(candidate.times), np.ptp(other.times))
            grid = np.linspace(lo, hi, 20)
            separation = np.median(
                np.abs(
                    np.interp(grid, candidate.times, candidate.frequencies)
                    - np.interp(grid, other.times, other.frequencies)
                )
            )
            if overlap > 0.4 and separation < 3 * df:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    kept.sort(key=lambda c: c.times[0])
    for i, candidate in enumerate(kept, 1):
        candidate.event_id = f"E{i:04d}"
    return kept
