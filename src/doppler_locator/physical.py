"""Bounded acoustic model consistency, not source identification.

See docs/physical-fitting.md for the retarded-time derivation and the deliberately
non-statistical profile envelope. Frequencies are compared at RECEIVE times.
"""

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FitConfig:
    sound_speed_m_s: float = 343.0  # assumption, not measured weather
    frequency_floor_hz: float = 2.0  # tolerance, NOT a measured standard deviation
    f0_min_hz: float = 5.0
    f0_max_hz: float = 2000.0
    beta_min: float = 0.01
    beta_max: float = 0.9
    tau_min_s: float = 0.5
    tau_max_s: float = 1000.0
    cpa_margin_spans: float = 3.0
    profile_points: int = 17
    max_nfev: int = 400

    def validate(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("physical fit settings must be finite")
        if self.sound_speed_m_s <= 0 or self.frequency_floor_hz <= 0:
            raise ValueError("sound speed and frequency tolerance must be positive")
        if not 0 < self.f0_min_hz < self.f0_max_hz:
            raise ValueError("require 0 < f0_min_hz < f0_max_hz")
        if not 0 < self.beta_min < self.beta_max < 1:
            raise ValueError("require 0 < beta_min < beta_max < 1 (subsonic)")
        if not 0 < self.tau_min_s < self.tau_max_s or self.cpa_margin_spans <= 0:
            raise ValueError("positive ordered time-scale bounds and CPA margin required")
        if self.profile_points < 5 or self.max_nfev < 20:
            raise ValueError("profile_points >= 5 and max_nfev >= 20 required")


def received_frequency(times, f0_hz, beta, tau_s, cpa_receive_s):
    """f0/(1+R_dot/c); tau=b/v; CPA receive time = t0 + b/c.

    The explicit inverse of t_receive=t_emit+R(t_emit)/c avoids a numerical
    propagation root solver. The equivalent derivative form is stable in tails.
    """
    t0 = cpa_receive_s - beta * tau_s
    z = np.asarray(times, dtype=float) - t0
    width = tau_s * np.sqrt(1 - beta**2)
    return f0_hz / (1 - beta**2) * (1 - beta * z / np.hypot(z, width))


def parameters(values, sound_speed):
    f0, beta, tau, arrival = map(float, values)
    return {
        "emitted_reference_hz": f0,
        "speed_ratio_v_over_c": beta,
        "range_over_speed_s": tau,
        "cpa_receive_s": arrival,
        "cpa_emission_s": arrival - beta * tau,
        "sound_delay_at_cpa_s": beta * tau,
        "speed_m_s": beta * sound_speed,
        "cpa_slant_distance_m": beta * sound_speed * tau,
        "model_inflection_receive_s": arrival - beta * tau,
    }


def fit_observations(times, frequencies, config=None):
    """Fit only supplied observations; never insert samples across a gap.

    Deterministic multi-start plus one-dimensional nuisance-optimized profiles
    of EVERY fitted parameter. Profile solutions within best MSE + tolerance^2
    form a sensitivity set, not confidence intervals. Export all searches so
    bounds, local alternatives and failed optimization remain inspectable.
    """
    config = config or FitConfig()
    config.validate()
    t, f = np.asarray(times, float), np.asarray(frequencies, float)
    if (
        t.ndim != 1
        or f.shape != t.shape
        or len(t) < 6
        or not np.all(np.isfinite(t))
        or not np.all(np.isfinite(f))
        or np.any(np.diff(t) <= 0)
        or np.any(f <= 0)
    ):
        raise ValueError("need >=6 finite positive-frequency observations at strictly increasing times")
    origin, span = float(np.mean(t)), float(np.ptp(t))
    # Seconds centered/scaled for optimizer conditioning; logarithms for scales.
    lower = np.array(
        [
            np.log(config.f0_min_hz),
            config.beta_min,
            np.log(config.tau_min_s),
            (t[0] - origin) / span - config.cpa_margin_spans,
        ]
    )
    upper = np.array(
        [
            np.log(config.f0_max_hz),
            config.beta_max,
            np.log(config.tau_max_s),
            (t[-1] - origin) / span + config.cpa_margin_spans,
        ]
    )

    def decode(x):
        return np.array([np.exp(x[0]), x[1], np.exp(x[2]), origin + span * x[3]])

    def residual(x):
        return (received_frequency(t, *decode(x)) - f) / config.frequency_floor_hz

    searches = []

    def solve(start, fixed=None):
        base = np.clip(np.array(start), lower + 1e-9, upper - 1e-9)
        free = np.ones(4, dtype=bool)
        if fixed is not None:
            index, value = fixed
            base[index] = value
            free[index] = False

        def expand(x):
            result = base.copy()
            result[free] = x
            return result

        result = least_squares(
            lambda x: residual(expand(x)),
            base[free],
            bounds=(lower[free], upper[free]),
            max_nfev=config.max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        x = expand(result.x)
        mse = float(np.mean((residual(x) * config.frequency_floor_hz) ** 2))
        searches.append((x, mse, bool(result.success), fixed))
        return x, mse

    for beta, fraction, tau_fraction in product((0.08, 0.35, 0.75), (-0.5, 0.5, 1.5), (0.15, 1.0)):
        solve([np.log(np.median(f)), beta, np.log(span * tau_fraction), fraction])
    best_x, best_mse, _, _ = min(searches, key=lambda row: row[1])
    # Profile grids include bounds AND the optimum; two starts per grid value.
    # These finite grids can under-sample a valley, so no probabilistic CI claim.
    for index in range(4):
        grid = np.unique(np.r_[np.linspace(lower[index], upper[index], config.profile_points), best_x[index]])
        profile = []
        for value in grid:
            alternatives = [solve(best_x, (index, value)), solve((lower + upper) / 2, (index, value))]
            profile.append(min(alternatives, key=lambda row: row[1]))
        cutoff = min(row[1] for row in searches) + config.frequency_floor_hz**2
        # Refine each sampled threshold crossing: a coarse grid must not report
        # a zero-width envelope merely because only the optimum was accepted.
        for left, right in zip(profile[:-1], profile[1:]):
            if (left[1] <= cutoff) == (right[1] <= cutoff):
                continue
            for _ in range(8):
                value = (left[0][index] + right[0][index]) / 2
                middle = solve(left[0] if left[1] < right[1] else right[0], (index, value))
                if (middle[1] <= cutoff) == (left[1] <= cutoff):
                    left = middle
                else:
                    right = middle
    best_x, best_mse, best_success, _ = min(searches, key=lambda row: row[1])
    threshold = best_mse + config.frequency_floor_hz**2
    accepted = [row for row in searches if row[1] <= threshold]
    best = parameters(decode(best_x), config.sound_speed_m_s)
    accepted_params = [parameters(decode(row[0]), config.sound_speed_m_s) for row in accepted]
    envelopes = {
        key: [min(p[key] for p in accepted_params), max(p[key] for p in accepted_params)] for key in best
    }
    names = ["emitted_reference_hz", "speed_ratio_v_over_c", "range_over_speed_s", "cpa_receive_s"]

    def hits(x):
        return [
            names[i] + (":lower" if abs(x[i] - lower[i]) <= 1e-4 * (upper[i] - lower[i]) else ":upper")
            for i in range(4)
            if min(abs(x[i] - lower[i]), abs(x[i] - upper[i])) <= 1e-4 * (upper[i] - lower[i])
        ]

    bound_hits = sorted({hit for row in accepted for hit in hits(row[0])})
    predicted = received_frequency(t, *decode(best_x))
    line = np.polyval(np.polyfit(t - origin, f, 1), t - origin)
    constant_rmse = float(np.std(f))
    linear_rmse = float(np.sqrt(np.mean((f - line) ** 2)))
    rmse = float(np.sqrt(best_mse))
    # Conservative descriptive comparisons; correlated FFT samples preclude IID AIC/CI claims.
    distinguishes_line = linear_rmse - rmse >= config.frequency_floor_hz
    model_consistent = rmse <= 2 * config.frequency_floor_hz
    arrival_width = float(np.ptp(envelopes["cpa_receive_s"]))
    emission_width = float(np.ptp(envelopes["cpa_emission_s"]))
    reasons = []
    if not best_success:
        reasons.append("best_optimizer_not_converged")
    if not model_consistent:
        reasons.append("residual_exceeds_two_frequency_tolerances")
    if not distinguishes_line:
        reasons.append("linear_alternative_not_distinguished")
    if bound_hits:
        reasons.append("comparable_solutions_reach_parameter_bounds")
    if arrival_width > 0.25 * span:
        reasons.append("receive_CPA_sensitivity_exceeds_quarter_observed_span")
    if emission_width > 0.25 * span:
        reasons.append("emission_CPA_sensitivity_exceeds_quarter_observed_span")
    status = "conditionally_constrained" if not reasons else "insufficient_evidence"
    records = []
    for x, mse, success, fixed in searches:
        records.append(
            {
                "parameters": parameters(decode(x), config.sound_speed_m_s),
                "rmse_hz": float(np.sqrt(mse)),
                "converged": success,
                "comparable": mse <= threshold,
                "bounds_hits": hits(x),
                "profile_parameter": None if fixed is None else names[fixed[0]],
            }
        )
    return {
        "status": status,
        "timing_claim": "conditional model feature only; association/source origin unverified",
        "reasons": reasons,
        "config": asdict(config),
        "best_parameters": best,
        "supported_receive_span_s": [float(t[0]), float(t[-1])],
        "observed_point_count": len(t),
        "constant_rmse_hz": constant_rmse,
        "linear_rmse_hz": linear_rmse,
        "physical_rmse_hz": rmse,
        "physical_max_absolute_residual_hz": float(np.max(np.abs(f - predicted))),
        "linear_improvement_hz": linear_rmse - rmse,
        "distinguishes_linear_at_tolerance": distinguishes_line,
        "model_consistent_at_tolerance": model_consistent,
        "sensitivity_envelopes": envelopes,
        "envelope_definition": "finite multistart/profile set with MSE <= best MSE + frequency_floor_hz^2; NOT confidence intervals",
        "comparable_bounds_hits": bound_hits,
        "best_bounds_hits": hits(best_x),
        "search_bounds": {
            name: [float(decode(lower)[i]), float(decode(upper)[i])] for i, name in enumerate(names)
        },
        "search_count": len(records),
        "nonconverged_search_count": sum(not row[2] for row in searches),
        "searches": records,
        "predicted_hz": predicted.tolist(),
        "residual_hz": (f - predicted).tolist(),
        "linear_predicted_hz": line.tolist(),
    }


def prediction_envelope(times, result):
    """Envelope of comparable model predictions, including extrapolation in gaps."""
    curves = []
    for record in result["searches"]:
        if record["comparable"]:
            p = record["parameters"]
            curves.append(
                received_frequency(
                    times,
                    p["emitted_reference_hz"],
                    p["speed_ratio_v_over_c"],
                    p["range_over_speed_s"],
                    p["cpa_receive_s"],
                )
            )
    return np.min(curves, axis=0), np.max(curves, axis=0)


def cpa_separation(first, second, feature="cpa_receive_s"):
    """Only compare the SAME conditional feature when both fits constrain it."""
    if feature not in {"cpa_receive_s", "cpa_emission_s"}:
        raise ValueError("feature must be cpa_receive_s or cpa_emission_s")
    if any(r["status"] != "conditionally_constrained" for r in (first, second)):
        return {"status": "insufficient_evidence", "feature": feature, "separation_s": None}
    a, b = (r["sensitivity_envelopes"][feature] for r in (first, second))
    return {
        "status": "conditional_model_comparison_not_source_identity",
        "feature": feature,
        "separation_s": second["best_parameters"][feature] - first["best_parameters"][feature],
        "sensitivity_envelope_s": [b[0] - a[1], b[1] - a[0]],
        "warning": "finite sensitivity envelope, not confidence interval",
    }
