"""Truth is generated in EMISSION time, independently of the inverse fit model."""

import numpy as np
import pytest

from doppler_locator.physical import (
    FitConfig,
    cpa_separation,
    fit_observations,
    received_frequency,
)
from doppler_locator.physical_output import harmonic_relations, load_diagnostic_points, support


def forward(emission, f0=140, v=80, b=1600, c=343, t0=1000):
    x = v * (emission - t0)
    distance = np.sqrt(b**2 + x**2)
    radial_speed = v * x / distance
    return emission + distance / c, f0 / (1 + radial_speed / c)


@pytest.mark.parametrize("v,b,c", [(80, 1600, 343), (250, 15000, 330), (10, 100, 350)])
def test_retarded_inverse_sign_units_and_delay(v, b, c):
    emission = np.linspace(800, 1200, 501)
    receive, frequency = forward(emission, v=v, b=b, c=c)
    actual = received_frequency(receive, 140, v / c, b / v, 1000 + b / c)
    np.testing.assert_allclose(actual, frequency, rtol=1e-13)
    assert frequency[0] > 140 > frequency[-1]
    assert receive[250] == pytest.approx(1000 + b / c)
    assert actual[250] == pytest.approx(140)
    # Independent finite-difference phase derivative verifies received frequency.
    h = 1e-3
    before, _ = forward(emission - h, v=v, b=b, c=c)
    after, _ = forward(emission + h, v=v, b=b, c=c)
    np.testing.assert_allclose(140 * 2 * h / (after - before), frequency, rtol=1e-7)


def test_model_inflection_is_not_arrival_of_cpa_sound():
    t = np.linspace(950, 1050, 10001)
    f = received_frequency(t, 140, 0.4, 20, 1008)
    steepest = t[np.argmin(np.gradient(f, t))]
    assert steepest == pytest.approx(1000, abs=0.01)
    assert steepest != pytest.approx(1008)


@pytest.mark.parametrize("phase,noise", [(0, 0), (0.17, 0.3), (0.39, 1.0)])
def test_recovery_and_honest_sensitivity(phase, noise):
    e = np.arange(800 + phase, 1200, 2)
    t, f = forward(e)
    f += np.random.default_rng(19).normal(0, noise, len(f))
    result = fit_observations(t, f, FitConfig(profile_points=7))
    assert result["status"] == "conditionally_constrained"
    p = result["best_parameters"]
    assert p["emitted_reference_hz"] == pytest.approx(140, abs=1)
    assert p["speed_m_s"] == pytest.approx(80, abs=2)
    assert p["cpa_slant_distance_m"] == pytest.approx(1600, abs=150)
    assert p["cpa_emission_s"] == pytest.approx(1000, abs=1)
    assert p["cpa_receive_s"] == pytest.approx(1000 + 1600 / 343, abs=1)
    for key, truth in [
        ("cpa_emission_s", 1000),
        ("cpa_receive_s", 1000 + 1600 / 343),
        ("speed_m_s", 80),
        ("cpa_slant_distance_m", 1600),
    ]:
        lo, hi = result["sensitivity_envelopes"][key]
        assert lo < truth < hi  # nonzero, not an inverse-Hessian CI
    assert result["search_count"] > 100


@pytest.mark.parametrize("kind", ["constant", "linear", "short", "far_tail"])
def test_partial_curves_and_nulls_do_not_force_cpa(kind):
    t = np.linspace(0, 30, 61)
    if kind == "constant":
        f = np.full_like(t, 140)
    elif kind == "linear":
        f = 170 - t
    else:
        e = np.linspace(995, 1005, 41) if kind == "short" else np.linspace(1100, 1130, 61)
        t, f = forward(e)
    result = fit_observations(t, f, FitConfig(profile_points=5))
    assert result["status"] == "insufficient_evidence"
    assert result["reasons"]
    assert cpa_separation(result, result)["separation_s"] is None
    assert len(result["predicted_hz"]) == len(t)


def test_speed_of_sound_scales_distance_not_time_or_fit():
    t, f = forward(np.linspace(900, 1100, 101))
    a = fit_observations(t, f, FitConfig(profile_points=5, sound_speed_m_s=343))
    b = fit_observations(t, f, FitConfig(profile_points=5, sound_speed_m_s=300))
    for key in ("cpa_receive_s", "cpa_emission_s", "emitted_reference_hz", "range_over_speed_s"):
        assert a["best_parameters"][key] == b["best_parameters"][key]
    assert b["best_parameters"]["cpa_slant_distance_m"] / a["best_parameters"][
        "cpa_slant_distance_m"
    ] == pytest.approx(300 / 343)


def track(name, t, f):
    return {
        "event_id": name,
        "time_step_s": 0.5,
        "points": [{"time_s": float(ti), "frequency_hz": float(fi)} for ti, fi in zip(t, f)],
    }


def test_harmonics_are_uncertain_pair_evidence_not_independent_sources():
    t = np.arange(0, 30, 0.5)
    f = 160 - t
    tracks = [track("low", t, f), track("twice", t, 2 * f), track("unrelated", t, 210 - t)]
    pairs = harmonic_relations(tracks)
    assert len(pairs) == 1
    assert pairs[0]["order_ratio"] == 2
    assert pairs[0]["grouping_uncertain"] is True
    assert pairs[0]["paired_observed_points"] == len(t)
    # Nonconcurrent and gappy data cannot gain harmonic support by interpolation.
    assert harmonic_relations([tracks[0], track("missing", t + 40, 2 * f)]) == []
    assert harmonic_relations([track("a", t[::4], f[::4]), track("b", t[1::4], 2 * f[1::4])]) == []


def test_gap_export_and_external_selection_provenance(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text("event_id,time_s,frequency_hz,competing_hz\nA,1,140,150\nA,1.5,139,151\nA,3,138,152\n")
    tracks = load_diagnostic_points(path, 0.5)
    assert tracks[0]["selection"] == "externally_selected_not_blind_recovery"
    assert tracks[0]["points"][0]["competing_hz"] == "150"
    assert support(tracks[0])["gaps"] == [{"after_s": 1.5, "before_s": 3.0, "unobserved_s": 1.0}]
    assert support(tracks[0])["observed_support_s"] == 1.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sound_speed_m_s": 0},
        {"beta_max": 1},
        {"frequency_floor_hz": float("nan")},
        {"profile_points": 1},
        {"tau_min_s": -1},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        FitConfig(**kwargs).validate()


@pytest.mark.parametrize(
    "t,f", [([0] * 6, [140] * 6), (list(range(6)), [140] * 5), (list(range(6)), [140] * 5 + [float("nan")])]
)
def test_invalid_observations(t, f):
    with pytest.raises(ValueError):
        fit_observations(t, f)


def test_separation_has_named_feature_and_rejects_mixed_definitions():
    t, f = forward(np.linspace(800, 1200, 101))
    a = fit_observations(t, f, FitConfig(profile_points=5))
    b = fit_observations(t + 900, f, FitConfig(profile_points=5))
    assert cpa_separation(a, b)["separation_s"] == pytest.approx(900)
    with pytest.raises(ValueError):
        cpa_separation(a, b, "midpoint")
