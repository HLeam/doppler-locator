import numpy as np
import pytest

from doppler_locator.audio import spectrogram
from doppler_locator.detector import (
    DESCENDING_DISCOVERY,
    STRICT_CURVE,
    DetectorConfig,
    detect,
    extract_ridges,
    score_ridge,
)


def analyze(path, **kwargs):
    return spectrogram(path, n_fft=800, fmin=300, fmax=600, **kwargs)


@pytest.mark.parametrize("seed", [7, 19, 41])
def test_curved_positive_localizes_ridge(make_audio, seed):
    events = detect(analyze(make_audio(seed=seed)))
    assert len(events) == 1
    event = events[0]
    assert event.times[0] < 35 < event.times[-1]
    truth = 420 - 85 * np.tanh((event.times - 35) / 8)
    assert np.median(abs(truth - event.frequencies)) < 4
    assert event.drop_hz > 100
    assert 0.65 <= event.score <= 1
    assert event.curve_gain > 0.12
    assert event.category == STRICT_CURVE
    assert event.event_id == "E0001"


def test_non_template_curved_positive(make_audio):
    events = detect(analyze(make_audio("radial_curve")))
    assert len(events) == 1
    event = events[0]
    x = event.times - 35
    truth = 420 - 85 * x / np.sqrt(x**2 + 8**2)
    assert event.times[0] < 35 < event.times[-1]
    assert np.median(abs(truth - event.frequencies)) < 4


@pytest.mark.parametrize("kind", ["rising", "constant", "step", "noise", "silence"])
def test_negative_controls(make_audio, kind):
    assert detect(analyze(make_audio(kind))) == []


def test_descending_linear_tone_is_discovery_not_strict_event(make_audio):
    events = detect(analyze(make_audio("linear", noise=0.03)))
    assert events
    assert all(event.category == DESCENDING_DISCOVERY for event in events)
    assert all(event.curve_gain < DetectorConfig().min_curve_gain for event in events)
    assert all("strict_curvature_gate_not_met" in event.summary()["ambiguity_flags"] for event in events)


@pytest.mark.parametrize("kind", ["crossing_linear", "dual_descending"])
def test_independent_linear_tones_do_not_form_strict_events(make_audio, kind):
    events = detect(analyze(make_audio(kind, noise=0.03)))
    assert all(event.category != STRICT_CURVE for event in events)
    assert events
    assert all(event.category == DESCENDING_DISCOVERY for event in events)
    assert all("source_identity_unconfirmed" in event.summary()["ambiguity_flags"] for event in events)


@pytest.mark.parametrize("seed", [1, 23, 47])
def test_noise_seeds(make_audio, seed):
    assert detect(analyze(make_audio("noise", seed=seed))) == []


def test_stereo_power_does_not_cancel_positive(make_audio):
    path = make_audio(stereo=True)
    stereo = analyze(path)
    left = analyze(path, channel="left")
    np.testing.assert_allclose(stereo.power_db, left.power_db, atol=1e-5)
    assert len(detect(stereo)) == 1


def test_chunk_boundary_is_not_an_event_boundary(make_audio):
    path = make_audio()
    small, large = analyze(path, chunk_seconds=7), analyze(path, chunk_seconds=90)
    np.testing.assert_array_equal(small.times, large.times)
    # FFT batching may change float32 roundoff, but not frames or evidence.
    np.testing.assert_allclose(small.power_db, large.power_db, atol=1e-5)
    a, b = detect(small), detect(large)
    assert len(a) == len(b) == 1
    np.testing.assert_allclose(a[0].frequencies, b[0].frequencies)
    assert a[0].score == pytest.approx(b[0].score)


def test_sparse_and_small_excursion_rejected():
    config = DetectorConfig()
    t = np.arange(0, 60, 0.5)
    f = 420 - 85 * np.tanh((t - 30) / 8)
    assert score_ridge(t, f, np.full_like(t, 7), 0.5, 5, config) is not None
    assert score_ridge(t[::3], f[::3], np.full_like(t[::3], 7), 0.5, 5, config) is None
    assert score_ridge(t, 420 + (f - 420) / 10, np.full_like(t, 7), 0.5, 5, config) is None


@pytest.mark.parametrize("phase", [0.0, 0.17, 0.39])
def test_short_coherent_sweep_survives_crop_phase(phase):
    config = DetectorConfig()
    t = np.arange(phase, 18.56, 0.5)
    f = 420 - 85 * np.tanh((t - 9.28) / 3.0)
    candidate = score_ridge(t, f, np.full_like(t, 8), 0.5, 5, config)
    assert candidate is not None
    summary = candidate.summary()
    assert summary["category"] == STRICT_CURVE
    assert summary["unobserved_intervals"] == []
    assert summary["observed_point_count"] == len(t)
    assert "unobserved_intervals" not in summary["ambiguity_flags"]
    assert candidate.times[-1] - candidate.times[0] >= 15


@pytest.mark.parametrize("kind", ["genuine_gap", "sequential_handoff"])
def test_detect_preserves_nonoverlapping_segments_without_claiming_continuity(make_audio, kind):
    spec = analyze(make_audio(kind, noise=0.01))
    events = detect(spec)
    assert events
    assert all(event.category == DESCENDING_DISCOVERY for event in events)
    assert any(event.times[-1] < 24 for event in events)
    assert any(event.times[0] > 24 for event in events)
    assert all(not (event.times[0] < 24 and event.times[-1] > 24) for event in events)
    assert all("source_identity_unconfirmed" in event.summary()["ambiguity_flags"] for event in events)


@pytest.mark.parametrize("kind", ["continuous_sweep", "seamless_handoff"])
def test_extracted_tracker_links_remain_explicitly_unverified(make_audio, kind):
    spec = analyze(make_audio(kind, noise=0.03))
    ridges = extract_ridges(spec, DetectorConfig())
    assert any(spec.times[ridge.frames[0]] < 35 < spec.times[ridge.frames[-1]] for ridge in ridges)

    events = detect(spec)
    crossing = [event for event in events if event.times[0] < 35 < event.times[-1]]
    assert crossing
    event = max(crossing, key=lambda candidate: candidate.score)
    summary = event.summary()
    assert summary["unverified_source_continuity_link_count"] == len(event.times) - 1
    assert "tracker_links_source_continuity_unverified" in summary["ambiguity_flags"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"peak_db": 0},
        {"min_score": 2},
        {"max_gap_s": -1},
        {"min_drop_hz": float("nan")},
        {"max_duration_s": 1},
    ],
)
def test_bad_detector_settings(kwargs):
    with pytest.raises(ValueError):
        DetectorConfig(**kwargs).validate()
