import numpy as np
import pytest

from doppler_locator.audio import spectrogram
from doppler_locator.detector import DetectorConfig, detect, score_ridge


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
    assert event.event_id == "E0001"


def test_non_template_curved_positive(make_audio):
    events = detect(analyze(make_audio("radial_curve")))
    assert len(events) == 1
    event = events[0]
    x = event.times - 35
    truth = 420 - 85 * x / np.sqrt(x**2 + 8**2)
    assert event.times[0] < 35 < event.times[-1]
    assert np.median(abs(truth - event.frequencies)) < 4


@pytest.mark.parametrize("kind", ["rising", "linear", "constant", "step", "noise", "silence"])
def test_negative_controls(make_audio, kind):
    assert detect(analyze(make_audio(kind))) == []


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
