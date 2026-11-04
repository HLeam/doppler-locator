"""Small deterministic fixtures generated locally; no source recording required."""

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def make_audio(tmp_path):
    def make(kind="curve", seed=7, duration=70, fs=4000, stereo=False, noise=0.12):
        t = np.arange(round(duration * fs)) / fs
        midpoint = duration / 2
        rng = np.random.default_rng(seed)
        if kind == "crossing_linear":
            frequencies = (350 + 160 * t / duration, 510 - 160 * t / duration)
        elif kind == "dual_descending":
            frequencies = (510 - 110 * t / duration, 490 - 80 * t / duration)
        else:
            frequencies = ({
                "curve": 420 - 85 * np.tanh((t - midpoint) / 8),
                "radial_curve": 420 - 85 * (t - midpoint) / np.sqrt((t - midpoint) ** 2 + 8**2),
                "rising": 420 + 85 * np.tanh((t - midpoint) / 8),
                "linear": 505 - 170 * t / duration,
                "constant": np.full_like(t, 420),
                "step": np.where(t < midpoint, 505, 335),
                "noise": np.zeros_like(t),
                "silence": np.zeros_like(t),
            }[kind],)
        samples = sum(
            0.15 * np.sin(2 * np.pi * np.cumsum(frequency) / fs) for frequency in frequencies
        ) + noise * rng.normal(size=len(t))
        if kind == "silence":
            samples[:] = 0
        if kind == "curve":
            # A stationary, non-harmonic interferer must not become a second event.
            samples += 0.08 * np.sin(2 * np.pi * 570 * t)
        if stereo:
            samples = np.column_stack((samples, -samples))
        path = tmp_path / f"{kind}-{seed}-{fs}-{duration}-{stereo}.wav"
        sf.write(path, samples, fs, subtype="FLOAT")
        return path

    return make
