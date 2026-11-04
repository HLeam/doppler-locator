import os
from pathlib import Path

import numpy as np
import pytest

from doppler_locator.audio import spectrogram
from doppler_locator.detector import DESCENDING_DISCOVERY, STRICT_CURVE, detect

REFERENCE = os.environ.get("DOPPLER_REFERENCE_WAV")
pytestmark = pytest.mark.skipif(not REFERENCE, reason="set DOPPLER_REFERENCE_WAV for local reference checks")


@pytest.mark.parametrize(
    ("target", "category", "contexts", "frequency_ranges"),
    [
        (
            (1230, 1290),
            STRICT_CURVE,
            [(1200, 1320), (900, 1500), (0, None)],
            (400, 540, 320, 420),
        ),
        (
            (2100, 2175),
            DESCENDING_DISCOVERY,
            [(2070, 2205), (1800, 2400), (0, None)],
            (380, 520, 320, 430),
        ),
    ],
)
def test_reference_categories_and_frequency_paths(target, category, contexts, frequency_ranges):
    matched = []
    counts = []
    for start, end in contexts:
        spec = spectrogram(
            Path(REFERENCE),
            start=start,
            end=end,
            fmin=300,
            fmax=600,
            n_fft=8024,
            hop=4012,
            time_step=0.5015,
        )
        candidates = detect(spec)
        counts.append(len(candidates))
        eligible = []
        for candidate in candidates:
            if candidate.category != category:
                continue
            overlap = min(candidate.times[-1], target[1]) - max(candidate.times[0], target[0])
            if overlap < 10:
                continue
            edge = max(3, len(candidate.times) // 10)
            start_hz = float(np.median(candidate.frequencies[:edge]))
            end_hz = float(np.median(candidate.frequencies[-edge:]))
            if (
                frequency_ranges[0] <= start_hz <= frequency_ranges[1]
                and frequency_ranges[2] <= end_hz <= frequency_ranges[3]
            ):
                eligible.append((candidate, overlap, start_hz, end_hz))
        assert eligible, (category, start, counts[-1], [c.summary() for c in candidates])
        candidate, overlap, start_hz, end_hz = max(eligible, key=lambda item: item[1])
        assert overlap / (target[1] - target[0]) > 0.1
        summary = candidate.summary()
        assert summary["observed_support_s"] <= candidate.times[-1] - candidate.times[0] + spec.time_step
        matched.append((candidate, start_hz, end_hz))

    for left, right in zip(matched, matched[1:]):
        a, b = left[0], right[0]
        lo, hi = max(a.times[0], b.times[0]), min(a.times[-1], b.times[-1])
        assert hi > lo
        grid = np.linspace(lo, hi, 20)
        deviation = np.abs(np.interp(grid, a.times, a.frequencies) - np.interp(grid, b.times, b.frequencies))
        assert float(np.max(deviation)) <= 6.0
        assert left[1] - left[2] >= 60
        assert right[1] - right[2] >= 60
