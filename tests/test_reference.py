import os
from pathlib import Path

import numpy as np
import pytest

from doppler_locator.audio import spectrogram
from doppler_locator.detector import detect


REFERENCE = os.environ.get("DOPPLER_REFERENCE_WAV")
pytestmark = pytest.mark.skipif(not REFERENCE, reason="set DOPPLER_REFERENCE_WAV for local reference checks")


@pytest.mark.parametrize(
    ("start", "end", "target_start", "target_end"),
    [
        (1200, 1320, 1230, 1290),
        (900, 1500, 1230, 1290),
        (0, None, 1230, 1290),
        (2070, 2205, 2100, 2175),
        (1800, 2400, 2100, 2175),
        (0, None, 2100, 2175),
    ],
)
def test_reference_curves_survive_context_changes(start, end, target_start, target_end):
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
    matches = []
    for candidate in candidates:
        overlap = min(candidate.times[-1], target_end) - max(candidate.times[0], target_start)
        if overlap < 10:
            continue
        first = candidate.frequencies[: max(3, len(candidate.times) // 10)]
        last = candidate.frequencies[-max(3, len(candidate.times) // 10) :]
        start_hz, end_hz = float(np.median(first)), float(np.median(last))
        if start_hz >= 380 and end_hz <= 450 and start_hz - end_hz >= 60:
            matches.append(candidate)
    assert matches, [
        (
            round(float(candidate.times[0]), 2),
            round(float(candidate.times[-1]), 2),
            round(float(np.median(candidate.frequencies[: max(3, len(candidate.times) // 10)])), 1),
            round(float(np.median(candidate.frequencies[-max(3, len(candidate.times) // 10) :])), 1),
        )
        for candidate in candidates
    ]
