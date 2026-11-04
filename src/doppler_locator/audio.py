"""Native-rate, bounded-memory, stereo-power spectrograms."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann


@dataclass(frozen=True)
class Spectrogram:
    times: np.ndarray
    frequencies: np.ndarray
    power_db: np.ndarray  # time x frequency; relative digital power, not calibrated SPL
    sample_rate: int
    n_fft: int
    hop: int
    frames_per_bin: int

    @property
    def time_step(self) -> float:
        return self.hop * self.frames_per_bin / self.sample_rate


def spectrogram(
    path: str | Path,
    *,
    start: float = 0,
    end: float | None = None,
    fmin: float = 300,
    fmax: float = 600,
    n_fft: int = 8024,
    hop: int | None = None,
    time_step: float = 0.5,
    chunk_seconds: float = 30,
    channel: str = "power",
) -> Spectrogram:
    """Average consecutive Hann-STFT powers; never mix opposing stereo phases.

    FFT frame and averaging-bin boundaries are anchored at recording time zero,
    so overlapping crops use the same frames. Frames have no padding. Times
    are absolute window-center means, in seconds; incomplete bins are omitted.
    """
    hop = n_fft // 2 if hop is None else hop
    scalars = [start, fmin, fmax, time_step, chunk_seconds]
    if end is not None:
        scalars.append(end)
    if not np.all(np.isfinite(scalars)):
        raise ValueError("time and frequency settings must be finite")
    if n_fft < 16 or not 1 <= hop <= n_fft:
        raise ValueError("n_fft must be >=16; hop must be in [1, n_fft]")
    if start < 0 or time_step <= 0 or chunk_seconds <= 0:
        raise ValueError("start must be nonnegative; time_step/chunk_seconds positive")
    if channel not in {"power", "left", "right"}:
        raise ValueError("channel must be power, left or right")
    with sf.SoundFile(path) as audio:
        fs = audio.samplerate
        if not 0 <= fmin < fmax <= fs / 2:
            raise ValueError("require 0 <= fmin < fmax <= Nyquist frequency")
        stop = len(audio) / fs if end is None else end
        if stop <= start or stop > len(audio) / fs + 1 / fs:
            raise ValueError("require start < end <= recording duration")
        if channel == "right" and audio.channels < 2:
            raise ValueError("right channel requested from mono recording")
        first_sample = round(start * fs)
        last_sample = min(round(stop * fs), len(audio))
        group = max(1, round(time_step * fs / hop))
        first_bin = (first_sample + hop * group - 1) // (hop * group)
        first_group_sample = first_bin * hop * group
        last_frame = (last_sample - n_fft) // hop
        first_frame = first_bin * group
        n_bins = (last_frame - first_frame + 1) // group
        if n_bins < 1:
            raise ValueError("selected interval is shorter than one complete averaged FFT bin")
        n_frames = n_bins * group
        bins_per_chunk = max(1, int(chunk_seconds * fs / (hop * group)))
        batch_frames = bins_per_chunk * group
        freq = rfftfreq(n_fft, 1 / fs)
        band = (freq >= fmin) & (freq <= fmax)
        if np.count_nonzero(band) < 5:
            raise ValueError("frequency band must contain at least five FFT bins")
        window = hann(n_fft, sym=False).astype(np.float32)
        powers, times = [], []
        for offset in range(0, n_frames, batch_frames):
            count = min(batch_frames, n_frames - offset)
            audio.seek(first_group_sample + offset * hop)
            samples = audio.read(n_fft + (count - 1) * hop, dtype="float32", always_2d=True)
            if not np.all(np.isfinite(samples)):
                raise ValueError("audio contains non-finite samples")
            if channel != "power":
                index = 0 if channel == "left" else 1
                samples = samples[:, index : index + 1]
            frames = np.lib.stride_tricks.sliding_window_view(samples, n_fft, axis=0)[::hop]
            spectrum = rfft(frames * window, axis=-1)[..., band]
            power = np.mean(np.abs(spectrum / window.sum()) ** 2, axis=1)
            for local in range(0, count, group):
                size = min(group, count - local)
                powers.append(power[local : local + size].mean(axis=0))
                center = first_group_sample + (offset + local + (size - 1) / 2) * hop + n_fft / 2
                times.append(center / fs)
    db = 10 * np.log10(np.maximum(np.asarray(powers), 1e-20))
    return Spectrogram(np.asarray(times), freq[band], db, fs, n_fft, hop, group)
