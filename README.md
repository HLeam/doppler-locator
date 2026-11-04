# Doppler Locator

A transparent Python baseline for **candidate descending Doppler-like audio curves**. It reads local audio, links frequency ridges, and writes reviewable CSVs and annotated spectrograms. It does **not** identify aircraft, estimate microphone location, or interpret its score as a probability.

**Current result:** the first supplied example appears as a strict curved event; the second appears as a partial, unconfirmed descending-tone discovery in padded, larger-context and whole-recording scans. These are development observations, not validated accuracy. See [evaluation and limitations](docs/evaluation.md).

## Install and test

Python 3.11+; NumPy, SciPy, SoundFile/libsndfile, Matplotlib. No GPU, model, network service, or credentials are needed at runtime. SoundFile wheels include libsndfile on common platforms; source installations may need a system libsndfile package.

Reproduce the locked environment with [uv](https://docs.astral.sh/uv/) (tested with uv 0.12.19):

```sh
uv sync --frozen --extra dev --python 3.12
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

Or install with standard pip (dependency ranges, not the exact lock):

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Tests synthesize small audio files in temporary directories; no reference media is needed in CI. `uv.lock` pins transitive packages and hashes. CI runs synthetic tests and lint on Python 3.11 and 3.12.

## Run locally

```sh
# Whole recording; use a new/empty output directory each run.
uv run --frozen doppler-locator --input recording.wav --output outputs/hour

# First development example, with context; FFT length is exactly 8024 samples.
uv run --frozen doppler-locator --input recording.wav --output outputs/example-1 \
  --start 20:00 --end 22:00 --fmin 300 --fmax 600 --n-fft 8024

# Second development example, with context.
uv run --frozen doppler-locator --input recording.wav --output outputs/example-2 \
  --start 34:30 --end 36:45

# Reproduce both development clips, a separate unlabeled interval, and the full scan.
uv run --frozen python scripts/evaluate_reference.py \
  --input recording.wav --output outputs/evaluation --full-hour

uv run --frozen doppler-locator --help
# Equivalent after installation: python -m doppler_locator ...
```

`--start`/`--end` accept seconds, `MM:SS`, or `HH:MM:SS`; all output times are seconds from the recording start. The end bounds audio reads (no padded FFT frames). Defaults target the supplied 500→330 Hz examples: 300–600 Hz, Hann FFT 8024, half-window hop, approximately 0.5 s power averages, 30 s IO batches, channel **power averaging**. `--channel left` / `right` selects a channel instead; power averaging avoids antiphase stereo cancellation. At 48 kHz, 8024 is a 0.16717 s window, Δf ≈ 5.982 Hz and the averaged step is 0.5015 s. Larger FFTs improve frequency spacing at the expense of time resolution; they are not interchangeable tuning settings.

`--hop`, `--time-step`, `--chunk-seconds`, all detector parameters below, and `--max-plots` are configurable. FFT frames and power averages align to the recording's time-zero grid; crop bounds select complete aligned averages, so overlapping crops use identical spectral evidence. Changing **only IO chunk size** also preserves the grid and results to floating-point tolerance. No sample-rate conversion occurs. WAV is recommended; compressed formats depend on libsndfile and seeking support. The validated detection default remains FFT 8024. FFT 32768 is a separate visual-review setting and must not be substituted as the detection FFT without its own validation.

Add `--review-fft 32768` to make a separate `visual-review-fft-32768.png` using the same detected paths over a higher-resolution spectrogram. Detection still uses `--n-fft` (8024 by default); the review plot and its `run.json` metadata are labeled separately.

### Output contract

- `candidates.csv`: category, observed start/end and support, point count, missing observation intervals, ambiguity flags, unverified source-continuity link count, edge medians, coverage, monotonicity, line/curve RMSE, relative and absolute curve improvement, and category-specific heuristic ranking. `strict_curved_event` passes the sigmoid-shape gates. `unconfirmed_descending_tone` has descending observed support but does not pass strict curvature; it is a review lead, not a complete source trajectory. A zero-detection CSV still has a header.
- `ridges.csv`: category, observed time/frequency/contrast points, comparison fit, point index, association method and per-point source-continuity uncertainty. `minimum_cost_peak_link` records a tracker association; it does not verify that the same physical source produced both peaks. No observations are invented inside gaps.
- `overview.png`: time/frequency axes, relative digital-power colorbar, cyan strict-event points and orange unconfirmed-tone points. Dotted white links show peak associations with unverified source continuity. Labels include observed support, gaps, curve gain and unverified-link count. Wide overviews average display pixels only; event detection uses the full reduced spectrogram.
- `E0001.png`, etc.: close-ups of up to `--max-plots` (default 20) highest-ranked events, labeled with category, support, gaps and curvature evidence. Green dashed sigmoid fits appear only for strict events. Event IDs remain chronological even when only the highest-ranked plots are written.
- `run.json`: source SHA-256, dimensions, configuration, category counts, exact dependency versions, timings and plotted IDs. No input pathname is embedded. Time excludes plot/output writing. IDs are local to each run, not stable identities across settings.

Output directories must be new or empty, to avoid overwriting source data or stale results. Plots use an offline headless renderer. No audio is downloaded or uploaded by these scripts.

## Method and heuristic score

1. Read overlapping native-rate audio batches. Compute Hann-window real FFT power, average across channels and adjacent frames, retain only the requested frequency band. Raw audio memory is bounded by the IO batch; the reduced time×frequency matrix is retained (memory still grows with duration/bandwidth).
2. Smooth log-power with a Gaussian of 1 time bin × 0.5 frequency bins. Subtract a frequency median background (`--background-hz 100`, rounded to an odd number of at least five bins). Extract local maxima at least two bins apart, exceeding `--peak-db 1.5` dB **local contrast**, not a calibrated SNR. Parabolic peak interpolation reduces bin stair steps, not the true resolution limit.
3. Link peaks to short-term velocity predictions with a minimum-cost one-to-one assignment across active ridges and peaks in each bin. A link permits `--max-rate-hz-s 12` times elapsed seconds **plus 2.25 FFT bins** of tolerance. `--max-gap-s 2` is the allowed missing span (last-to-next time may be one bin longer). Expired ridges remain separate; frequency and slope compatibility alone do not establish source continuity. Crossings can swap identities; this is not global multi-target tracking.
4. Score each tracked ridge and overlapping 15/30/45/60/90/120 s subwindows within `--min-duration-s 15` and `--max-duration-s 120` (configured endpoints included in the window bank), stepped every 5 s. Expired ridges are not concatenated based on compatible slope/frequency. Require at least six points, endpoint-median drop ≥ `--min-drop-hz 60`, and coverage ≥ `--min-coverage 0.65`. Endpoints use the first/last 10% of observations, at least three points. The strict category additionally requires curve error ≤0.16 of the drop, gain over a linear fit ≥`--min-curve-gain`, absolute RMSE improvement ≥half an FFT bin, monotonicity ≥0.75 and score ≥`--min-score`.
5. Compare a line with a bank of descending `offset - amplitude*tanh((x-center)/width)` templates, where normalized time `x` runs 0→1, centers are 0.3/0.4/0.5/0.6/0.7 and widths 0.10/0.15/0.22/0.32/0.45. Fit only offset and nonnegative amplitude by least squares. This is a **descriptive shape**, not an acoustic propagation or physical trajectory fit.

Let `C` be observed/expected time-bin coverage (capped at 1), `M` the fraction of observation comparisons roughly 2 s apart that do not rise by more than half an FFT bin (gaps lengthen the comparison), `E = shape_RMSE / endpoint_drop`, `G = clip(1 - shape_RMSE / max(line_RMSE, 1e-12), 0, 1)` and `A = clip((median_contrast - peak_db)/9, 0, 1)`. Require `M >= 0.75`, `E <= 0.16` and `G >= --min-curve-gain` (0.05). Rank survivors with:

```text
score = 0.25*C + 0.20*M + 0.25*max(0, 1-E/0.16) + 0.20*G + 0.10*A
```

For an observed descending segment that fails strict curvature, a separate discovery rank uses coverage, monotonicity, fraction of descending two-second changes and contrast. It still requires ≥0.10 descending movement fraction and score ≥`--min-score 0.65`; it is explicitly unconfirmed. Neither category fills missing points. Suppress overlapping subwindows when time overlap exceeds 40% of the shorter window and median frequency separation is <3 FFT bins; retain the higher-ranked result. The reported interval may therefore cover only the clearest **part** of a longer event.

All weights/thresholds are heuristic development choices. No class probabilities, precision guarantees, aircraft labels, speed, closest approach, or coordinates follow from them. Harmonics are **not required or scored**: inspection did not establish a consistent harmonic family for both examples. Other frequencies remain available via `--fmin`/`--fmax`, but broadening the band also changes competing peaks.

## Source, privacy, and scope

Reference: [Unknown location - raw audio](https://www.youtube.com/watch?v=4CxVxuMVkMc), approximately one hour. [Source attribution and reproducible conversion](docs/source.md) document the local reference without publishing it or any private absolute pathname. Analyze only media you own or have permission to use. Keep media and generated outputs in local ignored directories; never add raw audio/video or exact microphone coordinates to git.

QGIS is the intended desktop map workspace for any later geographic work, not a browser app. This implementation is **audio-curve detection only**: no flight APIs, DEM integration, map UI, source identification or microphone-location search. The repository does not assign a license to the recording or to the project.
