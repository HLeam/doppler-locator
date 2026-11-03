# Evaluation and limitations

## Plan and scope

The implementation sequence was: inspect local native-rate spectrograms; implement a chunked STFT and peak-linking baseline; test synthetic positives/controls; inspect the two supplied development clips and a separate interval; scan the hour; preserve and document shortcomings rather than chase perfect detections. No ML, external service, localization, or physical parameter fitting was used. The detailed initial plan and real-data evidence remain private/local.

The manual intervals (1230–1290 s and 2100–2175 s, approximately 500→330 Hz) influenced the default frequency band, event scale and contrast threshold. Initial local-contrast thresholds of 3 dB fragmented both target ridges; a small sweep through 2, 1.5 and 1 dB led to 1.5 dB. Suppression overlap was set to 40% to avoid duplicate partial windows. These are development choices, **not independently validated parameters**. No further tuning was applied to recover the whole-hour miss.

## Reproduction

```sh
uv sync --frozen --extra dev --python 3.12
uv run --frozen pytest -q
uv run --frozen python scripts/evaluate_reference.py \
  --input recording.wav --output outputs/evaluation --full-hour
```

Use the WAV identified in [source.md](source.md). The script runs the standard CLI without detector overrides: 300–600 Hz, FFT 8024, half-window hop, ~0.5 s power bins. It records temporal matches using **at least 10 seconds overlap** with each supplied interval. That is an explicit convenience check, not a segment-level ground-truth metric: inspect frequency ridges in the images too. The detector never receives the label times. Event IDs restart in each run.

## Initial baseline (before ridge correction)

| Analysis | Candidate interval | Endpoint medians | Heuristic score | Interpretation |
|---|---|---|---|---|
| 20:00–22:00 padded clip | 20:55.46–21:25.05 | 497→370 Hz | 0.838 | First example, clear central descent; early onset and tail toward 330 Hz not recovered |
| 34:30–36:45 padded clip | 35:25.96–35:54.54 | 447→348 Hz | 0.691 | Second example, partial descent; misses the weak higher-frequency portion and later tail |
| Whole recording | 20:55.55–21:25.14 | 497→368 Hz | 0.845 | Matches first development example |
| Whole recording | **No match at 35:00–36:15** | — | — | **Misses second development example** |
| Whole recording | 05:18.24–05:45.32 | 558→476 Hz | 0.698 | Unlabeled candidate, not a confirmed positive or false positive |
| Separate 10:00–13:00 interval | None | — | — | Zero candidates; no independent ground truth |

These are historical results from before the ridge correction below. There are no independently verified labels from which to estimate precision or a false-positive rate.

The whole-hour second-example miss came from crop-dependent frame/bin phase plus a broken ridge continuation. In continuous context, the main fragment ended at 376.6 Hz; the next visible continuation was 358.5 Hz, a jump just beyond the former link allowance. Frames and power averages now align to recording time zero, peak association uses a minimum-cost one-to-one assignment, and a descending ridge may continue through a compatible overlapping ridge. This keeps the shared evidence consistent between crops and full scans.

## Current detector: context comparison

The same defaults (FFT 8024, hop 4012, 0.5015 s power bins, 300–600 Hz) were used for padded clips, larger intervals, and the full recording. Frames and averaging bins align to recording time zero. No audio-derived output files were written.

| Example/context | Candidate count | Matching interval | Endpoint medians |
|---|---:|---|---|
| First, 1200–1320 s | 1 | 1255.55–1285.14 s | 497→368 Hz |
| First, 900–1500 s | 2 | 1255.55–1285.14 s | 497→368 Hz |
| First, whole recording | 5 | 1255.55–1285.14 s | 497→368 Hz |
| Second, 2070–2205 s | 0 | None | — |
| Second, 1800–2400 s | 0 | None | — |
| Second, whole recording | 5 | None | — |

The first example is detected in all contexts. The second remains unresolved: in the full scan, a descending fragment ends at 376.6 Hz while a separate ridge near 358.5 Hz is already being observed concurrently. Joining the latter's tail produced a plausible 445→348 Hz candidate, but the two observed ridges may be independent tones, so the current detector refuses that join. The remaining 18.56-second fragment is nearly linear and does not pass the current curve-gain score. A sound next step is multi-hypothesis tracking through ambiguous peak collisions, preserving both paths until later evidence distinguishes a continuation; the present one-ridge assignment cannot establish that relationship. Negative synthetic controls remain in the focused detector suite. These development examples are not independent recall or precision estimates, and scores are heuristic rankings rather than probabilities.

### Visual review (images kept local)

Opened and inspected the final annotated images:

- `development-1/E0001.png`: first supplied example; cyan ridge follows the visible main descent. Green dashed line is only the descriptive shape template.
- `development-2/E0001.png`: second supplied example; partial descending ridge amidst broadband interference and adjacent tones. It does not recover the entire supplied interval.
- `full-hour/E0001.png`: unlabeled 05:18–05:45 candidate, a visible tonal descent that becomes near-stationary. It is plausible curve evidence, but may be an unrelated changing-pitch source; its identity and truth status remain unknown.
- `separate-10-13min/overview.png`: a separately inspected 10:00–13:00 interval, with substantial broadband energy around 11:30 and stationary/narrowband components; no accepted 300–600 Hz curve. A wider preliminary view showed other-frequency structure outside the default band. This interval was **not relabeled as an independent negative** and was not used to select thresholds.

No harmonic family was consistently established for both examples in the wider preliminary views (200–1000 Hz). Harmonics therefore supply neither a requirement nor a score bonus.

## Synthetic and software checks

47 tests pass locally in each of the locked Python 3.11 and Python 3.12 environments. The primary synthetic signal cases are:

- Three seeded descending tanh curves (505→335 Hz approximately) in Gaussian noise plus a stationary non-harmonic interferer: each detected once, median ridge error <4 Hz.
- One descending radial-velocity-shaped `t/sqrt(t²+scale²)` curve, which is **not** the fitting template: detected once with median ridge error <4 Hz.
- Nine negative/control clip cases: ascending curve, linear drift, constant tone, abrupt frequency step, silence, and four noise seeds: zero detections in these generated cases.
- Additional stereo antiphase, sparse-ridge rejection, low-excursion rejection, FFT=8024/native-rate timing, chunk-boundary invariance, argument validation, NaN audio rejection, missing input, CSV/JSON/PNG output, and overwrite-refusal checks.

A regression during implementation caught a score bug: flooring the line residual at half an FFT bin falsely made a synthetic linear drift look curved. Gain now compares to the actual line residual (with only a numerical zero guard); the linear negative test prevents that regression. Synthetic signals remain much simpler and stronger than real recording conditions. Passing these tests does not validate real-world discrimination.

One local ARM64 macOS full-hour run took about 6 s wall time including plots and SHA-256, peaking at approximately 227 MiB RSS. The final evaluation recorded 4.52 s for full-hour analysis plus checksum, excluding plot writing; clip analyses were 0.51–0.63 s. These are machine/cache-specific observations, not benchmarks or guarantees. The retained reduced spectrogram scales with time and frequency bandwidth; the waveform/FFT batches do not require loading the hour into memory.

## Known limits and useful next experiments

- Real misses, fragmentary boundaries, crop sensitivity and crossing-track swaps already occur. A confidence score is a heuristic ranking, not calibrated uncertainty.
- Oscillator/engine pitch changes, instruments, machinery, vehicles, interference, and time-varying background can mimic a Doppler-like curve. Pure linear drifts are deliberately penalized, but some actual Doppler segments are nearly linear and will be missed.
- Only descending curves in the chosen band/duration/drop range are sought. Peaks at band edges, fast/short events, flat distant events, and low-contrast events can be missed. Wider bands admit more competing ridges.
- The sigmoid bank favors particular shapes and may reject valid propagation curves. Least squares is not robust to all ridge outliers. Subwindow suppression selects a clear fragment rather than a full event boundary; some duplicates or merged events remain possible.
- No independently annotated holdout exists. Inspect more recordings, annotate positives **and negatives** without tuning on them, then measure recall, false positives per hour, and crop/parameter sensitivity. Better multi-hypothesis ridge linking is a plausible next baseline comparison, not part of this bounded delivery.
- No aircraft identity, microphone coordinates, physical speed/distance/trajectory, or general localization performance is established. QGIS remains a future desktop workspace, not an implemented integration.
