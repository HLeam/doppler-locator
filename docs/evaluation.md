# Evaluation and limitations

## Plan and scope

The implementation sequence was: inspect local native-rate spectrograms; implement a chunked STFT and peak-linking baseline; test synthetic positives/controls; inspect the two supplied development clips and a separate interval; scan the hour; preserve and document shortcomings rather than chase perfect detections. No ML, external service, localization, or physical parameter fitting was used. The detailed initial plan and real-data evidence remain private/local.

The manual intervals (1230–1290 s and 2100–2175 s, approximately 500→330 Hz) influenced the default frequency band, event scale and contrast threshold. Initial local-contrast thresholds of 3 dB fragmented both target ridges; a small sweep through 2, 1.5 and 1 dB led to 1.5 dB. Suppression overlap was set to 40% to avoid duplicate partial windows. These are development choices, **not independently validated parameters**. The later category addition exposes a partial near-linear segment as a discovery; it does not assert identity or fill unseen portions.

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

The initial whole-hour second-example miss combined crop-dependent frame/bin phase, ridge fragmentation, and insufficient strict-curvature evidence. Frames and power averages now align to recording time zero, and peak association uses a minimum-cost one-to-one assignment. Concurrent ridges are not joined as if they were a sequential continuation. Instead, an observed near-linear descending fragment can be reported in its separate unconfirmed discovery category. Strict curved-event criteria remain distinct.

## Current detector: context comparison

The same defaults (FFT 8024, hop 4012, 0.5015 s power bins, 300–600 Hz) were used for padded clips, larger intervals, and the full recording. Frames and averaging bins align to recording time zero. No audio-derived output files were written. The strict category also now requires the curve fit to improve absolute RMSE by at least half an FFT bin; this avoids treating a sub-bin fit change on overlapping linear tones as meaningful curvature.

| Example/context | Category counts | Selected candidate interval | Endpoint medians | Observed support / label interval |
|---|---|---|---|---:|
| First, 1200–1320 s | 1 strict, 1 discovery | 1255.55–1285.14 s (strict) | 497.4→367.9 Hz | 28.6 / 60 s |
| First, 900–1500 s | 2 strict, 1 discovery | 1255.55–1285.14 s (strict) | 497.4→367.9 Hz | 28.6 / 60 s |
| First, whole recording | 4 strict, 5 discovery | 1255.55–1285.14 s (strict) | 497.4→367.9 Hz | 28.6 / 60 s |
| Second, 2070–2205 s | 0 strict, 1 discovery | 2126.15–2144.71 s (discovery) | 446.9→376.7 Hz | 18.1 / 75 s |
| Second, 1800–2400 s | 0 strict, 1 discovery | 2126.15–2144.71 s (discovery) | 446.9→376.7 Hz | 18.1 / 75 s |
| Second, whole recording | 4 strict, 5 discovery | 2126.15–2144.71 s (discovery) | 446.9→376.7 Hz | 18.1 / 75 s |

For each example, the selected category-specific candidate has the same observed interval and frequency path in the padded crop, larger interval, and whole recording (maximum interpolated path difference on the shared interval: 0.0 Hz at the 0.5 s comparison grid). The first strict candidate has 0.95 observation coverage and 3 short internal gaps, each about 0.5 s; the second discovery has 0.947 coverage and 2 short gaps, each about 0.5 s. These candidates cover only 47.7% and 24.1% of the approximate labeled intervals respectively. Their leading/trailing unobserved portions are not reconstructed. The second trajectory remains provisional: the overlapping ridge makes a source-continuity claim unsafe, and its observed fragment does not pass strict curvature.

The whole scan returns 9 candidates (4 strict, 5 discovery), compared with 2 candidates in the initial baseline: the first example and an unlabeled 05:18–05:45 event. The prior unlabeled 318–345 s candidate remains. Seven candidates are added: one partial second-example discovery, one discovery overlapping the first example (1265.58–1325.26 s in the larger and whole contexts), and five candidates outside the supplied intervals that remain unclassified. They are not established false positives. The separate 10:00–13:00 interval still returns zero candidates under the frozen defaults. Counts and intervals are descriptive development observations, not recall, precision, or general accuracy estimates. Scores are heuristic rankings, not probabilities.

### Visual review (images kept local)

Opened and inspected the final annotated images:

- `development-1/E0001.png`: first supplied example; cyan ridge follows the visible main descent. Green dashed line is only the descriptive shape template.
- `development-2/E0001.png`: second supplied example; partial descending ridge amidst broadband interference and adjacent tones. It does not recover the entire supplied interval.
- `full-hour/E0001.png`: unlabeled 05:18–05:45 candidate, a visible tonal descent that becomes near-stationary. It is plausible curve evidence, but may be an unrelated changing-pitch source; its identity and truth status remain unknown.
- `separate-10-13min/overview.png`: a separately inspected 10:00–13:00 interval, with substantial broadband energy around 11:30 and stationary/narrowband components; no accepted 300–600 Hz curve. A wider preliminary view showed other-frequency structure outside the default band. This interval was **not relabeled as an independent negative** and was not used to select thresholds.

No harmonic family was consistently established for both examples in the wider preliminary views (200–1000 Hz). Harmonics therefore supply neither a requirement nor a score bonus.

## Synthetic and software checks

The focused synthetic and CLI checks cover these signal cases:

- Three seeded descending tanh curves (505→335 Hz approximately) in Gaussian noise plus a stationary non-harmonic interferer: each detected once, median ridge error <4 Hz.
- One descending radial-velocity-shaped `t/sqrt(t²+scale²)` curve, which is **not** the fitting template: detected once with median ridge error <4 Hz.
- Negative/control clip cases: ascending curve, a linear drift reported only as discovery, constant tone, abrupt frequency step, silence, several noise seeds, crossing linear tones, and two independent descending tones. The independent-tone cases produce no strict curved event.
- Additional stereo antiphase, sparse-ridge rejection, low-excursion rejection, FFT=8024/native-rate timing, chunk-boundary invariance, argument validation, NaN audio rejection, missing input, CSV/JSON/PNG output, and overwrite-refusal checks.

A regression during implementation caught a score bug: flooring the line residual at half an FFT bin falsely made a synthetic linear drift look curved. Gain compares to the actual line residual (with only a numerical zero guard), and strict classification now also requires an absolute curve-fit RMSE improvement of half an FFT bin. Synthetic signals remain much simpler and stronger than real recording conditions. Passing these tests does not validate real-world discrimination.

One local ARM64 macOS full-hour run took about 6 s wall time including plots and SHA-256, peaking at approximately 227 MiB RSS. The final evaluation recorded 4.52 s for full-hour analysis plus checksum, excluding plot writing; clip analyses were 0.51–0.63 s. These are machine/cache-specific observations, not benchmarks or guarantees. The retained reduced spectrogram scales with time and frequency bandwidth; the waveform/FFT batches do not require loading the hour into memory.

## Known limits and useful next experiments

- Real misses, fragmentary boundaries, crop sensitivity and crossing-track swaps already occur. A confidence score is a heuristic ranking, not calibrated uncertainty.
- Oscillator/engine pitch changes, instruments, machinery, vehicles, interference, and time-varying background can mimic a Doppler-like curve. Pure linear drifts are never strict curved events; they may appear as unconfirmed discoveries. Some actual Doppler segments are nearly linear and will be absent from the strict category.
- Only descending curves in the chosen band/duration/drop range are sought. Peaks at band edges, fast/short events, flat distant events, and low-contrast events can be missed. Wider bands admit more competing ridges.
- The sigmoid bank favors particular shapes and may reject valid propagation curves. Least squares is not robust to all ridge outliers. Subwindow suppression selects a clear fragment rather than a full event boundary; some duplicates or merged events remain possible.
- No independently annotated holdout exists. Inspect more recordings, annotate positives **and negatives** without tuning on them, then measure recall, false positives per hour, and crop/parameter sensitivity. Better multi-hypothesis ridge linking is a plausible next baseline comparison, not part of this bounded delivery.
- No aircraft identity, microphone coordinates, physical speed/distance/trajectory, or general localization performance is established. QGIS remains a future desktop workspace, not an implemented integration.
