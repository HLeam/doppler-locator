# Experimental physical consistency fits

This is a bounded extension, **not reliable source separation or an exhaustive
Doppler-source finder**. The original `doppler-locator` detector/defaults are
unchanged. `doppler-physical` runs that detector with an explicit low-band preset,
then fits its observed points. It never uses the physical template to pick peaks,
join completed ridges, fill gaps, promote detector categories, or establish an
aircraft identity. A good fit to an incorrectly associated path is still possible.

## Reproduce

```sh
uv sync --frozen --extra dev
uv run doppler-physical --input recording.wav --output outputs/physical-hour
uv run doppler-physical --help
# Independent synthetic trajectory AND 48 kHz stereo signal controls:
uv run python scripts/evaluate_physical.py --output outputs/physical-controls
uv run pytest -q
```

Use new/empty output directories. Input audio is read-only; no network services.
Real media and generated evidence must remain outside Git. Analysis covers the
entire input by default; `--start` and `--end` explicitly select a subset.

### Frozen low-band preset and tradeoffs

| Setting | Physical CLI | Original detector CLI |
|---|---:|---:|
| Discovery band | **20–800 Hz** | 300–600 Hz |
| Detection Hann FFT | **16384 samples** | 8024 samples |
| Hop | **4012 samples** | half FFT unless overridden |
| Minimum duration | **20 s** | 15 s |
| Minimum endpoint-median drop | **30 Hz** | 60 Hz |
| Separate review Hann FFT | **32768 samples** | optional |

Other peak/link/shape thresholds are the existing `DetectorConfig` defaults.
All are configurable, including `--fmin`, `--fmax`, `--n-fft`, and `--review-fft`.
The preset deliberately includes strong sub-200 Hz trajectories and possible
upper harmonics; it is **not an accuracy-validated replacement default**. Lower
frequencies contain many strong stationary/stepped tones. The broader band and
smaller drop admit more unrelated paths and duplicate fragments. The 16384 FFT
is a compromise: at 48 kHz it has 2.9296875 Hz bins versus 5.982 Hz at 8024 and
1.46484375 Hz at review FFT 32768. It is not a display-only change.

At 48 kHz, six-frame means have a 0.5015 s step; detection sample support is
0.75925 s, review support is 1.100583 s. Existing Gaussian peak smoothing adds
roughly 2.006 s on each side. Adjacent points are correlated. Spectra use mean
stereo **powers**, not waveforms, recording-zero-aligned complete Hann windows,
no resampling/padding, and relative digital bin power (not SPL or PSD).

### Optional explicitly selected diagnostic points

```sh
uv run doppler-physical --input recording.wav --output outputs/seeded-check \
  --points selected-points.csv --points-step-s 0.5015
```

CSV columns: `event_id,time_s,frequency_hz`. Extra provenance columns (contrast,
competing peaks, manual corridor, measurement FFT, omission reason) are retained
in JSON. IDs contain letters/digits/underscore/hyphen only. Rows must be ordered
strictly by time within each ID, finite, in the selected audio interval/band,
and contain at least six observations. No silent sorting or repair occurs.
`--points-step-s` is the nominal underlying observation-bin cadence; sparse
selected points still leave gaps. It is **not** evidence that every bin in a
sparse point's neighborhood was measured.

All such tracks are `manually_selected_diagnostic`, even if the CSV came from
another automatic tool. They are **never counted as whole-hour automatic
recovery**. Record how points were selected; fitting cannot undo selection bias.
An anchor or interpolated guide is not itself a measured frequency.

## Baseline model and timing derivation

Assume a stationary receiver, a subsonic source moving along a straight path at
constant speed `v`, a constant emitted reference tone `f0`, and a uniform still
medium with sound speed `c`. Define:

- `t0`: emission-time closest approach on the recording-zero time coordinate;
- `b > 0`: minimum **slant** source–receiver distance, not altitude or map position;
- `tau = b/v > 0`, seconds;
- `beta = v/c`, dimensionless, strictly between zero and one;
- `e`: emission time; `t`: receive time.

Then

```text
R(e) = sqrt(b² + v²(e - t0)²)
dR/de = v²(e - t0)/R(e)       # negative while approaching
t = e + R(e)/c               # retarded propagation
f(t) = f0 * de/dt = f0 / (1 + (dR/de)/c)
T = t0 + b/c = t0 + beta*tau # arrival time of CPA-emitted sound
```

Approaching frequencies exceed `f0`; receding frequencies fall below it. At `T`
the received frequency equals `f0`. The code compares predicted frequencies to
observations at their **receive** times, not at emission times.

For `q=(t-t0)/tau`, the retarded inverse is

```text
(e-t0)/tau = [q - beta*sqrt(q² + 1 - beta²)] / (1 - beta²)
```

Equivalently, a stable derivative form is

```text
z = t - t0
w = tau*sqrt(1-beta²)
f(t) = f0/(1-beta²) * [1 - beta*z/sqrt(z²+w²)]
```

That last expression is used by the fitter. An independent emission-time
forward generator and numerical phase derivative check it in tests and the
control script; the truth data do not call the inverse fitter model.

**Four distinct notions of “middle”:**

1. Emission CPA: `t0`.
2. Received arrival of CPA-emitted sound: `T=t0+b/c`, later by the sound delay.
3. Maximum descending slope / inflection **under this exact baseline**: its
   receive-time coordinate equals `t0`, not `T`. It is sound emitted before CPA.
   An arbitrary observed bend need not satisfy this model-specific equality.
4. Midpoint of the visible fragment: arbitrary support arithmetic, neither CPA.

`--sound-speed-m-s 343` is an **assumption**, not measured site weather. At fixed
fitted `beta,tau,T`, changing `c` rescales `v` and `b` but not timing or frequencies.
If an observed ridge is harmonic order `h`, fitted `f0` is its emitted reference
frequency. Without independent order evidence it is **not an engine fundamental**.
Wind, acceleration, intrinsic pitch change, multipath, and source switches can
violate the baseline; the fit does not infer these away.

## Bounded fitting and identifiability

The four fitted quantities are emitted reference frequency `f0` (Hz), `beta`
(dimensionless), `tau` (s), and `T` (s). Derived outputs are `v=beta*c` (m/s),
`b=beta*c*tau` (m), `t0=T-beta*tau` (s), sound delay (s), and model inflection (s).
Times always refer to recording zero; no absolute date or geographic coordinate.

Full noiseless asymptotes can identify `beta=(f_approach-f_recede)/
(f_approach+f_recede)` and `f0=2*f_approach*f_recede/(f_approach+f_recede)`.
Shape width then informs `tau`; timing informs `t0`. Partial curves often do not
observe either asymptote: emitted pitch, speed ratio and range/time scale trade
off. An excellent residual can coexist with a large or unbounded received-CPA
arrival delay. No optimizer success flag establishes physical identifiability.

Default bounds (all configurable): `f0=5…2000 Hz`, `beta=.01….9`,
`tau=.5…1000 s`, and `T` from three observed spans before the first point to three
spans after the last. These are numerical search assumptions, not site or
vehicle priors. No inter-event spacing prior is used.

The implementation:

1. Minimizes ordinary squared frequency residuals on **only observed points**,
   using 18 deterministic starts (three speeds × three arrival phases × two
   time scales). It centers/scales times and log-parameterizes positive scales.
2. Profiles each of all four fitted parameters over 17 fixed grid positions
   plus the optimum, refitting the three nuisance parameters from two starts.
   The scale profiles are logarithmic. Eight bisections refine each sampled
   tolerance crossing so a coarse grid cannot produce a spurious zero width.
3. Exports **every** solution, convergence outcome, profile identity, bound hit,
   residual RMSE and whether it is comparably good. The comparable set satisfies
   `MSE <= best_MSE + frequency_floor_hz²`. Its extrema define sensitivity
   envelopes for fitted/derived parameters and predicted frequencies.

The default `--frequency-floor-hz 2` is an explicit **practical tolerance**, not
a calibrated Gaussian standard deviation. Envelopes are finite-search
sensitivity diagnostics, **not confidence/credible intervals**. They can miss
unsampled alternatives and do not include source-association or model error.
No inverse-Hessian covariance, IID sample-size shrinkage, or probabilistic
coverage is asserted. Inspect full exported ranges even when a status passes.
Bounds touching a comparable solution indicate unresolved sensitivity, not an
estimated upper/lower physical limit. Widen bounds or tolerance as a declared
sensitivity experiment, not as hidden tuning to obtain a desired CPA.

Constant and linear frequency fits are exported. `conditionally_constrained`
requires all of: a converged best fit; physical RMSE <= twice the tolerance;
linear RMSE improvement >= one tolerance; no comparable solution on any
parameter bound; and both CPA sensitivity widths <= one quarter of the observed
span. These are **conservative descriptive gates, not statistical significance
or verified continuity**. A long support span can still permit a sizable timing
envelope. Otherwise status is `insufficient_evidence`, with explicit reasons;
the numerical best solution remains available **as an example, not an estimate
to quote as established**. Nulls and near-linear fragments can fit extremely
well yet correctly have insufficient evidence.

`cpa_separation()` refuses a difference unless both results pass the conditional
gate and the caller names the **same** feature (`cpa_receive_s` or
`cpa_emission_s`). Its difference envelope is the conservative cross-product of
the two sensitivity envelopes, not a statistical interval or identity match.

## Association, harmonics and output contract

- `candidates.csv`, `ridges.csv`, `overview.png`, `run.json`: existing detector
  products in blind mode. `strict_curved_event` and
  `unconfirmed_descending_tone` remain separate and unchanged by physical fits.
- `physical-fits.json`: input/settings hashes and versions; assumptions; all
  raw selected points and their original provenance; support/gaps; null and
  physical residual summaries; best and alternative parameter values/bounds;
  parameter sensitivity envelopes; and per-point predictions/residuals.
- `physical-points.csv`: observed frequency, physical prediction, residual
  (`observed - predicted`, Hz), category and unverified association per point.
  Prediction rows correspond **one-to-one** to observations, never filled gaps.
- `E…-physical-review.png`: observed dots on raw review FFT 32768, fitted
  predictions and finite sensitivity envelope, linear alternative, gray omitted
  intervals. Dashed model curves between points are predictions, not measured
  support. Conditional CPA markers appear only when the timing gate passes;
  no marker claims an actual source CPA. Raw point errors can exceed the smooth
  model-envelope width: the envelope is not an observation-noise band.
- `audit-…png`: every 300-second portion, including zero-candidate intervals,
  with separate low/upper frequency panels, IDs and observed points. All physical
  close-ups are written; `--max-plots` only caps the legacy detector close-ups.
- `coverage.json`: analyzed bin-center extent, bin and track counts, status
  counts and possible pair relations; **independent physical-source count null**.

Harmonic evidence uses one-to-one **actual** time matches within a quarter bin,
not interpolation over gaps. It needs >=12 matched points spanning >=10 s,
reference movement > twice the frequency tolerance, and >=80% agreement with an
integer ratio 1…4. Tolerance is `max(2 Hz, detection bin spacing)` on the divided
reference scale. Order 1 means a possible duplicate; 2…4 a possible harmonic.
Relations export paired support, mismatch and fraction. Grouping is always
uncertain: an integer ratio does not prove a common source or a fundamental.
Tracks are retained, not silently counted as independent events or transitively
merged into an asserted source. Unmatched tracks may still belong to one source.

## Validation and bounded stopping rule

Tests cover independent forward/inverse agreement, phase-derivative frequency,
units/sign/delay and inflection, multiple trajectory sampling phases/noise
levels, known-parameter recovery, sound-speed scaling, null and truncated/far-tail
non-identifiability, gap retention, harmonic/nonconcurrent controls, output
provenance, and executable blind/seeded/empty CLI behavior.

The reproducible evaluation script also synthesizes 48 kHz antiphase stereo
signals with multiple signal phases/noise levels, a missing-signal interval,
a physical curve crossing a louder constant tone, harmonics, independent
crossing and descending lines, a single linear tone, and a constant tone. It
reports actual returned point paths and recovery/error/status rather than just
event overlap. The harmonic control's reference-track error is intentionally
large for the 2× member; pair evidence, not that error, records the relationship.
These finite controls are not a general false-positive or accuracy estimate.

A whole-recording audit is mandatory for a new real input/settings combination.
Review extra candidates, zero-output spans, duplicate fragments and competing
paths, not just two pleasing fits. If supported observations remain partial or
association is ambiguous, report that limitation. If the baseline is
underconstrained, stop with the residuals/ranges and concrete options (more
unambiguous tonal support, independently established emitted pitch/harmonic
order, or a separately authorized tracking study). Do not manufacture a CPA,
force a presumed spacing, or expand this module into universal source separation.
