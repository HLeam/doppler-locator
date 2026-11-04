"""Low-band-first blind discovery plus bounded physical model diagnostics."""

import hashlib
import json
from dataclasses import asdict, fields
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import spectrogram
from .cli import parser
from .detector import DetectorConfig, detect
from .output import write_outputs
from .physical import FitConfig, fit_observations
from .physical_output import candidate_track, load_diagnostic_points, plot_audit, plot_fit, write_fit_outputs


def main(argv=None):
    arg_parser = parser()
    arg_parser.description = (
        "Experimental 20–800 Hz discovery + retarded-time Doppler consistency, not source recovery."
    )
    arg_parser.set_defaults(
        fmin=20,
        fmax=800,
        n_fft=16384,
        hop=4012,
        min_duration_s=20,
        min_drop_hz=30,
        review_fft=32768,
        max_plots=0,
    )
    arg_parser.add_argument(
        "--points",
        type=str,
        help="externally selected CSV diagnostic instead of blind detection; never counted as automatic recovery",
    )
    arg_parser.add_argument(
        "--points-step-s",
        type=float,
        help="nominal observation cadence required with --points; gaps remain missing",
    )
    for field in fields(FitConfig):
        arg_parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=int if field.name in {"profile_points", "max_nfev"} else float,
            default=getattr(FitConfig(), field.name),
        )
    args = arg_parser.parse_args(argv)
    try:
        fit_config = FitConfig(**{field.name: getattr(args, field.name) for field in fields(FitConfig)})
        fit_config.validate()
        detector = DetectorConfig(
            **{field.name: getattr(args, field.name) for field in fields(DetectorConfig)}
        )
        detector.validate()
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("output must be new or empty")
        if args.max_plots < 0:
            raise ValueError("max-plots must be nonnegative")
        if args.points and (
            args.points_step_s is None or not np.isfinite(args.points_step_s) or args.points_step_s <= 0
        ):
            raise ValueError("--points requires a positive --points-step-s")
        if args.points_step_s is not None and not args.points:
            raise ValueError("--points-step-s requires --points")
        info = sf.info(args.input)
        options = dict(
            start=args.start,
            end=args.end,
            fmin=args.fmin,
            fmax=args.fmax,
            hop=args.hop,
            time_step=args.time_step,
            chunk_seconds=args.chunk_seconds,
            channel=args.channel,
        )
        spec = spectrogram(args.input, n_fft=args.n_fft, **options)
        review = spectrogram(args.input, n_fft=args.review_fft, **options)
        candidates = [] if args.points else detect(spec, detector)
        tracks = (
            load_diagnostic_points(args.points, args.points_step_s)
            if args.points
            else [candidate_track(c) for c in candidates]
        )
        results = []
        for track in tracks:
            t = np.array([p["time_s"] for p in track["points"]])
            f = np.array([p["frequency_hz"] for p in track["points"]])
            if t.min() < args.start or t.max() > (args.end if args.end is not None else info.duration):
                raise ValueError("point times must lie in the selected audio interval")
            if f.min() < args.fmin or f.max() > args.fmax:
                raise ValueError("point frequencies must lie in the selected review band")
            print(f"Fitting {track['event_id']}: {len(t)} observed points", flush=True)
            results.append(fit_observations(t, f, fit_config))
        with args.input.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = {
            "input_sha256": digest,
            "input_frames": info.frames,
            "input_channels": info.channels,
            "sample_rate_hz": info.samplerate,
            "analysis": {
                "start_s": args.start,
                "end_s": args.end if args.end is not None else info.duration,
                "fmin_hz": args.fmin,
                "fmax_hz": args.fmax,
                "channel": args.channel,
                "target_time_step_s": args.time_step,
                "chunk_seconds": args.chunk_seconds,
            },
            "detection_fft": None if args.points else spec.n_fft,
            "analysis_fft": spec.n_fft,
            "review_fft": review.n_fft,
            "hop_samples": spec.hop,
            "frames_per_bin": spec.frames_per_bin,
            "actual_time_step_s": spec.time_step,
            "frequency_step_hz": float(np.diff(spec.frequencies)[0]),
            "detector": asdict(detector),
            "physical": asdict(fit_config),
            "selection": "externally_selected_diagnostic" if args.points else "blind_whole_selected_interval",
            "diagnostic_points_sha256": hashlib.sha256(Path(args.points).read_bytes()).hexdigest()
            if args.points
            else None,
            "harmonic_tolerance_hz": max(2, info.samplerate / args.n_fft),
            "assumptions": [
                "stationary receiver",
                "constant emitted reference tone (not known fundamental)",
                "subsonic straight constant-speed path",
                "uniform still medium",
                "sound speed is assumed, not measured site weather",
                "slant CPA distance is not altitude or a coordinate",
            ],
            "coverage_limitations": [
                "finite-band descending tonal candidates only; no guarantee of all Doppler sources",
                "merged peak association can switch at crossings; good fits do not verify continuity",
                "no source separation, completion of tails, or interpolation into measured support",
                "profile envelopes are finite sensitivity searches, not confidence intervals",
            ],
            "versions": {name: version(name) for name in ("numpy", "scipy", "soundfile", "matplotlib")},
        }
        if not args.points:
            write_outputs(args.output, spec, candidates, detector, dict(metadata), args.max_plots)
        payload = write_fit_outputs(args.output, tracks, results, metadata)
        for track, result in zip(tracks, results, strict=True):
            plot_fit(
                review,
                track,
                result,
                args.output / f"{track['event_id']}-physical-review.png",
                metadata["detection_fft"],
            )
        # Complete review coverage, including zero-output intervals. No plotting cap.
        stop = metadata["analysis"]["end_s"]
        for start in np.arange(args.start, stop, 300):
            end = min(start + 300, stop)
            if np.count_nonzero((review.times >= start) & (review.times <= end)) < 2:
                continue
            plot_audit(
                review,
                tracks,
                start,
                end,
                args.output / f"audit-{start:08.1f}.png",
                metadata["detection_fft"],
            )
        (args.output / "coverage.json").write_text(
            json.dumps(
                {
                    "analyzed_receive_bin_centers_s": [float(spec.times[0]), float(spec.times[-1])],
                    "analysis_bins": len(spec.times),
                    "candidate_tracks": len(tracks),
                    "automatic_candidate_tracks": len(candidates),
                    "possible_pair_relations": len(payload["harmonic_relations"]),
                    "independent_physical_sources": None,
                    "fit_status_counts": {
                        status: sum(r["status"] == status for r in results)
                        for status in ("conditionally_constrained", "insufficient_evidence")
                    },
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    except (OSError, ValueError, sf.LibsndfileError) as error:
        arg_parser.error(str(error))
    print(f"{len(tracks)} fitted candidate/diagnostic tracks; not a count of sources. Outputs: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
