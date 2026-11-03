"""Local audio-curve CLI; never downloads or uploads media."""

import argparse
import hashlib
import time
from dataclasses import fields
from pathlib import Path

import soundfile as sf

from .audio import spectrogram
from .detector import DetectorConfig, detect
from .output import write_outputs


def seconds(value: str) -> float:
    """Accept seconds or MM:SS / HH:MM:SS."""
    try:
        parts = [float(part) for part in value.split(":")]
        if not 1 <= len(parts) <= 3 or any(part < 0 for part in parts):
            raise ValueError
        if len(parts) > 1 and any(part >= 60 for part in parts[1:]):
            raise ValueError
        result = 0.0
        for part in parts:
            result = result * 60 + part
        return result
    except ValueError as error:
        raise argparse.ArgumentTypeError("use nonnegative seconds or MM:SS / HH:MM:SS") from error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Find candidate descending Doppler-like frequency curves (not aircraft identification)."
    )
    result.add_argument(
        "--input", required=True, type=Path, help="local audio readable by libsndfile; WAV recommended"
    )
    result.add_argument("--output", required=True, type=Path, help="new or empty local output directory")
    result.add_argument("--start", type=seconds, default=0.0, help="seconds or MM:SS / HH:MM:SS")
    result.add_argument("--end", type=seconds, help="exclusive recording end; default: whole recording")
    result.add_argument("--fmin", type=float, default=300.0)
    result.add_argument("--fmax", type=float, default=600.0)
    result.add_argument(
        "--n-fft",
        type=int,
        default=8024,
        help="exact native-rate FFT/window samples, not rounded to power of two",
    )
    result.add_argument("--hop", type=int, help="native-rate samples; default n_fft // 2")
    result.add_argument(
        "--time-step",
        type=float,
        default=0.5,
        help="target averaged STFT step (seconds); rounded to whole hops",
    )
    result.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="IO/FFT batch duration; does not change detection grid",
    )
    result.add_argument(
        "--channel",
        choices=("power", "left", "right"),
        default="power",
        help="mean channel powers (no phase cancellation), or selected channel",
    )
    result.add_argument(
        "--max-plots",
        type=int,
        default=20,
        help="event close-ups, highest scores first; overview always written",
    )
    defaults = DetectorConfig()
    for item in fields(defaults):
        result.add_argument(
            "--" + item.name.replace("_", "-"),
            type=float,
            default=getattr(defaults, item.name),
            help=f"detector {item.name}; default %(default)s (see README for units/formula)",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    config = DetectorConfig(**{item.name: getattr(args, item.name) for item in fields(DetectorConfig)})
    started = time.monotonic()
    try:
        config.validate()
        if args.max_plots < 0:
            raise ValueError("max_plots must be nonnegative")
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("output must be a new or empty directory; refusing to overwrite files")
        info = sf.info(args.input)
        spec = spectrogram(
            args.input,
            start=args.start,
            end=args.end,
            fmin=args.fmin,
            fmax=args.fmax,
            n_fft=args.n_fft,
            hop=args.hop,
            time_step=args.time_step,
            chunk_seconds=args.chunk_seconds,
            channel=args.channel,
        )
        candidates = detect(spec, config)
        with args.input.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = {
            "input_sha256": digest,
            "input_frames": info.frames,
            "input_channels": info.channels,
            "analysis": {
                "start_s": args.start,
                "end_s": args.end if args.end is not None else info.duration,
                "fmin_hz": args.fmin,
                "fmax_hz": args.fmax,
                "channel": args.channel,
                "target_time_step_s": args.time_step,
                "chunk_seconds": args.chunk_seconds,
            },
            "analysis_elapsed_seconds": time.monotonic() - started,
        }
        write_outputs(args.output, spec, candidates, config, metadata, args.max_plots)
    except (OSError, ValueError, sf.LibsndfileError) as error:
        arg_parser.error(str(error))
    print(f"{len(candidates)} candidate(s); heuristic scores, not probabilities. Outputs: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
