# Reference source and reproducibility

- Public source URL: https://www.youtube.com/watch?v=4CxVxuMVkMc
- Source identifier: `4CxVxuMVkMc`; source describes itself as **“Unknown location - raw audio”**.
- Development inputs: an already-downloaded format-251 WebM/Opus audio stream and a local decoded WAV. No download is performed by this repository, and no media is redistributed.
- Inspected WAV: 48,000 Hz, 2 channels, signed 16-bit PCM; 172,808,847 frames (3600.1843125 seconds).
- WAV file SHA-256: `bcbe8f598092d7e6f5cfbef13cb6e475f02906cf08a85b4a85c6c0845580d769`.
- Compressed source file SHA-256: `6884324b4fb9babbc89b10de71f283dd338766a15d2fbd1aeedb52216276692d`.

These hashes identify the local files used for development; they do not grant permission to redistribute them. The media belongs to its respective rights holder(s). No microphone coordinates or private source paths are included here.

If you already have a permitted copy of the compressed audio, an illustrative conversion is:

```sh
ffmpeg -n -i recording.webm -map 0:a:0 -vn -ar 48000 -ac 2 -c:a pcm_s16le recording.wav
```

The experiment used the **existing WAV**, not a new decode. The original decoder version/command is not known; this recipe describes the format, not guaranteed byte-identical reconstruction. Decoder versions, headers and online source revisions can change file hashes or sample alignment. Check the sample rate, channel count and duration; retain your own source attribution and checksum. The CLI accepts other sample rates without resampling, so the same FFT length then gives different frequency spacing.

For reproducibility, use the committed `uv.lock` and the commands in the README. Each run records the input hash, library versions and all configurable analysis/detector settings in `run.json`. The only supplied manual labels are approximately 20:30–21:30 and 35:00–36:15 with a roughly 500→330 Hz descent; these are **development labels**, not independently verified events or aircraft identities.
