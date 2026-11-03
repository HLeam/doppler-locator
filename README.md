# Doppler Locator

Research tooling for finding candidate aircraft Doppler curves in audio and, later, evaluating microphone-location estimates against known flight paths.

## Initial experiment

Build a reproducible Python detector with annotated spectrogram and CSV outputs.
Two manually identified development examples occur around 20:30–21:30 and 35:00–36:15 in the reference recording, with tones roughly descending from 500 to 330 Hz.
These examples are development labels, not independent validation or proof of aircraft identity.

Reference recording: https://www.youtube.com/watch?v=4CxVxuMVkMc
The source identifies itself as "Unknown location - raw audio" and is approximately one hour long.
Downloaded media is not redistributed by this repository.

## Data and scope

Keep audio, flight data, DEM tiles, generated outputs, and exact recording coordinates in local ignored directories.
Use recordings and ground-truth locations that you own or have permission to analyze.
No historical flight-data integration or microphone localization is implemented yet.
QGIS is the intended desktop geographic workspace; Python scripts will provide reproducible analysis.
