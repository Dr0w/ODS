# ODS

Open Face Defense System local PoC.

## What it does

- `server.py` starts a local socket server on `127.0.0.1:5100`
- `client.py` sends a `START_DETECTION` command to the server
- the server opens the webcam, applies contrast enhancement and smoothing, detects frontal and profile faces plus eyes, and shows the live result
- press `q` in the OpenCV window to stop detection and send a summary back to the client

## Detection features

- tries camera index `0` first, then `1`
- uses CLAHE and Gaussian blur before detection for more stable results in uneven lighting
- combines frontal-face and profile-face Haar cascades
- runs profile detection on both the original and mirrored frame so side-facing heads are easier to catch
- merges overlapping detections and shows a live HUD with face count, eye count, frame count, and FPS

## Setup

```bash
pip install -r requirements.txt
```

## Run

Start the server first:

```bash
python server.py
```

In another terminal, trigger detection:

```bash
python client.py
```