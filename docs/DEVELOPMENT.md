# Development

## Prerequisites

- Python 3.11–3.13 (3.12 recommended)
- [uv](https://docs.astral.sh/uv/)
- ffmpeg + ffprobe on `PATH`
- Optional: Node.js only if you want to modify the inline web UI build process
  (the UI is currently a single HTML string in `server.py`, so none is needed)

## Setup

```bash
cd skate-metrics
uv sync --extra dev --extra pose --extra web
```

Extras:

| Extra | Installs |
| --- | --- |
| `pose` | MediaPipe + model runtime |
| `yolo` | Ultralytics (optional alternative backend) |
| `web` | FastAPI, uvicorn, python-multipart, yt-dlp |
| `dev` | pytest, httpx |

The MediaPipe pose model (~5 MB) downloads on first analysis to
`~/.cache/skate-metrics/`.

## Commands

```bash
make setup      # uv sync with all extras
make demo       # synthetic end-to-end demo
make serve      # local web app on :8000
make test       # pytest
```

```bash
uv run skate-metrics demo -o out/demo
uv run skate-metrics analyze clip.mp4 -o out/clip --pose mediapipe
uv run skate-metrics analyze clip.mp4 -o out/clip --skeleton --save-pose
uv run skate-metrics youtube "https://youtu.be/..." -o out/yt --cookies safari
uv run skate-metrics compare low60.mp4 high240.mp4 -o out/validation
uv run skate-metrics serve --host 127.0.0.1 --port 8000
```

Useful analysis flags:

- `--start` / `--end` — seconds
- `--playback-speed` — `0.4` for 40% slow motion
- `--skeleton` — write `skeleton.mp4`
- `--save-pose` — write `keypoints.json` for `--pose json`
- `--stabilize` — experimental camera-motion compensation
- `--no-overlay` — metrics/report only

## Tests

```bash
make test
# 16 passed
```

No network is required. The suite covers:

- synthetic jump metrics and 60 vs 240 fps comparison
- playback-speed correction
- time-range analysis
- one-foot false positives
- camera-drift detection
- web API end-to-end, report/video/skeleton serving, job listing/deletion
- YouTube URL and input validation

## Adding a pose backend

Implement `PoseEstimator.estimate(info, start=..., end=...)` in
`src/skatemetrics/pose.py` and return `list[PoseFrame]`.

For useful rotation counting, populate `world_keypoints` with metric 3D points.
Without them the app falls back to 2D and warns about full-revolution ambiguity.

Register the backend in `get_estimator` and add it to the CLI/UI choices if it is
safe to expose.

## Code conventions

- Python 3.11+, type hints on public functions.
- Keep pure signal processing in `geometry.py`, `kinematics.py`, `segmentation.py`
  and `metrics.py`; keep I/O in `video.py`, `pose.py`, `server.py`.
- Tests should use synthetic data, not downloaded clips.
- Do not silently hide uncertainty. Add a warning or a `method` label instead.
- Videos intended for the browser must be H.264.

## Data layout

```
out/
  demo/            synthetic demo outputs
  web/<job_id>/    server jobs (job.json, input, analysis, videos)
```

`out/` is gitignored.

## Troubleshooting

### “TypeError: Failed to fetch”

The browser cannot reach the server. Check that `make serve` is still running and
reload. The background server must stay alive; use a detached process or a
terminal that stays open.

### Browser camera does not start

`getUserMedia` needs HTTPS or `localhost`. On an iPad over LAN, use mkcert or a
tunnel. Upload still works over HTTP.

### Video is black in the browser

The file is likely `mp4v`. New renders are H.264; the server transcodes old files
on first request. Re-run the analysis if needed.

### YouTube download fails

YouTube may require sign-in or block the client. Choose “Use Safari/Chrome/
Firefox cookies” in the UI or set `SKATE_YTDLP_COOKIES` to a cookies.txt file.

### MediaPipe model download fails

Check network access to `storage.googleapis.com`, or pass a local model path via
the `MediaPipePoseEstimator(model_path=...)` constructor.

### “No plausible jumps”

Check the capture protocol. Common causes: skater too small, camera moving, only
one foot visible, wrong playback speed, or the jump is outside the selected range.
