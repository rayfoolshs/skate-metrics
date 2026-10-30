# Web API

The local FastAPI server exposes a small job-based API. The browser UI uses the
same endpoints.

Base URL: `http://127.0.0.1:8000` (or your LAN/HTTPS address).

## Jobs

A job moves through:

```
queued → running → done
                 ↘ error
```

`stage` is human-readable (`Detecting pose in every frame`, `Rendering skeleton
overlay`, …). The full job record is persisted at
`out/web/<job_id>/job.json`.

### Create an upload job

```http
POST /api/analyze
Content-Type: multipart/form-data
```

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `file` | file | yes | `.mp4`, `.mov`, `.m4v`, `.webm`, `.mkv`, `.avi` |
| `pose` | string | no | `mediapipe` (default) or `ultralytics` |
| `start` | float | no | analysis start in seconds |
| `end` | float | no | analysis end in seconds |
| `playback_speed` | float | no | `1.0` real time, `0.4` for 40% slow motion |

```bash
curl -F "file=@jump.mp4" -F "pose=mediapipe" \
  http://127.0.0.1:8000/api/analyze
```

Response: `{"job_id": "..."}`

### Create a YouTube job

```http
POST /api/youtube
Content-Type: application/json
```

```json
{
  "url": "https://www.youtube.com/shorts/...",
  "pose": "mediapipe",
  "cookies": "safari",
  "start": 9.6,
  "end": 10.2,
  "playback_speed": 0.4
}
```

| Field | Notes |
| --- | --- |
| `url` | YouTube watch/shorts/youtu.be URL |
| `pose` | `mediapipe` or `ultralytics` |
| `cookies` | optional: `safari`, `chrome`, `firefox`, `edge`, `brave` |
| `start` / `end` | optional range in seconds |
| `playback_speed` | optional; `0.4` for 40% speed |

The server uses yt-dlp. If YouTube blocks the download, the job error explains
that cookies may be required.

### Create the synthetic demo

```http
POST /api/demo
```

No body. Useful for testing the UI and the report without a video.

### Get a job

```http
GET /api/jobs/{job_id}
```

Example:

```json
{
  "id": "137da550fcfa",
  "status": "done",
  "stage": "Done",
  "error": null,
  "source": "Synthetic demo",
  "kind": "demo",
  "jumps": [
    {
      "takeoff_index": 85,
      "landing_index": 119,
      "takeoff_time": 1.4015,
      "landing_time": 1.9985,
      "flight_time": 0.5971,
      "height_m": 0.437,
      "rotations": 1.4927,
      "half_turns": 1.5,
      "direction": "clockwise",
      "avg_rotation_rps": 2.5,
      "peak_rotation_rps": 2.6623,
      "playback_speed": 1.0,
      "method": "foot",
      "rotation_profile": [[1.4167, -1.8159]]
    }
  ],
  "warnings": [],
  "rotation_source": "shoulders-3d",
  "fps": 60.0,
  "width": 1280,
  "height": 720,
  "stabilization": {"applied": false},
  "report_url": "/jobs/137da550fcfa/report",
  "overlay_url": "/jobs/137da550fcfa/overlay.mp4",
  "skeleton_url": "/jobs/137da550fcfa/skeleton.mp4",
  "metrics_url": "/jobs/137da550fcfa/metrics.json",
  "csv_url": "/jobs/137da550fcfa/rotation.csv"
}
```

`method` is `foot` (primary airborne detector) or `com` (moving-camera
fallback). Treat `com` results and any result with a camera-motion warning as
approximate.

### List jobs

```http
GET /api/jobs
```

Returns up to 100 summaries, newest first, without the per-frame rotation
profiles.

### Delete a job

```http
DELETE /api/jobs/{job_id}
```

Queued and running jobs cannot be deleted.

## Result files

| URL | Content |
| --- | --- |
| `/jobs/{id}/report` | standalone HTML report |
| `/jobs/{id}/overlay.mp4` | skeleton + metrics over the source footage |
| `/jobs/{id}/skeleton.mp4` | skeleton-only video |
| `/jobs/{id}/metrics.json` | machine-readable result |
| `/jobs/{id}/rotation.csv` | per-frame time, angle and rate |

Videos are H.264. If an older result is `mp4v`, the server transcodes it once to
`*.web.mp4` and serves that.

## Health

```http
GET /health
```

```json
{"ok": true, "jobs": 3}
```

## Errors

Errors use ordinary HTTP status codes with a `detail` field:

```json
{"detail": "not a YouTube URL"}
```

Long-running failures are reported through the job:

```json
{"status": "error", "error": "YouTube blocked this download. ..."}
```

## CLI equivalents

```bash
skate-metrics analyze clip.mp4 -o out/job --pose mediapipe
skate-metrics youtube "https://youtu.be/..." -o out/job --cookies safari
skate-metrics demo -o out/demo
skate-metrics compare low60.mp4 high240.mp4 -o out/validation
```
