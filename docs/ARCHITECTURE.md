# Architecture

How Skate Metrics turns a video into jump measurements, and why it is built this
way.

## 1. Product constraints

The initial brief came from a skater and coach:

- iPhone 16 Pro Max as the camera, iPad as the coach display.
- 60 fps capture on the ice.
- Feedback within 2–5 seconds of landing.
- Must report jump height and rotation speed.
- YouTube clips for research/reference.

Those constraints drove three decisions:

1. **Keep the analysis in Python first.** The computer-vision stack (MediaPipe,
   OpenCV, SciPy) is mature there. A native iOS port can follow once the metric
   definitions and failure modes are understood.
2. **Separate capture from analysis.** The local web app, CLI and future native
   app all call the same `analyze_video` pipeline. There is one implementation of
   the metrics.
3. **Be explicit about uncertainty.** A single 60 fps camera cannot measure
   everything a judging panel sees. The app labels approximate results and
   refuses to pretend that a wide broadcast shot is a lab measurement.

## 2. Pipeline

```
video / YouTube / browser recording
        │
        ▼
┌──────────────────────────────┐
│ video.py                     │  ffprobe metadata + presentation timestamps
│  probe_video / iter_frames   │  (never assume index/fps on phone footage)
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ pose.py                      │  MediaPipe Tasks, Ultralytics, or JSON
│  2D keypoints                │  MediaPipe also gives metric 3D world points
│  + 3D world keypoints        │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ kinematics.py                │  lowest foot, body centre, 3D yaw
│ segmentation.py              │  airborne window: foot baseline → CoM fallback
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ metrics.py                   │  flight time, height, rotations, rates
│                              │  playback-speed correction
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ overlay.py / report.py       │  H.264 overlay, skeleton-only video,
│                              │  JSON, CSV, standalone HTML report
└──────────────┬───────────────┘
               ▼
        CLI / FastAPI web app
```

## 3. Data model

`src/skatemetrics/models.py`:

- `Keypoint(x, y, confidence)` — 2D image pixels.
- `Keypoint3D(x, y, z, confidence)` — MediaPipe metric world landmarks.
- `PoseFrame(index, timestamp, keypoints, world_keypoints)` — one frame. The
  timestamp is the video presentation time, not `index / fps`.
- `JumpWindow` — takeoff/landing frame indices and interpolated times.
- `JumpMetrics` — the public result: flight time, height, rotations, half-turn
  count, direction, average/peak rotation rate, playback speed, detector method
  (`foot` or `com`), and the rotation profile.

`JumpMetrics.to_dict()` is the JSON contract used by the CLI, web app and report.

## 4. Pose backends

### MediaPipe (default)

`MediaPipePoseEstimator` uses the Tasks `PoseLandmarker` in VIDEO mode. It
provides:

- 33 2D landmarks, including heels and foot indices.
- 33 metric 3D world landmarks.

The 3D landmarks are important: a projected 2D shoulder line is only defined
modulo 180°. When the shoulders go edge-on, the line flips and a naive unwrap
loses half a revolution. The 3D shoulder yaw has no such singularity, so full
revolution counting is possible.

The model auto-downloads to `~/.cache/skate-metrics/` on first use.

### Ultralytics

YOLO pose is available for comparison. It is 2D-only (COCO 17 keypoints, no
heels/toes), so rotation counting is less reliable and the app warns about it.

### JSON

`--save-pose` writes the exact keypoint format accepted by `--pose json`. This is
used by the synthetic demo, tests, and anyone experimenting with another tracker.

## 5. Timestamps and frame rate

Phone slow-motion and HEVC files are frequently variable-frame-rate. Using
`frame_index / nominal_fps` silently corrupts flight time. `video.py` runs
`ffprobe` and reads `best_effort_timestamp_time` for every frame. If timestamps
are unavailable, it falls back to constant fps.

The synthetic demo is 60 fps; the validation harness compares it against a
240 fps rendering of the same motion.

## 6. Jump segmentation

### Primary: foot baseline

`kinematics.foot_height_series` takes the lowest visible foot point (max y among
ankles, heels and foot indices). It estimates the ice line as a rolling high
percentile (default 3 s window), then computes:

```
foot_height = ice_line(t) - lowest_foot_y(t)
```

Airborne candidates are runs above a threshold. Boundaries are refined with:

1. Linear interpolation at the threshold crossing.
2. Extrapolation to the ice line (`zero_crossing_time`).
3. A quadratic fit of the airborne arc when the fit is clean (`_parabola_flight`).

This is the most accurate method when the camera is fixed and the skater stays
large in frame.

### Fallback: body-centre excursion

When the foot baseline is unreliable — moving camera, zoom, the skater moving in
depth, or a missed planted foot — the foot detector may return no plausible
window. `detect_com_jumps` uses the body centre instead:

- Smooth the body-centre y with a 1.5 s Savitzky–Golay trend.
- `residual = trend - body_center_y`.
- A jump is a positive lobe above a threshold, 0.2–1.0 s wide.
- Takeoff/landing are the surrounding zero crossings.

The result is marked `method="com"` and the UI labels it **approximate**. It is
better than returning nothing for broadcast footage, but it is not a substitute
for a fixed camera.

### Why not just trust the foot signal?

The ground line in a projective image is not a constant. If the skater moves away
from the camera, their feet rise in the image even though they are still on the
ice. A global horizontal baseline is only valid for a fixed camera and a roughly
constant depth. The rolling baseline and CoM fallback are mitigations, not a
calibration.

## 7. Rotation

`rotation_signal` chooses the best available signal:

1. 3D shoulder yaw (`atan2(z_right - z_left, x_right - x_left)`).
2. 3D hip yaw.
3. 2D shoulder line.
4. 2D hip line.

Angles are unwrapped with `np.unwrap`. `unwrap_angles` also repairs impossible
single-frame jumps (> 2.5 rad), which are usually left/right label swaps or
tracking glitches rather than real motion.

Total rotation is the angle difference between takeoff and landing. Average rate
is `rotations / real_flight_time`. Peak rate is the maximum smoothed angular rate
inside the airborne window, excluding the first/last samples where the derivative
is transitioning between entry and flight.

### Playback speed

If a clip is slowed to 40%, video time is not real time:

- `real_flight = video_flight × playback_speed`
- `height = g × real_flight² / 8`
- `real_rate = video_rate / playback_speed`

The web app and CLI expose `playback_speed` for this. Forgetting it is the most
common way to get a spectacular but wrong height.

## 8. Metrics

| Metric | Formula / method |
| --- | --- |
| Flight time | `landing_time - takeoff_time`, corrected for playback speed |
| Height | `g × flight_time² / 8` |
| Rotations | `abs(angle_landing - angle_takeoff) / 2π` |
| Skating count | Rotations rounded to nearest 0.5 |
| Average rate | `rotations / flight_time` |
| Peak rate | Max smoothed angular rate in the airborne window |
| Direction | Sign of the total angle change |

Height does not require camera calibration because it comes from timing. That is
why it is the most robust metric in this pipeline.

## 9. Rendering

`overlay.py` produces two videos in one decode pass:

- `overlay.mp4` — skeleton on the source footage, metrics card, rotation graph,
  timeline.
- `skeleton.mp4` — clean dark studio background with the skeleton, trail,
  metrics and graph.

Both use H.264 (`avc1`). The first implementation used OpenCV's `mp4v`, which
produced files that Chrome/Safari would not play. The server also keeps a
`*.web.mp4` transcode cache for older analyses.

The report (`report.py`) is standalone HTML with inline CSS/SVG. It embeds the
overlay and skeleton videos, metric cards, a per-jump rotation-speed chart, and
the measurement notes.

## 10. Web app

`server.py` is a FastAPI app with a single background worker:

- Uploads are written to `out/web/<job_id>/input.*`.
- YouTube jobs are downloaded by `youtube.py` (yt-dlp).
- A job's status is persisted to `out/web/<job_id>/job.json`, so the session
  list survives restarts.
- The browser polls `/api/jobs/{id}` and renders the result.

The UI has three views: Analyze, Sessions and Guide. It also supports deep links
(`?job=<id>`, `?view=sessions`).

### Browser recording

`getUserMedia` + `MediaRecorder` with a request for 1920×1080 @ 60 fps. The UI
reports the actual track settings. iOS Safari may cap below 60 fps; camera access
also requires a secure context (HTTPS or localhost). Uploads work over plain
HTTP.

### Video serving

`/jobs/{id}/overlay.mp4` and `/skeleton.mp4` check the codec with ffprobe. If the
file is not browser-playable (for example an older `mp4v` result), they transcode
once to `*.web.mp4` and serve that.

## 11. Stabilization (experimental)

`stabilize.py` estimates a background homography with ORB features and writes a
stabilized clip. It is exposed as `--stabilize` and is **not enabled by default**.
On wide broadcast shots the homography can lock onto the wrong texture and push
the skater out of frame. It remains available for controlled experiments; the
CoM fallback is the safer default for moving-camera clips.

## 12. Validation

Three levels:

1. **Synthetic ground truth** — `synthetic.py` generates a skater with a known
   0.60 s flight and 1.5 revolutions. Tests assert the recovered values.
2. **60 vs 240 fps** — `validation.py` matches jumps by takeoff time and reports
   deltas. The synthetic comparison is within ~1.4 cm height and ~9 ms flight.
3. **Real rink footage** — still the missing piece. The capture protocol in
   `docs/CAPTURE.md` defines the session and acceptance targets (20 ms flight,
   3 cm height, 0.1 rotation, 0.2 rev/s average rate).

## 13. Testing

```bash
make test
```

- `tests/test_pipeline.py` — synthetic detection, metrics, playback speed, time
  ranges, one-foot false positives, camera drift, 60 vs 240.
- `tests/test_server.py` — full demo job through the web API, report/video/skeleton
  serving, session listing, deletion.
- `tests/test_youtube.py` — URL validation and API input validation.

16 tests; no network access required.

## 14. Deliberate trade-offs

| Decision | Why |
| --- | --- |
| Python core instead of browser-only | Reuses the validated CV stack; native/PWA can call the same logic later |
| Local server instead of cloud | Privacy for minors/rink footage, no upload latency, works offline |
| MediaPipe default | Fast on Apple Silicon, provides 3D world landmarks |
| CoM fallback instead of forced foot baseline | Produces something useful on moving-camera clips, clearly labelled approximate |
| No auto-stabilization | The experimental homography can fail badly; opt-in is safer |
| H.264 output | Browser playback is a product requirement, not an optimization |
| Height from flight time | No calibration needed; robust to camera position |

## 15. Next work

- Capture real 60 fps rink footage and run the validation harness.
- Native iPhone capture (AVFoundation + rolling buffer + Core ML) for guaranteed
  60 fps.
- iPad coach view over peer-to-peer.
- Skater-centred crop/zoom to improve pose on broadcast footage.
- Boot/blade keypoints, takeoff edge and under-rotation research.
- Session trends and cross-session comparison.
