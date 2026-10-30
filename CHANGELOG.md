# Changelog

## 0.1.0 — 2026-10-30

First working build.

### Analysis

- Video → pose → airborne window → metrics pipeline.
- MediaPipe Tasks backend with 2D + metric 3D world landmarks.
- Optional Ultralytics YOLO backend and JSON keypoint input.
- ffprobe presentation timestamps for variable-frame-rate phone footage.
- Foot-contact segmentation with rolling ice-line baseline, zero-crossing
  extrapolation and quadratic flight refinement.
- Body-centre fallback for moving-camera footage, labelled approximate.
- Height from flight time (`h = g·t²/8`).
- Rotation from 3D shoulder/hip yaw, with 2D fallback and unwrap repair.
- Playback-speed correction for slow-motion clips.
- Rotation profile, average and peak rev/s.

### Outputs

- `overlay.mp4` — skeleton over the source footage.
- `skeleton.mp4` — clean skeleton-only view.
- `report.html` — standalone coach report with rotation graph.
- `metrics.json`, `rotation.csv`, optional `keypoints.json`.
- H.264 output for browser playback.

### CLI

- `analyze` with range, playback speed, skeleton and pose export.
- `demo` with a deterministic synthetic skater.
- `compare` for 60 fps vs high-frame-rate validation.
- `youtube` with optional browser cookies.
- `serve` local web app.

### Web app

- Analyze, Sessions and Guide views.
- Drag-and-drop upload and browser camera recording.
- YouTube links with range, playback speed and browser-cookie options.
- Live job stages, metric tiles, overlay/skeleton toggle, rotation chart.
- Session persistence, reopen and delete.
- Deep links (`?job=`, `?view=`).
- Secure-context detection for camera capture.
- Friendly error for an unreachable server.

### Tests

- 16 tests covering synthetic metrics, 60 vs 240 fps, playback speed, ranges,
  one-foot false positives, camera drift, web API, and YouTube validation.

### Known limitations

- No real-rink validation yet.
- Single camera, single skater.
- No edge/toe-pick/under-rotation judging.
- Broadcast/moving-camera results are approximate.
- Camera stabilization is experimental and opt-in.
