# How this was built

A short build log: the decisions, the dead ends, and what is still unproven.

## Starting point

The brief was:

> A markerless app to analyse figure-skating jumps such as an Axel, for me and my
> coach. iPhone 16 Pro Max, single phone on a tripod, 60 fps, iPad available,
> feedback 2–5 seconds after landing. Must have height and rotation speed.

The first research pass compared Theia3D (commercial), Pose2Sim, FreeMoCap and
OpenCap. The conclusion was that the open-source ecosystem can do the pipeline —
multi-camera video → 2D pose → 3D triangulation → kinematics — but none of them
ships a turnkey figure-skating product. Theia3D is closed source. So this became a
small product rather than an integration.

## Why a Python core plus a web app

The obvious target is an iPhone app. But the hard part is not the UI; it is
deciding what the numbers mean and where the pipeline fails. Porting to Swift
before that is expensive.

So the build order became:

1. Python analysis core with a synthetic ground-truth demo.
2. CLI for offline experiments.
3. Local FastAPI web app for the coach-facing workflow.
4. Native iPhone/iPad layer once the metrics survive real footage.

This also means the web app, CLI and future native client share one metric
implementation.

## Milestones

### M1 — Analysis core

Built the frame model, pose backends, segmentation, rotation and metrics.

Key decisions:

- **Height from flight time** (`h = g·t²/8`), because it needs timing rather than
  calibration. This is the most robust number a single camera can produce.
- **MediaPipe 3D world landmarks** for rotation. A 2D shoulder line is only
  defined modulo 180°; when the shoulders go edge-on, a naive unwrap loses half a
  revolution.
- **ffprobe timestamps**, not `index / fps`, because phone slow-mo is often
  variable frame rate.
- **Sub-frame contact refinement**: threshold crossing, zero-line extrapolation,
  then a quadratic fit when the airborne arc is clean.

Verification: a synthetic skater with a known 0.60 s flight and 1.5 revolutions.
The pipeline recovers 0.44 m, 1.5 rev, 2.50 rev/s average, 0.60 s flight. The
60 fps vs 240 fps comparison stays within ~1.4 cm and ~9 ms.

### M2 — Overlay and report

Added the skeleton overlay, a standalone HTML report and a rotation-speed graph.
The first videos used OpenCV `mp4v`, which produced files the browser would not
play. Fixed by preferring `avc1` (H.264) and adding a one-time `*.web.mp4`
transcode cache for older results.

### M3 — Web app

FastAPI with a single background worker, job JSON persisted per session, and an
Analyze / Sessions / Guide interface. Added drag-and-drop upload, browser
recording, deep links, delete, and download links.

Browser recording needs a secure context. On localhost it works; on an iPad over
LAN it needs HTTPS. The UI detects this and explains it rather than failing with a
generic camera error.

### M4 — YouTube links

Added yt-dlp. The first attempts hit YouTube's “Sign in to confirm you're not a
bot” check, so the UI now offers browser-cookie import (Safari/Chrome/Firefox/
Edge/Brave) and a `SKATE_YTDLP_COOKIES` file path. Downloads are best-effort.

Two controls were added after testing real clips:

- **Analyse range** — isolate a 1–2 s section.
- **Playback speed** — `0.4` for a 40% slow-mo clip, otherwise height and flight
  time are scaled by the slow-motion factor.

### M5 — Moving-camera handle

Broadcast clips broke the foot baseline. The app originally returned “no plausible
jumps”. Added a fallback detector based on the body-centre excursion above a
slow trend. It finds a jump on moving-camera footage and labels the result
**approximate**.

Also renamed “turns” to **revolutions** in the UI after a confusing result. A
value of 0.5 means half a revolution, not one. A single Axel is 1.5 rev; a quad
Axel is 4.5 rev. The raw measured value is now shown alongside the rounded skating
count.

## Problems hit along the way

| Problem | Fix |
| --- | --- |
| MediaPipe 1.2 tasks API landmarks had no `.name` | Map by index with `vision.PoseLandmark(i).name` |
| Rotation rate was reported in rad/s but labelled rev/s | Divide by 2π |
| 2D shoulder line lost half a turn at edge-on poses | Prefer 3D world-landmark yaw |
| Peak rate spiked at the takeoff/landing transition | Exclude the first/last samples |
| Foot baseline broke when the camera moved | Rolling baseline + CoM fallback |
| Slow-mo clips inflated height | Explicit playback-speed correction |
| YouTube bot check | Browser-cookie option and cookies.txt support |
| `mp4v` videos were black in the browser | H.264 output and transcode cache |
| Server died when the terminal command timed out | Detached `nohup`/`disown` run, friendly fetch error |
| A quad Axel advertised 0.5 rev | Wide arena shot, skater ~20 px; marked approximate, docs updated |
| Browser “TypeError: Failed to fetch” | Caused by the server not running; clearer UI message |

## What is verified

- 16 automated tests, no network required.
- Synthetic ground truth for flight time, height, rotation count, average/peak
  rate, playback speed and time ranges.
- One-foot glide is not counted as a jump.
- Camera drift is detected.
- Full web API flow: demo job → report/video/skeleton → session list → delete.
- 60 fps vs 240 fps synthetic comparison.

## What is not verified yet

- Real 60 fps rink footage against a 240 fps reference and force plate/IMU.
- Triple and quad rotation accuracy.
- Takeoff edge (Lutz/Flip), toe-pick assist, under-rotation judging.
- Multi-person clips.
- Browser capture at a true 60 fps on iOS Safari.

The next honest step is not more UI. It is a rink session with the capture
protocol in [`CAPTURE.md`](CAPTURE.md), then comparing the output to 240 fps
ground truth.

## Lessons

- The timing model matters more than the pose model. Bad timestamps make every
  metric wrong.
- A single camera cannot measure what it cannot see. A wide arena shot of a
  20-pixel skater is not a measurement problem that software can solve.
- “Just works” and “honest” pull in opposite directions. The compromise is to
  produce the best available result, label it approximate, and keep the caveats
  visible.
