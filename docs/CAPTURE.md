# Capture protocol (M0)

The analysis is only as good as the footage. Run this protocol before tuning the
software.

## Hardware

- iPhone 16 Pro Max on a tripod at the boards.
- 1080p60 or 4K60 for the live/normal clip.
- 1080p240 or 4K120 slow-mo for the reference clip.
- Enough storage for 4K60 (roughly 400 MB/minute) and a power bank in the cold.

## Camera position

- Perpendicular to the jump's travel direction, at boards height.
- Frame the skater from the ice to ~1 m above the head at the jump apex.
- The skater should occupy at least 40% of the frame height.
- Fixed camera. No pan, no zoom, no digital stabilization that crops.
- Front lighting on the skater; avoid shooting into bright boards or windows.

## Session recipe

Record **10–20 attempts** of the same jump, then repeat for another jump type.
For each attempt:

1. Stand still for about 1 second with both skates on the ice.
2. Perform the jump.
3. Hold the landing for 1 second.
4. Reset to the same spot.

The still frames at the start/end give the ice-line estimate and the rotation
baseline. Keeping the same spot makes the clips comparable.

## Reference clips

For at least five attempts, also record the same jump at 240 fps. Do not change
the tripod between the 60 fps and 240 fps clips.

Create a 60 fps copy of the 240 fps reference with:

```bash
ffmpeg -i high240.mp4 -vf "fps=60" -c:v libx264 -crf 18 low60.mp4
```

Then compare the two analyses:

```bash
skate-metrics compare low60.mp4 high240.mp4 --output out/validation --pose mediapipe
```

## Naming

```
sessions/2026-11-02/
  axel_01_60.mp4
  axel_01_240.mp4
  axel_02_60.mp4
  ...
  notes.md
```

Record jump type, skater, camera position and any fall/occlusion in `notes.md`.
Occluded attempts should be excluded from validation, not silently averaged.

## What to check before trusting a clip

- Full body visible for the whole jump (including the landing foot).
- No camera movement.
- No other person crossing in front of the skater.
- Ice line visible near the landing.
- Exposure high enough that the boots are not black blobs.

## Acceptance targets for M1

- Flight time within 20 ms of the 240 fps reference.
- Height within 3 cm.
- Rotation count within 0.1 turns for singles/doubles.
- Average rotation rate within 0.2 rev/s.
- The jump is detected exactly once.

If 60 fps misses these on triples, the live mode should fall back to 120 fps
(or accept a slower post-jump slow-mo analysis) before moving to the iOS app.
