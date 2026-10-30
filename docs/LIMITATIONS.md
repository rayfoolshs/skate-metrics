# Limitations and accuracy

What the app can measure, what it cannot, and how to read the warnings.

## The short version

Skate Metrics is **coaching-grade**, not judging-grade. It measures jump height,
rotation count and rotation speed from a single camera. It does not measure edges,
toe-pick assist, under-rotation or 3D joint angles.

For trustworthy numbers:

- fixed camera
- skater large in the frame
- real-time playback speed declared correctly
- full body and the ice visible for the whole jump
- 60 fps minimum

## Capture requirements

| Requirement | Why |
| --- | --- |
| Fixed camera | A moving ice line breaks the ground estimate |
| Skater ≥ 40% of frame height | Pose and rotation need pixels |
| Full body + ice in frame | Takeoff/landing detection needs both |
| 60 fps or better | A triple rotates ~5 rev/s; 30 fps is too coarse |
| Known playback speed | Slow motion scales flight time and height |
| One skater | Multi-person tracking is not implemented |

Broadcast wide shots, arena cameras and montages are usually **not** suitable for
height/rotation. They are useful for pose, skeleton and jump classification.

## Accuracy expectations

| Metric | 60 fps expectation |
| --- | --- |
| Flight time | ~±1 frame raw; sub-frame fit improves it |
| Height | roughly ±2–3 cm on a clean fixed-camera clip |
| Rotation count | reliable for singles/doubles; triples noisier |
| Average rotation speed | within a few percent |
| Peak rotation speed | sensitive to motion blur and pose jitter |
| Direction | reliable if the rotation signal is clean |

These are expectations, not guarantees. Real validation against 240 fps and a
force plate/IMU is still pending.

## Rotation is in revolutions

- `0.5` = half a revolution
- `1.5` = single Axel
- `2.5` = double Axel
- `3.5` = triple Axel
- `4.5` = quad Axel

The app also shows a rounded “skating count” to the nearest 0.5. A value of `0.5`
does **not** mean one revolution.

If a quad Axel reports `0.5`, that is a failed measurement, not a quad. Check the
warning, the skater size and the camera.

## Moving-camera fallback

When the foot baseline is unreliable, the app uses the body-centre excursion
(`method: "com"`). This can find a jump in moving-camera footage, but:

- timing is approximate
- height is approximate
- rotation may undercount
- the result is labelled **approximate**

Use a fixed camera or a 1–2 s range around the jump for better results.

## Slow motion

Set playback speed explicitly:

- `0.4` for “40% speed”
- `0.5` for half speed
- `0.25` for quarter speed
- `1.0` for real time

Forgetting this inflates flight time and height. A 40% clip analysed as real time
will report a flight time 2.5× too long and a height 6.25× too high (and may be
filtered as implausible).

## 2D vs 3D rotation

MediaPipe provides 3D world landmarks, which are used for rotation. If a backend
only provides 2D keypoints (YOLO, or custom JSON without world points), a
projected shoulder line is ambiguous at edge-on poses and can undercount full
revolutions. The app warns when the 2D signal is used.

## Camera stabilization

`--stabilize` is experimental. It estimates a background homography with ORB
features. On wide broadcast shots it can lock onto the wrong texture and push the
skater out of frame. It is opt-in for that reason.

## YouTube

- Downloads are best-effort and subject to YouTube's terms.
- Some videos need browser cookies or a `cookies.txt`.
- Broadcast edits, cuts, zooms and title cards remain in the clip and confuse
  segmentation.
- Only download content you have the right to use.

## Privacy

All processing is local by default. Uploaded videos and results stay under
`out/web/`. The app does not call a cloud analysis service. If you use YouTube
links, the server contacts YouTube to download the clip.

Rink footage often includes minors. Get consent, limit retention, and delete
sessions you no longer need (`Sessions → ×`).

## What would make it substantially better

1. Real rink validation against 240 fps ground truth.
2. Native 60 fps iPhone capture with a rolling buffer.
3. Skater-centred crop/zoom for broadcast footage.
4. Boot/blade keypoints for edge and toe-pick detection.
5. Multi-camera 3D for depth and joint angles.
