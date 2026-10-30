"""Command-line entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import __version__
from .pipeline import AnalysisResult, analyze_video
from .segmentation import SegmentationConfig
from .synthetic import write_demo
from .validation import match_and_compare, summarize
from .youtube import YouTubeError, download_youtube, is_youtube_url


def _print_result(result: AnalysisResult) -> None:
    print(f"Source: {result.info.path}")
    print(
        f"Video: {result.info.width}x{result.info.height} @ {result.info.fps:.2f} fps, "
        f"{result.info.frame_count} frames"
    )
    print(f"Rotation signal: {result.rotation_source}")
    if result.stabilization and result.stabilization.get("applied"):
        print(
            "Stabilization: applied "
            f"(camera motion score {result.stabilization.get('motion_score', 0):.3f})"
        )
    if not result.jumps:
        print("No jumps detected.")
    for i, jump in enumerate(result.jumps, start=1):
        print(
            f"  Jump {i}: {jump.height_m:.2f} m | "
            f"{jump.half_turns:.1f} turns | "
            f"{jump.avg_rotation_rps:.2f} rev/s avg | "
            f"{jump.peak_rotation_rps:.2f} rev/s peak | "
            f"{jump.flight_time:.2f} s flight"
        )
    for warning in result.warnings:
        print(f"  Warning: {warning}")
    print(f"Report:  {result.report_path}")
    if result.overlay_path:
        print(f"Overlay: {result.overlay_path}")
    if result.skeleton_path:
        print(f"Skeleton: {result.skeleton_path}")
    if result.pose_path:
        print(f"Pose:    {result.pose_path}")
    print(f"Metrics: {result.metrics_path}")


def _print_comparison(pairs, summary) -> None:
    print("Reference vs measured")
    for i, pair in enumerate(pairs, start=1):
        print(
            f"  Jump {i}: "
            f"height Δ {pair.height_delta_m:+.3f} m | "
            f"flight Δ {pair.flight_delta_s:+.3f} s | "
            f"rotation Δ {pair.rotation_delta:+.3f} turns | "
            f"avg rate Δ {pair.avg_rate_delta_rps:+.3f} rev/s"
        )
    if not pairs:
        print("  No matching jumps.")
        return
    print(
        f"  Mean |height| {summary['mean_abs_height_delta_m']:.3f} m | "
        f"|flight| {summary['mean_abs_flight_delta_s']:.3f} s | "
        f"|rotation| {summary['mean_abs_rotation_delta']:.3f} turns | "
        f"|avg rate| {summary['mean_abs_avg_rate_delta_rps']:.3f} rev/s"
    )


def _add_analysis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video", type=Path, help="input video")
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--pose",
        choices=("mediapipe", "ultralytics", "json"),
        default="mediapipe",
    )
    parser.add_argument("--keypoints", type=Path, help="keypoints JSON for --pose json")
    parser.add_argument(
        "--model",
        choices=("lite", "full", "heavy"),
        default="lite",
        help="MediaPipe pose model size",
    )
    parser.add_argument("--start", type=float, help="start time in seconds")
    parser.add_argument("--end", type=float, help="end time in seconds")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument(
        "--skeleton",
        action="store_true",
        help="also write a skeleton-only video",
    )
    parser.add_argument(
        "--save-pose",
        action="store_true",
        help="write keypoints.json so metrics can be re-run with --pose json",
    )
    parser.add_argument("--min-flight", type=float, default=0.15)
    parser.add_argument("--max-flight", type=float, default=1.0)
    parser.add_argument("--threshold-px", type=float)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="1 = real time, 0.4 = 40%% speed slow motion",
    )
    parser.add_argument(
        "--stabilize",
        action="store_true",
        help="compensate for camera motion before analysis (experimental)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="skate-metrics",
        description=(
            "Markerless figure-skating jump analysis · "
            "developed by Ray Foo Productions · @edgeswithray"
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analyse a video")
    _add_analysis_args(analyze)

    demo = sub.add_parser("demo", help="run the synthetic end-to-end demo")
    demo.add_argument("--output", "-o", type=Path, default=Path("out/demo"))
    demo.add_argument("--duration", type=float, default=4.0)
    demo.add_argument("--fps", type=int, default=60)
    demo.add_argument("--takeoff", type=float, default=1.4)
    demo.add_argument("--flight", type=float, default=0.60)
    demo.add_argument("--turns", type=float, default=1.5)

    compare = sub.add_parser(
        "compare",
        help="compare 60 fps analysis against a high-frame-rate reference",
    )
    compare.add_argument("low", type=Path, help="low-frame-rate video (e.g. 60 fps)")
    compare.add_argument("high", type=Path, help="reference video (e.g. 240 fps)")
    compare.add_argument("--output", "-o", type=Path, required=True)
    compare.add_argument(
        "--pose",
        choices=("mediapipe", "ultralytics", "json"),
        default="mediapipe",
    )
    compare.add_argument("--low-keypoints", type=Path)
    compare.add_argument("--high-keypoints", type=Path)
    compare.add_argument("--model", choices=("lite", "full", "heavy"), default="lite")

    youtube = sub.add_parser("youtube", help="download and analyse a YouTube URL")
    youtube.add_argument("url")
    youtube.add_argument("--output", "-o", type=Path, required=True)
    youtube.add_argument(
        "--pose",
        choices=("mediapipe", "ultralytics"),
        default="mediapipe",
    )
    youtube.add_argument(
        "--cookies",
        choices=("safari", "chrome", "firefox", "edge", "brave"),
        help="read cookies from this browser for sign-in restricted videos",
    )
    youtube.add_argument("--start", type=float, help="analysis start in seconds")
    youtube.add_argument("--end", type=float, help="analysis end in seconds")
    youtube.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="1 = real time, 0.4 = 40%% speed slow motion",
    )
    youtube.add_argument("--skeleton", action="store_true")

    serve = sub.add_parser("serve", help="run the local web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--ssl-keyfile", type=Path, help="TLS key for camera access")
    serve.add_argument("--ssl-certfile", type=Path, help="TLS certificate")

    args = parser.parse_args(argv)

    if args.command == "demo":
        video, keypoints = write_demo(
            args.output,
            duration=args.duration,
            fps=args.fps,
            takeoff=args.takeoff,
            flight=args.flight,
            turns=args.turns,
        )
        result = analyze_video(
            video,
            args.output,
            pose_backend="json",
            keypoints_path=keypoints,
            write_skeleton=True,
        )
        _print_result(result)
        return 0

    if args.command == "youtube":
        if not is_youtube_url(args.url):
            raise SystemExit("not a YouTube URL")
        source_dir = args.output / "source"
        try:
            video, info = download_youtube(
                args.url,
                source_dir,
                cookies=args.cookies,
                cookiefile=os.environ.get("SKATE_YTDLP_COOKIES"),
                progress=print,
            )
        except YouTubeError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Downloaded: {info.get('title', video.name)}")
        result = analyze_video(
            video,
            args.output,
            pose_backend=args.pose,
            start=args.start,
            end=args.end,
            playback_speed=args.playback_speed,
            write_skeleton=args.skeleton,
        )
        _print_result(result)
        return 0

    if args.command == "serve":
        try:
            import uvicorn

            from .server import app
        except ImportError as exc:
            raise SystemExit(
                "web dependencies are missing; run `uv sync --extra web`"
            ) from exc
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            ssl_keyfile=str(args.ssl_keyfile) if args.ssl_keyfile else None,
            ssl_certfile=str(args.ssl_certfile) if args.ssl_certfile else None,
        )
        return 0

    if args.command == "compare":
        low = analyze_video(
            args.low,
            args.output / "low",
            pose_backend=args.pose,
            keypoints_path=args.low_keypoints,
            model=args.model,
            no_overlay=True,
        )
        high = analyze_video(
            args.high,
            args.output / "high",
            pose_backend=args.pose,
            keypoints_path=args.high_keypoints,
            model=args.model,
            no_overlay=True,
        )
        pairs = match_and_compare(high.jumps, low.jumps)
        _print_comparison(pairs, summarize(pairs))
        return 0

    if args.command == "analyze":
        segmentation = SegmentationConfig(
            min_flight_s=args.min_flight,
            max_flight_s=args.max_flight,
            threshold_px=args.threshold_px,
        )
        result = analyze_video(
            args.video,
            args.output,
            pose_backend=args.pose,
            keypoints_path=args.keypoints,
            model=args.model,
            start=args.start,
            end=args.end,
            no_overlay=args.no_overlay,
            write_skeleton=args.skeleton,
            save_pose=args.save_pose,
            segmentation=segmentation,
            playback_speed=args.playback_speed,
            stabilize=args.stabilize,
        )
        _print_result(result)
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
