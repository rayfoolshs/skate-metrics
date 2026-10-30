"""Local web app for Skate Metrics.

The browser uploads a clip (or records one with getUserMedia); a single local
worker runs the same Python analysis as the CLI and the UI polls for status.
Jobs are persisted under ``out/web/<id>/job.json`` so the session list survives
a server restart.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .pipeline import analyze_video
from .synthetic import write_demo
from .youtube import YouTubeError, download_youtube, is_youtube_url, normalize_url

BASE_OUT = Path(os.environ.get("SKATE_METRICS_OUT", "out/web")).resolve()
EXECUTOR = ThreadPoolExecutor(max_workers=1)
LOCK = threading.Lock()

ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
ALLOWED_BROWSERS = {"safari", "chrome", "firefox", "edge", "brave"}


@dataclass
class Job:
    id: str
    status: str = "queued"
    stage: str = "Queued"
    error: str | None = None
    source: str = ""
    kind: str = "upload"
    output_dir: str = ""
    jumps: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rotation_source: str = ""
    fps: float = 0.0
    width: int = 0
    height: int = 0
    stabilization: dict | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class YouTubeRequest(BaseModel):
    url: str
    pose: str = "mediapipe"
    cookies: str | None = None
    start: float | None = None
    end: float | None = None
    playback_speed: float = 1.0


JOBS: dict[str, Job] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_jobs()
    yield


app = FastAPI(title="Skate Metrics", version="0.1.0", lifespan=lifespan)


def _job_dir(job_id: str) -> Path:
    if not job_id.isalnum() or len(job_id) > 64:
        raise HTTPException(404, "job not found")
    return BASE_OUT / job_id


def _browser_video(path: Path) -> Path:
    """Return an H.264 version for browsers, transcoding older mp4v files once."""
    if not path.exists():
        return path
    cached = path.with_name(path.stem + ".web.mp4")
    if cached.exists() and cached.stat().st_mtime >= path.stat().st_mtime:
        return cached
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if probe.stdout.strip() in {"h264", "vp9", "av1"}:
            return path
        temp = path.with_name(path.stem + ".tmp.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temp),
            ],
            check=True,
        )
        temp.replace(cached)
        return cached
    except (OSError, subprocess.SubprocessError):
        return path


def _save_job(job: Job) -> None:
    directory = _job_dir(job.id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "job.json").write_text(json.dumps(asdict(job), indent=2))


def _load_jobs() -> None:
    """Reload persisted jobs from disk (called at startup and by tests)."""
    with LOCK:
        JOBS.clear()
        if not BASE_OUT.exists():
            return
        for path in BASE_OUT.glob("*/job.json"):
            try:
                data = json.loads(path.read_text())
                JOBS[data["id"]] = Job(**data)
            except (OSError, ValueError, KeyError, TypeError):
                continue


def _update(job_id: str, **changes) -> None:
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        for key, value in changes.items():
            setattr(job, key, value)
        _save_job(job)


def _job_payload(job: Job) -> dict:
    payload = asdict(job)
    payload["jump_count"] = len(job.jumps)
    payload["max_height_m"] = (
        max((jump.get("height_m", 0.0) for jump in job.jumps), default=0.0)
    )
    if job.status == "done":
        payload["report_url"] = f"/jobs/{job.id}/report"
        payload["overlay_url"] = f"/jobs/{job.id}/overlay.mp4"
        payload["metrics_url"] = f"/jobs/{job.id}/metrics.json"
        payload["csv_url"] = f"/jobs/{job.id}/rotation.csv"
        if (Path(job.output_dir) / "skeleton.mp4").exists():
            payload["skeleton_url"] = f"/jobs/{job.id}/skeleton.mp4"
    return payload


def _job_summary(job: Job) -> dict:
    """Lighter payload for the session list (no per-frame rotation profiles)."""
    payload = _job_payload(job)
    payload.pop("jumps", None)
    payload.pop("warnings", None)
    payload.pop("rotation_source", None)
    return payload


def _run_analysis(
    job_id: str,
    video_path: Path,
    *,
    pose_backend: str,
    start: float | None = None,
    end: float | None = None,
    playback_speed: float = 1.0,
) -> None:
    output_dir = _job_dir(job_id) / "analysis"
    try:
        _update(job_id, status="running", stage="Starting analysis")
        result = analyze_video(
            video_path,
            output_dir,
            pose_backend=pose_backend,
            start=start,
            end=end,
            playback_speed=playback_speed,
            write_skeleton=True,
            save_pose=False,
            progress=lambda stage: _update(job_id, stage=stage),
        )
        _update(
            job_id,
            status="done",
            stage="Done",
            jumps=[jump.to_dict() for jump in result.jumps],
            warnings=result.warnings,
            rotation_source=result.rotation_source,
            fps=result.info.fps,
            width=result.info.width,
            height=result.info.height,
            stabilization=result.stabilization,
            output_dir=str(output_dir),
        )
    except Exception as exc:  # surfaced to the browser instead of a hung job
        _update(job_id, status="error", stage="Failed", error=str(exc))


def _run_demo(job_id: str) -> None:
    source_dir = _job_dir(job_id) / "source"
    try:
        _update(job_id, status="running", stage="Building synthetic skater")
        video, keypoints = write_demo(source_dir, fps=60)
        output_dir = _job_dir(job_id) / "analysis"
        result = analyze_video(
            video,
            output_dir,
            pose_backend="json",
            keypoints_path=keypoints,
            write_skeleton=True,
            save_pose=False,
            progress=lambda stage: _update(job_id, stage=stage),
        )
        _update(
            job_id,
            status="done",
            stage="Done",
            jumps=[jump.to_dict() for jump in result.jumps],
            warnings=result.warnings,
            rotation_source=result.rotation_source,
            fps=result.info.fps,
            width=result.info.width,
            height=result.info.height,
            stabilization=result.stabilization,
            output_dir=str(output_dir),
        )
    except Exception as exc:
        _update(job_id, status="error", stage="Failed", error=str(exc))


def _run_youtube(
    job_id: str,
    url: str,
    *,
    pose_backend: str,
    cookies: str | None,
    start: float | None = None,
    end: float | None = None,
    playback_speed: float = 1.0,
) -> None:
    source_dir = _job_dir(job_id) / "source"
    try:
        _update(job_id, status="running", stage="Downloading from YouTube")
        cookiefile = os.environ.get("SKATE_YTDLP_COOKIES")
        video, info = download_youtube(
            url,
            source_dir,
            cookies=cookies,
            cookiefile=cookiefile,
            progress=lambda stage: _update(job_id, stage=stage),
        )
        title = (info.get("title") or url).strip()
        _update(job_id, source=title, stage="Analysing downloaded video")
        output_dir = _job_dir(job_id) / "analysis"
        result = analyze_video(
            video,
            output_dir,
            pose_backend=pose_backend,
            start=start,
            end=end,
            playback_speed=playback_speed,
            write_skeleton=True,
            save_pose=False,
            progress=lambda stage: _update(job_id, stage=stage),
        )
        _update(
            job_id,
            status="done",
            stage="Done",
            jumps=[jump.to_dict() for jump in result.jumps],
            warnings=result.warnings,
            rotation_source=result.rotation_source,
            fps=result.info.fps,
            width=result.info.width,
            height=result.info.height,
            stabilization=result.stabilization,
            output_dir=str(output_dir),
        )
    except YouTubeError as exc:
        _update(job_id, status="error", stage="Failed", error=str(exc))
    except Exception as exc:
        _update(job_id, status="error", stage="Failed", error=str(exc))


def _new_job(source: str, kind: str) -> Job:
    job_id = uuid.uuid4().hex[:12]
    job = Job(
        id=job_id,
        source=source,
        kind=kind,
        output_dir=str(_job_dir(job_id) / "analysis"),
    )
    with LOCK:
        JOBS[job_id] = job
        _save_job(job)
    return job


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/health")
def health() -> dict:
    return {"ok": True, "jobs": len(JOBS)}


@app.get("/api/jobs")
def list_jobs() -> JSONResponse:
    with LOCK:
        jobs = sorted(JOBS.values(), key=lambda job: job.created_at, reverse=True)
    return JSONResponse([_job_summary(job) for job in jobs[:100]])


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    pose: str = Form("mediapipe"),
    start: float | None = Form(None),
    end: float | None = Form(None),
    playback_speed: float = Form(1.0),
) -> JSONResponse:
    name = Path(file.filename or "upload.mp4").name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported video type: {suffix or 'unknown'}")
    if pose not in {"mediapipe", "ultralytics"}:
        raise HTTPException(400, "pose must be mediapipe or ultralytics")
    if start is not None and end is not None and end <= start:
        raise HTTPException(400, "end must be after start")
    if not 0.05 <= playback_speed <= 2.0:
        raise HTTPException(400, "playback speed must be between 0.05 and 2.0")

    job = _new_job(name, "upload")
    job_dir = _job_dir(job.id)
    job_dir.mkdir(parents=True, exist_ok=True)
    destination = job_dir / f"input{suffix}"
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    file.file.close()

    if destination.stat().st_size == 0:
        raise HTTPException(400, "empty upload")

    EXECUTOR.submit(
        _run_analysis,
        job.id,
        destination,
        pose_backend=pose,
        start=start,
        end=end,
        playback_speed=playback_speed,
    )
    return JSONResponse({"job_id": job.id})


@app.post("/api/youtube")
def youtube(request: YouTubeRequest) -> JSONResponse:
    if not is_youtube_url(request.url):
        raise HTTPException(400, "not a YouTube URL")
    if request.pose not in {"mediapipe", "ultralytics"}:
        raise HTTPException(400, "pose must be mediapipe or ultralytics")
    if request.cookies not in ALLOWED_BROWSERS | {None}:
        raise HTTPException(400, "unsupported browser for cookies")
    if (
        request.start is not None
        and request.end is not None
        and request.end <= request.start
    ):
        raise HTTPException(400, "end must be after start")
    if not 0.05 <= request.playback_speed <= 2.0:
        raise HTTPException(400, "playback speed must be between 0.05 and 2.0")

    url = normalize_url(request.url)
    job = _new_job(url, "youtube")
    EXECUTOR.submit(
        _run_youtube,
        job.id,
        url,
        pose_backend=request.pose,
        cookies=request.cookies,
        start=request.start,
        end=request.end,
        playback_speed=request.playback_speed,
    )
    return JSONResponse({"job_id": job.id})


@app.post("/api/demo")
def demo() -> JSONResponse:
    job = _new_job("Synthetic demo", "demo")
    EXECUTOR.submit(_run_demo, job.id)
    return JSONResponse({"job_id": job.id})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    with LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return JSONResponse(_job_payload(job))


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> JSONResponse:
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.status in {"running", "queued"}:
            raise HTTPException(409, "cannot delete a running or queued job")
        del JOBS[job_id]
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    return JSONResponse({"deleted": job_id})


@app.get("/jobs/{job_id}/report", response_class=FileResponse)
def job_report(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "analysis" / "report.html"
    if not path.exists():
        raise HTTPException(404, "report not ready")
    return FileResponse(path, media_type="text/html")


@app.get("/jobs/{job_id}/overlay.mp4")
def job_overlay(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "analysis" / "overlay.mp4"
    if not path.exists():
        raise HTTPException(404, "overlay not ready")
    return FileResponse(_browser_video(path), media_type="video/mp4")


@app.get("/jobs/{job_id}/skeleton.mp4")
def job_skeleton(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "analysis" / "skeleton.mp4"
    if not path.exists():
        raise HTTPException(404, "skeleton video not ready")
    return FileResponse(_browser_video(path), media_type="video/mp4")


@app.get("/jobs/{job_id}/metrics.json")
def job_metrics(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "analysis" / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "metrics not ready")
    return FileResponse(path, media_type="application/json")


@app.get("/jobs/{job_id}/rotation.csv")
def job_csv(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "analysis" / "rotation.csv"
    if not path.exists():
        raise HTTPException(404, "csv not ready")
    return FileResponse(path, media_type="text/csv")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="author" content="Ray Foo Productions">
<title>Skate Metrics · Ray Foo Productions</title>
<style>
  :root { color-scheme: dark; --amber:#f0a020; --teal:#00c2b2; --panel:#181818;
          --line:#2a2a2a; --muted:#8a8a8a; --bg:#0e0e0e; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:#eee;
         font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .app { display:grid; grid-template-columns:230px 1fr; min-height:100vh; }
  aside { background:#121212; border-right:1px solid var(--line); padding:22px 14px;
          display:flex; flex-direction:column; gap:18px; position:sticky; top:0; height:100vh; }
  .brand { display:flex; align-items:center; gap:10px; padding:0 6px; }
  .brand-mark { width:30px; height:30px; border-radius:50%; border:2px solid var(--amber);
                position:relative; flex:0 0 auto; }
  .brand-mark:after { content:""; position:absolute; inset:7px; border-radius:50%;
                      border:2px solid var(--teal); }
  .brand h1 { font-size:16px; margin:0; letter-spacing:.03em; }
  .brand span { display:block; font-size:10px; color:var(--muted); text-transform:uppercase;
                letter-spacing:.14em; margin-top:2px; }
  nav { display:flex; flex-direction:column; gap:4px; }
  nav button { text-align:left; background:transparent; color:#aaa; border:0; border-radius:8px;
               padding:9px 12px; font:inherit; cursor:pointer; }
  nav button:hover { background:#1d1d1d; color:#eee; }
  nav button.active { background:#242424; color:#fff; box-shadow:inset 2px 0 0 var(--amber); }
  .side-note { margin-top:auto; font-size:11px; color:#666; padding:0 8px; line-height:1.5; }
  .credit { margin-top:12px; padding-top:10px; border-top:1px solid #242424; }
  .credit a, .site-footer a { color:#999; text-decoration:none; }
  .credit a:hover, .site-footer a:hover { color:var(--amber); }
  .site-footer { margin-top:44px; padding-top:14px; border-top:1px solid var(--line);
                 color:#666; font-size:11px; letter-spacing:.02em; }
  .site-footer strong { color:#aaa; font-weight:500; }
  main { min-width:0; padding:30px 28px 70px; max-width:1180px; }
  .topline { display:flex; align-items:flex-end; justify-content:space-between; gap:14px;
             flex-wrap:wrap; margin-bottom:20px; }
  .topline > div { min-width:0; }
  .topline h2 { margin:0; font-size:26px; letter-spacing:.01em; }
  .topline p { margin:4px 0 0; color:var(--muted); font-size:13px; }
  .grid2 { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:16px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px;
          min-width:0; }
  .card h3 { margin:0 0 4px; font-size:14px; letter-spacing:.08em; text-transform:uppercase; color:#bbb; }
  .card .sub { color:var(--muted); font-size:12px; margin:0 0 14px; }
  .drop { border:1.5px dashed #3a3a3a; border-radius:12px; padding:26px 16px; text-align:center;
          color:#999; cursor:pointer; transition:.15s; }
  .drop:hover, .drop.over { border-color:var(--amber); color:#ddd; background:#1c1a14; }
  .drop strong { display:block; color:#ddd; margin-bottom:4px; }
  input[type=file] { display:none; }
  input[type=url], input[type=text], input[type=number] { width:100%; padding:9px 10px;
    border-radius:8px; background:#101010; color:#eee; border:1px solid #333; font:inherit; }
  .range-row { display:flex; align-items:center; gap:8px; margin-top:6px; }
  .range-row span { color:#888; font-size:12px; }
  .wide-card { margin-top:16px; }
  label.field { display:block; font-size:11px; color:#999; text-transform:uppercase;
                letter-spacing:.09em; margin:14px 0 5px; }
  select { width:100%; max-width:100%; padding:9px 10px; border-radius:8px; background:#101010;
           color:#eee; border:1px solid #333; font:inherit; }
  button.primary { background:var(--amber); color:#151515; border:0; border-radius:9px;
                   padding:10px 15px; font:inherit; font-weight:650; cursor:pointer; }
  button.secondary { background:#262626; color:#eee; border:1px solid #3a3a3a; border-radius:9px;
                     padding:10px 15px; font:inherit; cursor:pointer; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .row { display:flex; gap:9px; flex-wrap:wrap; margin-top:14px; }
  video { width:100%; max-width:100%; border-radius:12px; background:#000; display:block; }
  .video-tabs { display:flex; gap:6px; margin:14px 0 8px; }
  .video-tab { background:#1d1d1d; color:#aaa; border:1px solid #333; border-radius:7px;
    padding:6px 11px; font:inherit; font-size:12px; cursor:pointer; }
  .video-tab.active { background:#2a2315; color:var(--amber); border-color:#5a4218; }
  #preview { aspect-ratio:16/9; object-fit:cover; }
  .note { color:var(--muted); font-size:12px; margin:9px 0 0; }
  .status { display:none; margin-top:18px; background:#141414; border:1px solid var(--line);
            border-radius:14px; padding:16px; }
  .status.visible { display:block; }
  .stage { display:flex; align-items:center; gap:12px; }
  .spinner { width:17px; height:17px; border:2px solid #444; border-top-color:var(--amber);
             border-radius:50%; animation:spin .8s linear infinite; flex:0 0 auto; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .stage-title { font-weight:600; }
  .steps { display:flex; gap:6px; margin-top:12px; flex-wrap:wrap; }
  .step { font-size:10px; color:#666; border:1px solid #2e2e2e; border-radius:999px; padding:2px 8px; }
  .step.done { color:var(--teal); border-color:#1d4a46; }
  .error { color:#ff8a80; white-space:pre-wrap; margin-top:10px; }
  .warn { border:1px solid #6b4a16; background:#221a0d; color:#f0c674; border-radius:10px;
          padding:10px 14px; margin:14px 0; font-size:13px; }
  .warn summary { cursor:pointer; font-weight:600; }
  .warn ul { margin:6px 0 0; padding-left:18px; }
  .result-head { display:flex; align-items:center; justify-content:space-between; gap:12px;
                 flex-wrap:wrap; margin:20px 0 10px; }
  .pill { font-size:11px; color:var(--amber); border:1px solid #5a4218; border-radius:999px;
          padding:2px 9px; }
  .jump-grid { display:grid; gap:12px; }
  .jump-block { background:#111; border:1px solid #262626; border-radius:11px; padding:13px 15px; }
  .jump-head { display:flex; justify-content:space-between; align-items:center; gap:10px;
               margin-bottom:10px; }
  .jump-head strong { font-size:14px; }
  .stat-grid { display:grid; grid-template-columns:repeat(5, 1fr); gap:10px; }
  .stat { background:#161616; border:1px solid #242424; border-radius:8px; padding:9px 11px; }
  .stat .v { font-size:22px; font-variant-numeric:tabular-nums; }
  .stat .v small { font-size:11px; color:#999; margin-left:3px; }
  .stat .k { display:block; font-size:10px; color:#888; text-transform:uppercase;
             letter-spacing:.08em; margin-top:3px; }
  .charts { display:grid; gap:12px; margin-top:14px; }
  .chart { background:#141414; border:1px solid var(--line); border-radius:12px; padding:12px; }
  .chart h4 { margin:0 0 6px; font-size:12px; color:#aaa; font-weight:500; }
  .actions { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }
  .actions a { color:var(--amber); font-size:13px; text-decoration:none; border:1px solid #5a4218;
               border-radius:8px; padding:7px 11px; }
  .actions a:hover { background:#221a0d; }
  .sessions { display:grid; gap:10px; }
  .session { display:flex; align-items:center; gap:14px; background:var(--panel); border:1px solid var(--line);
             border-radius:12px; padding:13px 16px; cursor:pointer; transition:.15s; }
  .session:hover { border-color:#3d3d3d; background:#1c1c1c; }
  .session .dot { width:8px; height:8px; border-radius:50%; background:#555; flex:0 0 auto; }
  .session .dot.done { background:var(--teal); }
  .session .dot.error { background:#ff6b5f; }
  .session .dot.running { background:var(--amber); animation:pulse 1s infinite; }
  @keyframes pulse { 50% { opacity:.35; } }
  .session .name { flex:1; min-width:0; }
  .session .name strong { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .session .name span { font-size:11px; color:var(--muted); }
  .session .metric { text-align:right; font-size:12px; color:#bbb; }
  .session .metric b { font-size:15px; }
  .session .trash { background:transparent; border:0; color:#666; font-size:16px; cursor:pointer; }
  .session .trash:hover { color:#ff8a80; }
  .empty { color:var(--muted); text-align:center; padding:36px 10px; border:1px dashed #333; border-radius:12px; }
  .guide h3 { margin:22px 0 8px; font-size:17px; }
  .guide h3:first-child { margin-top:0; }
  .guide ul { margin:8px 0; padding-left:20px; color:#ccc; }
  .guide li { margin:5px 0; }
  .guide code { background:#1d1d1d; border:1px solid #333; border-radius:4px; padding:1px 5px; font-size:12px; }
  .kbd { color:#ddd; }
  [hidden] { display:none !important; }
  @media (max-width: 820px) {
    body { overflow-x:hidden; }
    .app { grid-template-columns:minmax(0, 1fr); }
    aside { position:static; height:auto; flex-direction:row; flex-wrap:wrap; align-items:center;
            padding:12px 14px; gap:10px 14px; }
    .brand span { display:none; }
    nav { flex-direction:row; flex-wrap:wrap; margin-left:0; width:100%; }
    nav button { padding:8px 10px; font-size:13px; }
    .side-note { display:none; }
    main { padding:20px 16px 60px; width:100%; max-width:100%; }
    .grid2 { grid-template-columns:minmax(0, 1fr); }
    .topline h2 { font-size:22px; }
    .topline p { max-width:100%; }
    .stat-grid { grid-template-columns:repeat(2, 1fr); }
    .row button { flex:1 1 auto; }
  }
</style>
</head>
<body>
<div class="app">
  <aside>
    <div class="brand">
      <div class="brand-mark"></div>
      <h1>Skate Metrics<span>Ray Foo Productions</span></h1>
    </div>
    <nav>
      <button data-view="analyze" class="active">Analyze</button>
      <button data-view="sessions">Sessions</button>
      <button data-view="guide">Guide</button>
    </nav>
    <div class="side-note">
      Runs locally on this machine.<br>Fixed camera · 60 fps.<br>Coaching-grade estimates.
      <div class="credit">
        Developed by <a href="https://www.rayfooproductions.com" target="_blank" rel="noopener">rayfooproductions.com</a><br>
        <a href="https://www.instagram.com/edgeswithray/" target="_blank" rel="noopener">@edgeswithray</a>
      </div>
    </div>
  </aside>

  <main>
    <section id="view-analyze">
      <div class="topline">
        <div>
          <h2>Jump analysis</h2>
          <p>Height, rotation count and rotation speed from a single fixed camera.</p>
        </div>
        <button class="secondary" id="demo">Try the demo</button>
      </div>

      <div class="grid2">
        <section class="card">
          <h3>Upload a clip</h3>
          <p class="sub">MP4, MOV, WebM. 60 fps or 240 fps slow-mo.</p>
          <div class="drop" id="drop">
            <strong id="drop-title">Drop a video here</strong>
            <span id="drop-sub">or click to choose a file</span>
          </div>
          <input id="file" type="file" accept="video/*">
          <label class="field" for="pose">Pose model</label>
          <select id="pose">
            <option value="mediapipe">MediaPipe · 3D landmarks (recommended)</option>
            <option value="ultralytics">YOLO · 2D (less reliable rotation)</option>
          </select>
          <label class="field">Analyse range in seconds (optional)</label>
          <div class="range-row">
            <input id="upload-start" type="number" min="0" step="0.1" placeholder="start">
            <span>to</span>
            <input id="upload-end" type="number" min="0" step="0.1" placeholder="end">
          </div>
          <label class="field" for="upload-speed">Playback speed (1 = real time, 0.4 = 40%)</label>
          <input id="upload-speed" type="number" min="0.05" max="2" step="0.05" value="1">
          <div class="row">
            <button class="primary" id="analyse">Analyse clip</button>
            <button class="secondary" id="clear-file">Clear</button>
          </div>
          <p class="note">Fixed camera. Full body and ice in frame. No panning.</p>
        </section>

        <section class="card">
          <h3>Record on this device</h3>
          <p class="sub">Browser recording; frame rate is reported after starting.</p>
          <video id="preview" playsinline muted></video>
          <div class="row">
            <button class="primary" id="start-camera">Start camera</button>
            <button class="secondary" id="record" disabled>Record jump</button>
            <button class="secondary" id="stop" disabled>Stop &amp; analyse</button>
          </div>
          <p class="note" id="camera-note">The browser may cap capture below 60 fps.</p>
        </section>
      </div>

      <section class="card wide-card">
        <h3>Analyse a YouTube link</h3>
        <p class="sub">Public videos only. Some videos need browser cookies to download.</p>
        <label class="field" for="youtube-url">YouTube URL</label>
        <input id="youtube-url" type="url" placeholder="https://www.youtube.com/watch?v=…">
        <label class="field">Analyse range in seconds (optional)</label>
        <div class="range-row">
          <input id="youtube-start" type="number" min="0" step="0.1" placeholder="start">
          <span>to</span>
          <input id="youtube-end" type="number" min="0" step="0.1" placeholder="end">
        </div>
        <label class="field" for="youtube-speed">Playback speed (1 = real time, 0.4 = 40%)</label>
        <input id="youtube-speed" type="number" min="0.05" max="2" step="0.05" value="1">
        <div class="row">
          <select id="youtube-cookies" style="max-width:220px">
            <option value="">No cookies</option>
            <option value="safari">Use Safari cookies</option>
            <option value="chrome">Use Chrome cookies</option>
            <option value="firefox">Use Firefox cookies</option>
            <option value="edge">Use Edge cookies</option>
            <option value="brave">Use Brave cookies</option>
          </select>
          <button class="primary" id="analyse-youtube">Analyse link</button>
        </div>
        <p class="note">Only download videos you have the right to use. YouTube's terms may prohibit downloading.</p>
      </section>

      <section class="status" id="status">
        <div class="stage">
          <div class="spinner" id="spinner"></div>
          <div>
            <div class="stage-title" id="stage-title">Queued</div>
            <div class="note" id="stage-sub"></div>
          </div>
        </div>
        <div class="steps" id="steps"></div>
        <div class="error" id="error"></div>
        <div id="warnings"></div>
        <div id="result"></div>
      </section>
    </section>

    <section id="view-sessions" hidden>
      <div class="topline">
        <div>
          <h2>Sessions</h2>
          <p>Past analyses on this machine. Click one to reopen it.</p>
        </div>
        <button class="secondary" id="refresh-sessions">Refresh</button>
      </div>
      <div class="sessions" id="sessions"></div>
    </section>

    <section id="view-guide" hidden class="guide">
      <div class="topline">
        <div>
          <h2>Capture guide</h2>
          <p>What makes a clip trustworthy.</p>
        </div>
      </div>
      <div class="card">
        <h3>Filming</h3>
        <ul>
          <li><strong>Fixed camera, no pan or zoom.</strong> A moving camera breaks the ice-line estimate.</li>
          <li><strong>Tripod at the boards, perpendicular to the jump.</strong> Full body from skates to ~1 m above the head at the apex.</li>
          <li><strong>60 fps minimum.</strong> 120 or 240 fps is better for triples; use slow-mo for reference clips.</li>
          <li><strong>Fill the frame.</strong> The skater should be at least 40% of the frame height.</li>
          <li><strong>Front lighting.</strong> Avoid shooting into bright boards or windows.</li>
        </ul>
        <h3>Metrics</h3>
        <ul>
          <li><strong>Height</strong> comes from flight time: <code>h = g·t²/8</code>. It does not need camera calibration.</li>
          <li><strong>Rotation</strong> is in revolutions: 0.5 is half a revolution, 1.5 is a single Axel, 2.5 a double, 3.5 a triple and 4.5 a quad Axel.</li>
          <li><strong>Average speed</strong> is revolutions ÷ flight time; peak is the maximum smoothed rate.</li>
          <li><strong>Skating count</strong> rounds the measured rotation to the nearest 0.5, as skating convention does.</li>
          <li><strong>2D-only models</strong> can undercount full revolutions; the 3D shoulder line is preferred.</li>
        </ul>
        <h3>Limits</h3>
        <ul>
          <li>Coaching-grade, not judging-grade. Expect roughly ±2–3 cm height at 60 fps.</li>
          <li>Single skater, single camera. Edges, toe-pick assist and under-rotation judging are later work.</li>
          <li>If the report warns that the ice line drifted, the clip is not suitable for metrics.</li>
        </ul>
      </div>
    </section>

    <footer class="site-footer">
      <strong>Skate Metrics</strong> · developed by
      <a href="https://www.rayfooproductions.com" target="_blank" rel="noopener">Ray Foo Productions</a>
      · <a href="https://www.instagram.com/edgeswithray/" target="_blank" rel="noopener">@edgeswithray</a>
    </footer>
  </main>
</div>

<script>
const $ = (id) => document.getElementById(id);
const state = { jobId: null, timer: null, camera: null, recorder: null, chunks: [] };
const STEPS = ["Reading video metadata", "Detecting pose in every frame",
               "Detecting jumps and measuring rotation", "Rendering skeleton overlay",
               "Writing report"];

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => (
    { "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c]
  ));
}
function fmt(value, digits = 2) {
  return Number(value || 0).toFixed(digits);
}
function rangeValue(id) {
  const value = $(id).value.trim();
  return value === "" ? null : Number(value);
}
function rangePair(startId, endId) {
  return [rangeValue(startId), rangeValue(endId)];
}
function speedValue(id) {
  const value = $(id).value.trim();
  return value === "" ? 1.0 : Number(value);
}
function switchView(view) {
  document.querySelectorAll("nav button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  ["analyze", "sessions", "guide"].forEach((name) => {
    $(`view-${name}`).hidden = name !== view;
  });
  if (view === "sessions") loadSessions();
}
document.querySelectorAll("nav button").forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view));
});

function setStatus(title, sub = "") {
  $("status").classList.add("visible");
  $("stage-title").textContent = title;
  $("stage-sub").textContent = sub;
  $("spinner").style.display = title === "Done" || title === "Failed" ? "none" : "block";
  const stageIndex = STEPS.indexOf(title);
  const allDone = title === "Done";
  $("steps").innerHTML = STEPS.map((step, index) => {
    const done = allDone || (stageIndex >= 0 && index < stageIndex);
    return `<span class="step ${done ? "done" : ""}">${escapeHtml(step)}</span>`;
  }).join("");
}
function showError(message) {
  $("error").textContent = message || "";
}
function friendlyError(error) {
  const text = String(error);
  if (text.includes("Failed to fetch") || text.includes("NetworkError")) {
    return "Cannot reach the analysis server. Check that `make serve` is still running, then reload this page.";
  }
  return text;
}

async function startJob(request, label) {
  $("result").innerHTML = "";
  $("warnings").innerHTML = "";
  showError("");
  setStatus("Starting", label);
  try {
    const response = await request();
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    watch(data.job_id);
  } catch (error) {
    setStatus("Failed", "");
    showError(friendlyError(error));
  }
}

function watch(jobId) {
  state.jobId = jobId;
  localStorage.setItem("skate.lastJob", jobId);
  if (history.replaceState) {
    history.replaceState(null, "", `?job=${encodeURIComponent(jobId)}`);
  }
  clearInterval(state.timer);
  state.timer = setInterval(tick, 900);
  tick();
}

async function tick() {
  if (!state.jobId) return;
  try {
    const response = await fetch(`/api/jobs/${state.jobId}`);
    if (!response.ok) throw new Error(`status ${response.status}`);
    const job = await response.json();
    renderJob(job);
    if (job.status === "done" || job.status === "error") {
      clearInterval(state.timer);
    }
  } catch (error) {
    clearInterval(state.timer);
    setStatus("Failed", "");
    showError(friendlyError(error));
  }
}

function renderJob(job) {
  if (job.status === "error") {
    setStatus("Failed", job.source || "");
    showError(job.error || "Unknown error");
    return;
  }
  if (job.status !== "done") {
    setStatus(job.stage || "Working", job.source || "");
    return;
  }
  const stabilized = job.stabilization && job.stabilization.applied ? " · stabilized" : "";
  setStatus("Done", `${job.width}×${job.height} · ${fmt(job.fps, 1)} fps · ${job.rotation_source}${stabilized}`);
  showError("");
  const notes = job.warnings || [];
  $("warnings").innerHTML = notes.length
    ? `<details class="warn"><summary>Measurement notes (${notes.length})</summary><ul>${notes.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul></details>`
    : "";
  $("result").innerHTML = resultHtml(job);
  wireVideoTabs();
}

function wireVideoTabs() {
  const video = $("result-video");
  if (!video) return;
  document.querySelectorAll(".video-tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".video-tab").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      video.src = button.dataset.src;
      video.play().catch(() => {});
    });
  });
}

function stat(value, unit, label) {
  return `<div class="stat"><div class="v">${value}<small>${unit}</small></div><span class="k">${label}</span></div>`;
}

function resultHtml(job) {
  const jumps = job.jumps || [];
  const approximate = (job.warnings || []).some((w) => w.includes("camera moved"));
  const cards = jumps.map((jump, index) => {
    const direction = jump.direction === "clockwise" ? "CW" : "CCW";
    const approx = approximate || jump.method === "com" ? " · approximate" : "";
    return `
    <div class="jump-block">
      <div class="jump-head">
        <strong>Jump ${index + 1}</strong>
        <span class="pill">${fmt(jump.rotations, 2)} rev · ${direction}${approx}</span>
      </div>
      <div class="stat-grid">
        ${stat(fmt(jump.height_m), "m", "height")}
        ${stat(fmt(jump.rotations, 2), "rev", "rotation")}
        ${stat(fmt(jump.avg_rotation_rps), "rev/s", "avg speed")}
        ${stat(fmt(jump.peak_rotation_rps), "rev/s", "peak speed")}
        ${stat(fmt(jump.flight_time), "s", "flight time")}
      </div>
    </div>`;
  }).join("");
  const charts = jumps.map((jump, index) => `
    <div class="chart">
      <h4>Jump ${index + 1} · rotation speed (rev/s)</h4>
      ${rotationChart(jump.rotation_profile)}
    </div>`).join("");
  return `
    <div class="result-head">
      <div>
        <span class="pill">${jumps.length} jump${jumps.length === 1 ? "" : "s"}</span>
        ${job.stabilization && job.stabilization.applied ? '<span class="pill">camera stabilized</span>' : ""}
      </div>
      <div class="actions">
        <a href="${job.report_url}" target="_blank" rel="noopener">Full report</a>
        <a href="${job.metrics_url}" target="_blank" rel="noopener">metrics.json</a>
        <a href="${job.csv_url}" target="_blank" rel="noopener">rotation.csv</a>
      </div>
    </div>
    <div class="jump-grid">${cards || "<div class='empty'>No plausible jumps detected.</div>"}</div>
    ${jumps.length ? `
      <div class="video-tabs">
        <button class="video-tab active" data-src="${job.overlay_url}">Overlay</button>
        ${job.skeleton_url ? `<button class="video-tab" data-src="${job.skeleton_url}">Skeleton only</button>` : ""}
      </div>
      <video id="result-video" src="${job.overlay_url}" controls playsinline muted loop></video>
    ` : ""}
    <div class="charts">${charts}</div>
  `;
}

function rotationChart(profile) {
  if (!profile || profile.length < 2) return "<p class='note'>No samples.</p>";
  const width = 760, height = 190, left = 40, right = 12, top = 12, bottom = 24;
  const plotW = width - left - right, plotH = height - top - bottom;
  const times = profile.map((p) => p[0]);
  const rates = profile.map((p) => Math.abs(p[1]));
  const t0 = times[0], t1 = times[times.length - 1];
  const span = Math.max(t1 - t0, 1e-6);
  const peak = Math.max(...rates, 1e-6) * 1.12;
  const X = (t) => left + ((t - t0) / span) * plotW;
  const Y = (r) => top + plotH - (r / peak) * plotH;
  const points = rates.map((r, i) => `${X(times[i]).toFixed(1)},${Y(r).toFixed(1)}`).join(" ");
  const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const x = left + f * plotW;
    const t = t0 + f * span;
    return `<line x1="${x}" y1="${top}" x2="${x}" y2="${top + plotH}" stroke="#262626"/>
            <text x="${x}" y="${height - 7}" fill="#777" font-size="10" text-anchor="middle">${t.toFixed(2)}s</text>`;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}" style="width:100%;height:auto">
    ${grid}
    <line x1="${left}" y1="${top + plotH}" x2="${left + plotW}" y2="${top + plotH}" stroke="#444"/>
    <polyline points="${points}" fill="none" stroke="#f0a020" stroke-width="2.5"
              stroke-linejoin="round" stroke-linecap="round"/>
    <text x="${left + 6}" y="${top + 12}" fill="#ccc" font-size="11">${Math.max(...rates).toFixed(2)} rev/s peak</text>
  </svg>`;
}

/* upload */
$("drop").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", () => {
  const file = $("file").files[0];
  if (!file) return;
  window.pendingFile = null;
  $("drop-title").textContent = file.name;
  $("drop-sub").textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;
});
["dragover", "dragenter"].forEach((name) => {
  $("drop").addEventListener(name, (event) => {
    event.preventDefault();
    $("drop").classList.add("over");
  });
});
["dragleave", "drop"].forEach((name) => {
  $("drop").addEventListener(name, (event) => {
    event.preventDefault();
    $("drop").classList.remove("over");
  });
});
$("drop").addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  try {
    const transfer = new DataTransfer();
    transfer.items.add(file);
    $("file").files = transfer.files;
  } catch (error) {
    window.pendingFile = file;
  }
  $("drop-title").textContent = file.name;
  $("drop-sub").textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;
});
$("clear-file").addEventListener("click", () => {
  $("file").value = "";
  window.pendingFile = null;
  $("drop-title").textContent = "Drop a video here";
  $("drop-sub").textContent = "or click to choose a file";
});
$("analyse").addEventListener("click", () => {
  const file = $("file").files[0] || window.pendingFile;
  if (!file) {
    showError("Choose a video first.");
    return;
  }
  const body = new FormData();
  body.append("file", file);
  body.append("pose", $("pose").value);
  const [start, end] = rangePair("upload-start", "upload-end");
  if (start !== null) body.append("start", start);
  if (end !== null) body.append("end", end);
  body.append("playback_speed", speedValue("upload-speed"));
  startJob(() => fetch("/api/analyze", { method: "POST", body }), file.name);
});
$("demo").addEventListener("click", () => {
  startJob(() => fetch("/api/demo", { method: "POST" }), "Synthetic demo");
});
$("analyse-youtube").addEventListener("click", () => {
  const url = $("youtube-url").value.trim();
  if (!url) {
    showError("Paste a YouTube link first.");
    return;
  }
  const [start, end] = rangePair("youtube-start", "youtube-end");
  const payload = {
    url,
    pose: $("pose").value,
    cookies: $("youtube-cookies").value || null,
    start,
    end,
    playback_speed: speedValue("youtube-speed"),
  };
  startJob(
    () => fetch("/api/youtube", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
    url
  );
});

/* sessions */
async function loadSessions() {
  const container = $("sessions");
  container.innerHTML = "<div class='empty'>Loading…</div>";
  try {
    const jobs = await (await fetch("/api/jobs")).json();
    if (!jobs.length) {
      container.innerHTML = "<div class='empty'>No sessions yet. Analyse a clip to start.</div>";
      return;
    }
    container.innerHTML = jobs.map((job) => `
      <div class="session" data-id="${job.id}">
        <span class="dot ${job.status}"></span>
        <div class="name">
          <strong>${escapeHtml(job.source || "Untitled")}</strong>
          <span>${new Date(job.created_at).toLocaleString()} · ${job.kind} · ${job.status}</span>
        </div>
        <div class="metric">
          <b>${job.jump_count}</b> jump${job.jump_count === 1 ? "" : "s"}<br>
          ${job.max_height_m ? fmt(job.max_height_m) + " m best" : ""}
        </div>
        <button class="trash" title="Delete" data-delete="${job.id}">×</button>
      </div>`).join("");
    container.querySelectorAll(".session").forEach((row) => {
      row.addEventListener("click", (event) => {
        if (event.target.dataset.delete) return;
        switchView("analyze");
        watch(row.dataset.id);
      });
    });
    container.querySelectorAll("[data-delete]").forEach((button) => {
      button.addEventListener("click", async (event) => {
        event.stopPropagation();
        if (!confirm("Delete this session?")) return;
        await fetch(`/api/jobs/${button.dataset.delete}`, { method: "DELETE" });
        loadSessions();
      });
    });
  } catch (error) {
    container.innerHTML = `<div class="empty">Could not load sessions: ${escapeHtml(error)}</div>`;
  }
}
$("refresh-sessions").addEventListener("click", loadSessions);

/* camera recording */
if (!window.isSecureContext) {
  $("start-camera").disabled = true;
  $("camera-note").textContent =
    "Camera capture needs HTTPS (or localhost). Upload still works over plain HTTP.";
}
$("start-camera").addEventListener("click", async () => {
  try {
    state.camera = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: 60, max: 60 },
      },
      audio: false,
    });
    $("preview").srcObject = state.camera;
    await $("preview").play();
    const track = state.camera.getVideoTracks()[0];
    const settings = track.getSettings();
    const fps = settings.frameRate ? Number(settings.frameRate).toFixed(1) : "?";
    $("camera-note").textContent = `${settings.width}×${settings.height} @ ${fps} fps`;
    if (settings.frameRate && settings.frameRate < 50) {
      $("camera-note").textContent += " — browser capped below 60 fps; accuracy will suffer.";
    }
    $("start-camera").disabled = true;
    $("record").disabled = false;
  } catch (error) {
    setStatus("Failed", "");
    showError("Camera error: " + String(error));
  }
});
$("record").addEventListener("click", () => {
  if (!state.camera) return;
  state.chunks = [];
  const candidates = [
    "video/mp4;codecs=h264",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  const mimeType = candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
  state.recorder = new MediaRecorder(
    state.camera,
    mimeType ? { mimeType, videoBitsPerSecond: 12_000_000 } : {}
  );
  state.recorder.ondataavailable = (event) => {
    if (event.data.size) state.chunks.push(event.data);
  };
  state.recorder.onstop = () => {
    const blob = new Blob(state.chunks, { type: state.recorder.mimeType || "video/webm" });
    const extension = blob.type.includes("mp4") ? "mp4" : "webm";
    const file = new File([blob], `rink-jump.${extension}`, { type: blob.type });
    const body = new FormData();
    body.append("file", file);
    body.append("pose", "mediapipe");
    startJob(() => fetch("/api/analyze", { method: "POST", body }), "Recorded jump");
  };
  state.recorder.start();
  $("record").disabled = true;
  $("stop").disabled = false;
  setStatus("Recording", "Perform the jump, then press Stop & analyse.");
});
$("stop").addEventListener("click", () => {
  if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
  $("record").disabled = false;
  $("stop").disabled = true;
});

/* deep links: ?view=sessions, ?job=<id> */
const params = new URLSearchParams(location.search);
const initialView = params.get("view");
if (initialView === "sessions" || initialView === "guide") switchView(initialView);
const queryJob = params.get("job");
const lastJob = queryJob || localStorage.getItem("skate.lastJob");
if (lastJob) watch(lastJob);
</script>
</body>
</html>
"""
