"""Download a public YouTube clip for analysis.

YouTube downloads are best-effort: many videos work without credentials, but
some require sign-in or trigger bot checks. The web app and CLI therefore expose
an optional "cookies from browser" switch instead of failing silently.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
}


class YouTubeError(RuntimeError):
    """Friendly download error for the UI and CLI."""


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise ValueError("empty URL")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def is_youtube_url(url: str) -> bool:
    try:
        host = (urlparse(normalize_url(url)).hostname or "").lower()
    except ValueError:
        return False
    return host in YOUTUBE_HOSTS or host.endswith(".youtube.com")


def _format_selector() -> str:
    # Prefer 60 fps and <=1080p; fall back progressively. mp4+m4a avoids a
    # remux on most videos, but the final fallback still works with webm.
    return (
        "bv*[ext=mp4][height<=1080][fps>=50]+ba[ext=m4a]/"
        "b[ext=mp4][height<=1080][fps>=50]/"
        "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/"
        "b[ext=mp4][height<=1080]/"
        "bv*[height<=1080]+ba/"
        "b[height<=1080]/b"
    )


def download_youtube(
    url: str,
    output_dir: str | Path,
    *,
    cookies: str | None = None,
    cookiefile: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict]:
    """Download a video and return ``(path, info)``.

    ``cookies`` is a browser name (safari, chrome, firefox, edge, brave) and is
    used for videos that require a signed-in session.
    """
    if not is_youtube_url(url):
        raise ValueError("not a YouTube URL")
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError as exc:  # pragma: no cover - dependency is in the web extra
        raise YouTubeError(
            "yt-dlp is not installed; run `uv sync --extra web`"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def say(stage: str) -> None:
        if progress is not None:
            progress(stage)

    def hook(data: dict) -> None:
        if data.get("status") == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            done = data.get("downloaded_bytes") or 0
            if total:
                say(f"Downloading from YouTube ({100 * done / total:.0f}%)")
            else:
                say("Downloading from YouTube")
        elif data.get("status") == "finished":
            say("Download finished; merging video")

    options: dict = {
        "format": _format_selector(),
        "outtmpl": str(output_dir / "input.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "socket_timeout": 30,
        "overwrites": True,
        "progress_hooks": [hook],
    }
    if cookies:
        options["cookiesfrombrowser"] = (cookies, None, None, None)
    elif cookiefile:
        options["cookiefile"] = str(cookiefile)

    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as exc:
        message = str(exc)
        if "sign in" in message.lower() or "bot" in message.lower():
            message = (
                "YouTube blocked this download. Try again with 'Use browser "
                "cookies' enabled, or export a cookies.txt file and set "
                "SKATE_YTDLP_COOKIES."
            )
        raise YouTubeError(message) from exc

    candidates = sorted(output_dir.glob("input.*"))
    if not candidates:
        raise YouTubeError("YouTube download finished but produced no video file")
    # Prefer mp4 if both a merged and an intermediate file are present.
    video = next((p for p in candidates if p.suffix.lower() == ".mp4"), candidates[0])
    return video, info if isinstance(info, dict) else {}
