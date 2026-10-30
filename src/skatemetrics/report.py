"""Standalone HTML report, no external JS or CSS dependencies."""

from __future__ import annotations

import html
from pathlib import Path

from .models import JumpMetrics


def _rotation_svg(metrics: JumpMetrics, index: int) -> str:
    profile = metrics.rotation_profile
    if len(profile) < 2:
        return "<p class='muted'>No rotation samples.</p>"

    width, height = 760, 220
    left, right, top, bottom = 52, 18, 18, 34
    plot_w = width - left - right
    plot_h = height - top - bottom

    times = [t for t, _ in profile]
    rates = [abs(r) for _, r in profile]
    t0, t1 = times[0], times[-1]
    span = max(t1 - t0, 1e-6)
    peak = max(max(rates), 1e-6) * 1.12

    def x(t: float) -> float:
        return left + (t - t0) / span * plot_w

    def y(r: float) -> float:
        return top + plot_h - (r / peak) * plot_h

    points = " ".join(f"{x(t):.1f},{y(r):.1f}" for t, r in zip(times, rates))
    zero = top + plot_h

    grid = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = left + frac * plot_w
        label = t0 + frac * span
        grid.append(
            f"<line x1='{gx:.1f}' y1='{top}' x2='{gx:.1f}' y2='{top + plot_h}' "
            f"stroke='#2a2a2a'/>"
            f"<text x='{gx:.1f}' y='{height - 10}' fill='#888' font-size='11' "
            f"text-anchor='middle'>{label:.2f}s</text>"
        )

    peak_label = f"{max(abs(r) for r in rates):.2f} rev/s peak"
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Rotation rate for jump {index}">
  <rect width="{width}" height="{height}" fill="#161616" rx="10"/>
  {''.join(grid)}
  <line x1="{left}" y1="{zero:.1f}" x2="{left + plot_w}" y2="{zero:.1f}" stroke="#555"/>
  <polyline points="{points}" fill="none" stroke="#f0a020" stroke-width="2.5"
            stroke-linejoin="round" stroke-linecap="round"/>
  <text x="{left + 8}" y="{top + 14}" fill="#ccc" font-size="12">{peak_label}</text>
</svg>
"""


def build_report(
    output_dir: str | Path,
    jumps: list[JumpMetrics],
    *,
    source_name: str,
    video_name: str = "overlay.mp4",
    skeleton_name: str | None = None,
    fps: float = 60.0,
    warnings: list[str] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    for i, jump in enumerate(jumps, start=1):
        direction = "CCW" if jump.direction > 0 else "CW"
        cards.append(
            f"""
<section class="jump">
  <div class="jump-head">
    <h2>Jump {i}</h2>
    <span class="tag">{direction}</span>
    <span class="tag">{jump.rotations:.2f} rev</span>
    <span class="tag">{jump.half_turns:.1f} rev skating count</span>
  </div>
  <div class="stats">
    <div><span class="value">{jump.height_m:.2f}<small>m</small></span><span class="label">height</span></div>
    <div><span class="value">{jump.avg_rotation_rps:.2f}<small>rev/s</small></span><span class="label">avg rotation</span></div>
    <div><span class="value">{jump.peak_rotation_rps:.2f}<small>rev/s</small></span><span class="label">peak rotation</span></div>
    <div><span class="value">{jump.flight_time:.2f}<small>s</small></span><span class="label">flight time</span></div>
  </div>
  {_rotation_svg(jump, i)}
</section>
"""
        )

    if not cards:
        cards.append(
            "<section class='jump'><h2>No jumps detected</h2>"
            "<p class='muted'>Check that the full body and the ice are in frame, "
            "and that the camera does not pan during the jump.</p></section>"
        )

    source = html.escape(source_name)
    video = html.escape(video_name)
    skeleton_html = ""
    if skeleton_name:
        skeleton_html = (
            "<h2 class='subhead'>Skeleton view</h2>"
            f"<video src='{html.escape(skeleton_name)}' controls playsinline muted loop></video>"
        )
    warning_html = ""
    if warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
        warning_html = f"<div class='warning'><strong>Measurement notes</strong><ul>{items}</ul></div>"
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="author" content="Ray Foo Productions">
<title>Skate Metrics — {source}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #0e0e0e; color: #eee;
         font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  main {{ max-width: 960px; margin: 0 auto; padding: 32px 20px 64px; }}
  header {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
            border-bottom: 1px solid #2a2a2a; padding-bottom: 16px; margin-bottom: 24px; }}
  h1 {{ font-size: 22px; margin: 0; letter-spacing: .02em; }}
  h2 {{ font-size: 18px; margin: 0; }}
  .subhead {{ margin: 22px 0 10px; font-size: 15px; color: #bbb; text-transform: uppercase;
             letter-spacing: .08em; }}
  .muted {{ color: #888; }}
  .meta {{ color: #888; font-size: 13px; }}
  video {{ width: 100%; border-radius: 12px; background: #000; display: block; }}
  .jump {{ margin-top: 28px; border-top: 1px solid #242424; padding-top: 20px; }}
  .jump-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }}
  .tag {{ font-size: 12px; color: #f0a020; border: 1px solid #5a4218;
          border-radius: 999px; padding: 2px 9px; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
            margin-bottom: 16px; }}
  .stats div {{ background: #181818; border: 1px solid #262626; border-radius: 10px;
                padding: 12px 14px; display: flex; flex-direction: column; }}
  .value {{ font-size: 24px; font-variant-numeric: tabular-nums; }}
  .value small {{ font-size: 12px; color: #999; margin-left: 4px; }}
  .label {{ font-size: 11px; color: #888; text-transform: uppercase;
            letter-spacing: .08em; margin-top: 4px; }}
  .warning {{ border: 1px solid #6b4a16; background: #221a0d; color: #f0c674;
              border-radius: 10px; padding: 12px 16px; margin-bottom: 18px; }}
  .warning ul {{ margin: 6px 0 0; padding-left: 18px; }}
  footer {{ margin-top: 32px; color: #777; font-size: 12px; }}
  .credit {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid #242424; }}
  .credit a {{ color: #999; text-decoration: none; }}
  .credit a:hover {{ color: #f0a020; }}
  @media (max-width: 620px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>
<main>
  <header>
    <h1>Skate Metrics</h1>
    <span class="meta">{source} · {fps:.0f} fps · {len(jumps)} jump(s) detected</span>
  </header>
  {warning_html}
  <video src="{video}" controls playsinline muted loop></video>
  {skeleton_html}
  {''.join(cards)}
  <footer>
    Height is derived from flight time (h = g·t²/8). Rotation is measured from
    the shoulder line and shown in revolutions: 0.5 is half a revolution,
    1.5 is a single Axel, 3.5 a triple Axel and 4.5 a quad Axel. Coaching-grade
    estimates from a single 60 fps camera, not judging-grade measurements.
    <div class="credit">
      Developed by
      <a href="https://www.rayfooproductions.com" target="_blank" rel="noopener">Ray Foo Productions</a>
      · <a href="https://www.instagram.com/edgeswithray/" target="_blank" rel="noopener">@edgeswithray</a>
    </div>
  </footer>
</main>
</body>
</html>
"""
    path = output_dir / "report.html"
    path.write_text(page)
    return path
