"""Compare a low-frame-rate analysis against a high-frame-rate reference."""

from __future__ import annotations

from dataclasses import dataclass

from .models import JumpMetrics


@dataclass
class JumpComparison:
    reference: JumpMetrics
    measured: JumpMetrics
    height_delta_m: float
    flight_delta_s: float
    rotation_delta: float
    avg_rate_delta_rps: float


def match_and_compare(
    reference: list[JumpMetrics],
    measured: list[JumpMetrics],
    *,
    match_window_s: float = 0.25,
) -> list[JumpComparison]:
    """Pair jumps by takeoff time and report the measurement deltas."""
    pairs: list[JumpComparison] = []
    used: set[int] = set()
    for ref in reference:
        candidates = [
            (i, jump)
            for i, jump in enumerate(measured)
            if i not in used
            and abs(jump.takeoff_time - ref.takeoff_time) <= match_window_s
        ]
        if not candidates:
            continue
        i, jump = min(
            candidates, key=lambda item: abs(item[1].takeoff_time - ref.takeoff_time)
        )
        used.add(i)
        pairs.append(
            JumpComparison(
                reference=ref,
                measured=jump,
                height_delta_m=jump.height_m - ref.height_m,
                flight_delta_s=jump.flight_time - ref.flight_time,
                rotation_delta=jump.rotations - ref.rotations,
                avg_rate_delta_rps=jump.avg_rotation_rps - ref.avg_rotation_rps,
            )
        )
    return pairs


def summarize(pairs: list[JumpComparison]) -> dict:
    if not pairs:
        return {"count": 0}
    return {
        "count": len(pairs),
        "mean_abs_height_delta_m": sum(abs(p.height_delta_m) for p in pairs) / len(pairs),
        "mean_abs_flight_delta_s": sum(abs(p.flight_delta_s) for p in pairs) / len(pairs),
        "mean_abs_rotation_delta": sum(abs(p.rotation_delta) for p in pairs) / len(pairs),
        "mean_abs_avg_rate_delta_rps": sum(
            abs(p.avg_rate_delta_rps) for p in pairs
        )
        / len(pairs),
    }
