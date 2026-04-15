"""Inverted Generational Distance (IGD)."""

from __future__ import annotations

from typing import Sequence


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def igd(approx_front: Sequence[Sequence[float]], reference_front: Sequence[Sequence[float]]) -> float:
    if not approx_front or not reference_front:
        return float("inf")
    dists = []
    for r in reference_front:
        dists.append(min(_dist(r, a) for a in approx_front))
    return sum(dists) / len(dists)
