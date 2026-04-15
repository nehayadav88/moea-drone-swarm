"""Diversity metrics including spread (Δ) and pure diversity proxy."""

from __future__ import annotations

from typing import Sequence


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def spread_delta(front: Sequence[Sequence[float]]) -> float:
    if len(front) < 3:
        return 0.0
    sorted_front = sorted(front, key=lambda x: x[0])
    d = [_dist(sorted_front[i], sorted_front[i + 1]) for i in range(len(sorted_front) - 1)]
    d_bar = sum(d) / len(d)
    return sum(abs(x - d_bar) for x in d) / max(1e-12, len(d) * d_bar)


def pure_diversity(front: Sequence[Sequence[float]]) -> float:
    if len(front) < 2:
        return 0.0
    mins = []
    for i in range(len(front)):
        nearest = min(_dist(front[i], front[j]) for j in range(len(front)) if i != j)
        mins.append(nearest)
    return sum(mins) / len(mins)
