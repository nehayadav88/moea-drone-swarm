"""Hypervolume metric utilities for minimization problems."""

from __future__ import annotations

import random
from typing import Sequence, Tuple


def hypervolume(front: Sequence[Sequence[float]], reference_point: Sequence[float], samples: int = 20000, seed: int = 0) -> float:
    """Monte-Carlo hypervolume estimate for arbitrary objective dimension."""
    if not front:
        return 0.0
    dim = len(reference_point)
    mins = [min(p[i] for p in front) for i in range(dim)]
    maxs = [reference_point[i] for i in range(dim)]
    if any(maxs[i] <= mins[i] for i in range(dim)):
        return 0.0

    rng = random.Random(seed)
    dominated = 0
    for _ in range(samples):
        point = [rng.uniform(mins[i], maxs[i]) for i in range(dim)]
        if any(all(p[k] <= point[k] for k in range(dim)) for p in front):
            dominated += 1

    box_vol = 1.0
    for i in range(dim):
        box_vol *= maxs[i] - mins[i]
    return box_vol * dominated / samples
