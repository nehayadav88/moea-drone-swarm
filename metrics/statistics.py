"""Statistical comparison helpers for algorithm benchmarking."""

from __future__ import annotations

import math
from typing import Dict, Sequence


def wilcoxon_rank_sum(sample_a: Sequence[float], sample_b: Sequence[float]) -> Dict[str, float]:
    """Two-sided Mann-Whitney-Wilcoxon rank-sum approximation using normal z."""
    if not sample_a or not sample_b:
        return {"u": float("nan"), "z": float("nan"), "p_value": float("nan")}

    combined = [(v, 0) for v in sample_a] + [(v, 1) for v in sample_b]
    combined.sort(key=lambda x: x[0])

    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    r1 = sum(ranks[i] for i, (_, g) in enumerate(combined) if g == 0)
    n1, n2 = len(sample_a), len(sample_b)
    u1 = r1 - n1 * (n1 + 1) / 2.0

    mean_u = n1 * n2 / 2.0
    sigma_u = math.sqrt(max(1e-12, n1 * n2 * (n1 + n2 + 1) / 12.0))
    z = (u1 - mean_u) / sigma_u

    cdf = 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
    p_value = 2 * (1 - cdf)
    return {"u": u1, "z": z, "p_value": p_value}


def improvement_rate(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - candidate) / abs(baseline) * 100.0


def convergence_trace(history: Sequence[dict], key: str = "best_sum") -> Sequence[float]:
    return [entry[key] for entry in history if key in entry]
