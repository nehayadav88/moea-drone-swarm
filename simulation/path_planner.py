"""Path planning helpers for encoded trajectory initialization and repair."""

from __future__ import annotations

import random
from typing import List, Sequence

from core.problem_definition import SwarmPathPlanningProblem


class EncodedPathPlanner:
    """Utility methods to initialize and repair encoded path decision vectors."""

    def __init__(self, problem: SwarmPathPlanningProblem) -> None:
        self.problem = problem
        self.lower, self.upper = self.problem.bounds()

    def random_solution(self, rng: random.Random) -> List[float]:
        return [rng.uniform(lo, hi) for lo, hi in zip(self.lower, self.upper)]

    def repair(self, vector: Sequence[float]) -> List[float]:
        repaired = []
        for x, lo, hi in zip(vector, self.lower, self.upper):
            repaired.append(min(hi, max(lo, x)))
        return repaired

    def random_population(self, size: int, seed: int) -> List[List[float]]:
        rng = random.Random(seed)
        return [self.random_solution(rng) for _ in range(size)]
