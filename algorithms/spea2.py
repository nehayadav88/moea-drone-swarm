"""SPEA2 implementation for swarm path planning."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from algorithms.common import (
    MOEAResult,
    dominates,
    evaluate_population,
    extract_pareto,
    polynomial_mutation,
    sbx_crossover,
)
from core.problem_definition import SwarmPathPlanningProblem
from simulation.path_planner import EncodedPathPlanner


@dataclass
class SPEA2Config:
    population_size: int = 80
    archive_size: int = 80
    generations: int = 80
    crossover_rate: float = 0.9
    mutation_rate: float | None = None
    seed: int = 0
    parallel_evaluation: bool = True
    workers: int = 4


class SPEA2:
    def __init__(self, config: SPEA2Config | None = None) -> None:
        self.config = config or SPEA2Config()

    @staticmethod
    def _distance(a: tuple, b: tuple) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def _fitness(self, union):
        n = len(union)
        strength = [0] * n
        raw = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and dominates(union[i], union[j]):
                    strength[i] += 1
        for i in range(n):
            raw[i] = sum(strength[j] for j in range(n) if i != j and dominates(union[j], union[i]))

        density = [0.0] * n
        k = max(1, int(n**0.5))
        for i in range(n):
            d = sorted(self._distance(union[i].objectives, union[j].objectives) for j in range(n) if i != j)
            sigma = d[min(k - 1, len(d) - 1)] if d else 0.0
            density[i] = 1.0 / (sigma + 2.0)
        return [raw[i] + density[i] + union[i].violation * 10.0 for i in range(n)]

    def run(self, problem: SwarmPathPlanningProblem) -> MOEAResult:
        cfg = self.config
        rng = random.Random(cfg.seed)
        planner = EncodedPathPlanner(problem)
        p_mut = cfg.mutation_rate if cfg.mutation_rate is not None else 1.0 / max(1, problem.n_var)

        pop_x = planner.random_population(cfg.population_size, cfg.seed)
        pop = evaluate_population(pop_x, problem.evaluate, cfg.parallel_evaluation, cfg.workers)
        archive = []
        history: List[dict] = []

        for gen in range(cfg.generations):
            union = pop + archive
            fit = self._fitness(union)
            selected = sorted(range(len(union)), key=lambda i: fit[i])
            archive = [union[i] for i in selected[: cfg.archive_size]]
            archive_fit = [fit[i] for i in selected[: cfg.archive_size]]

            def mate_pick():
                i, j = rng.randrange(len(archive)), rng.randrange(len(archive))
                return archive[i] if archive_fit[i] <= archive_fit[j] else archive[j]

            offspring_x = []
            while len(offspring_x) < cfg.population_size:
                p1 = mate_pick().x
                p2 = mate_pick().x
                if rng.random() < cfg.crossover_rate:
                    c1, c2 = sbx_crossover(p1, p2, rng)
                else:
                    c1, c2 = list(p1), list(p2)
                c1 = planner.repair(polynomial_mutation(c1, rng, p_mut))
                c2 = planner.repair(polynomial_mutation(c2, rng, p_mut))
                offspring_x.extend([c1, c2])

            pop = evaluate_population(offspring_x[: cfg.population_size], problem.evaluate, cfg.parallel_evaluation, cfg.workers)
            sols, objs = extract_pareto(archive)
            history.append({"generation": gen, "pareto_size": len(sols), "best_sum": min(sum(o) for o in objs) if objs else float("inf")})

        pareto_sols, pareto_objs = extract_pareto(archive if archive else pop)
        return MOEAResult(pareto_solutions=pareto_sols, pareto_objectives=pareto_objs, history=history)
