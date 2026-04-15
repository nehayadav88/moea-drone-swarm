"""NSGA-II implementation for swarm path planning."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from algorithms.common import (
    MOEAResult,
    crowding_distance,
    evaluate_population,
    extract_pareto,
    fast_non_dominated_sort,
    polynomial_mutation,
    sbx_crossover,
)
from core.problem_definition import SwarmPathPlanningProblem
from simulation.path_planner import EncodedPathPlanner


@dataclass
class NSGA2Config:
    population_size: int = 80
    generations: int = 80
    crossover_rate: float = 0.9
    mutation_rate: float | None = None
    seed: int = 0
    parallel_evaluation: bool = True
    workers: int = 4


class NSGA2:
    def __init__(self, config: NSGA2Config | None = None) -> None:
        self.config = config or NSGA2Config()

    def run(self, problem: SwarmPathPlanningProblem) -> MOEAResult:
        cfg = self.config
        rng = random.Random(cfg.seed)
        planner = EncodedPathPlanner(problem)
        p_mut = cfg.mutation_rate if cfg.mutation_rate is not None else 1.0 / max(1, problem.n_var)

        pop_x = planner.random_population(cfg.population_size, cfg.seed)
        pop = evaluate_population(pop_x, problem.evaluate, cfg.parallel_evaluation, cfg.workers)
        history: List[dict] = []

        for gen in range(cfg.generations):
            fronts = fast_non_dominated_sort(pop)
            crowd = {}
            for front in fronts:
                crowd.update(crowding_distance(front, pop))

            def tournament() -> List[float]:
                i, j = rng.randrange(len(pop)), rng.randrange(len(pop))
                if pop[i].violation < pop[j].violation:
                    return pop[i].x
                if pop[j].violation < pop[i].violation:
                    return pop[j].x
                rank_i = next((r for r, f in enumerate(fronts) if i in f), len(fronts))
                rank_j = next((r for r, f in enumerate(fronts) if j in f), len(fronts))
                if rank_i < rank_j:
                    return pop[i].x
                if rank_j < rank_i:
                    return pop[j].x
                return pop[i].x if crowd.get(i, 0.0) >= crowd.get(j, 0.0) else pop[j].x

            offspring_x: List[List[float]] = []
            while len(offspring_x) < cfg.population_size:
                p1, p2 = tournament(), tournament()
                if rng.random() < cfg.crossover_rate:
                    c1, c2 = sbx_crossover(p1, p2, rng)
                else:
                    c1, c2 = list(p1), list(p2)
                c1 = planner.repair(polynomial_mutation(c1, rng, p_mut))
                c2 = planner.repair(polynomial_mutation(c2, rng, p_mut))
                offspring_x.extend([c1, c2])
            offspring = evaluate_population(offspring_x[: cfg.population_size], problem.evaluate, cfg.parallel_evaluation, cfg.workers)

            merged = pop + offspring
            merged_fronts = fast_non_dominated_sort(merged)
            next_pop = []
            for front in merged_fronts:
                if len(next_pop) + len(front) <= cfg.population_size:
                    next_pop.extend(merged[i] for i in front)
                else:
                    d = crowding_distance(front, merged)
                    front_sorted = sorted(front, key=lambda i: d[i], reverse=True)
                    needed = cfg.population_size - len(next_pop)
                    next_pop.extend(merged[i] for i in front_sorted[:needed])
                    break
            pop = next_pop

            sols, objs = extract_pareto(pop)
            history.append({"generation": gen, "pareto_size": len(sols), "best_sum": min(sum(o) for o in objs) if objs else float("inf")})

        pareto_sols, pareto_objs = extract_pareto(pop)
        return MOEAResult(pareto_solutions=pareto_sols, pareto_objectives=pareto_objs, history=history)
