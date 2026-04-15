"""MOEA/D implementation for swarm path planning."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from algorithms.common import MOEAResult, evaluate_population, extract_pareto, polynomial_mutation, sbx_crossover, weighted_tchebycheff
from core.problem_definition import SwarmPathPlanningProblem
from simulation.path_planner import EncodedPathPlanner


@dataclass
class MOEADConfig:
    population_size: int = 80
    generations: int = 80
    neighborhood_size: int = 12
    mutation_rate: float | None = None
    seed: int = 0
    parallel_evaluation: bool = True
    workers: int = 4


class MOEAD:
    def __init__(self, config: MOEADConfig | None = None) -> None:
        self.config = config or MOEADConfig()

    @staticmethod
    def _weight_vectors(pop_size: int, n_obj: int, rng: random.Random) -> List[List[float]]:
        w = []
        for _ in range(pop_size):
            vals = [rng.random() for _ in range(n_obj)]
            s = sum(vals)
            w.append([v / max(1e-12, s) for v in vals])
        return w

    def run(self, problem: SwarmPathPlanningProblem) -> MOEAResult:
        cfg = self.config
        rng = random.Random(cfg.seed)
        planner = EncodedPathPlanner(problem)
        p_mut = cfg.mutation_rate if cfg.mutation_rate is not None else 1.0 / max(1, problem.n_var)

        pop_x = planner.random_population(cfg.population_size, cfg.seed)
        pop = evaluate_population(pop_x, problem.evaluate, cfg.parallel_evaluation, cfg.workers)

        weights = self._weight_vectors(cfg.population_size, problem.n_obj, rng)
        neighbors = []
        for i in range(cfg.population_size):
            dists = sorted(range(cfg.population_size), key=lambda j: sum((weights[i][k] - weights[j][k]) ** 2 for k in range(problem.n_obj)))
            neighbors.append(dists[: cfg.neighborhood_size])

        ideal = [min(ind.objectives[j] for ind in pop) for j in range(problem.n_obj)]
        history: List[dict] = []

        for gen in range(cfg.generations):
            for i in range(cfg.population_size):
                b = neighbors[i]
                p1, p2 = pop[rng.choice(b)].x, pop[rng.choice(b)].x
                c1, _ = sbx_crossover(p1, p2, rng)
                child_x = planner.repair(polynomial_mutation(c1, rng, p_mut))
                child = evaluate_population([child_x], problem.evaluate, False, 1)[0]

                for j in range(problem.n_obj):
                    ideal[j] = min(ideal[j], child.objectives[j])

                for idx in b:
                    old = weighted_tchebycheff(pop[idx].objectives, weights[idx], ideal)
                    new = weighted_tchebycheff(child.objectives, weights[idx], ideal)
                    if (child.violation < pop[idx].violation) or (child.violation == pop[idx].violation and new <= old):
                        pop[idx] = child

            sols, objs = extract_pareto(pop)
            history.append({"generation": gen, "pareto_size": len(sols), "best_sum": min(sum(o) for o in objs) if objs else float("inf")})

        pareto_sols, pareto_objs = extract_pareto(pop)
        return MOEAResult(pareto_solutions=pareto_sols, pareto_objectives=pareto_objs, history=history)
