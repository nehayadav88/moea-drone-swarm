"""Shared utilities for MOEA implementations."""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

from core.problem_definition import EvaluationResult


@dataclass
class Individual:
    x: List[float]
    objectives: Tuple[float, ...]
    violation: float


@dataclass
class MOEAResult:
    pareto_solutions: List[List[float]]
    pareto_objectives: List[Tuple[float, ...]]
    history: List[dict]


def dominates(a: Individual, b: Individual) -> bool:
    if a.violation != b.violation:
        return a.violation < b.violation
    better_or_equal = all(x <= y for x, y in zip(a.objectives, b.objectives))
    strictly_better = any(x < y for x, y in zip(a.objectives, b.objectives))
    return better_or_equal and strictly_better


def fast_non_dominated_sort(population: Sequence[Individual]) -> List[List[int]]:
    dominated_by = [0] * len(population)
    dominates_set = [set() for _ in population]
    fronts: List[List[int]] = [[]]
    for p, ind_p in enumerate(population):
        for q, ind_q in enumerate(population):
            if p == q:
                continue
            if dominates(ind_p, ind_q):
                dominates_set[p].add(q)
            elif dominates(ind_q, ind_p):
                dominated_by[p] += 1
        if dominated_by[p] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates_set[p]:
                dominated_by[q] -= 1
                if dominated_by[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def crowding_distance(front: Sequence[int], population: Sequence[Individual]) -> dict:
    if not front:
        return {}
    m = len(population[0].objectives)
    dist = {idx: 0.0 for idx in front}
    for obj in range(m):
        sorted_idx = sorted(front, key=lambda i: population[i].objectives[obj])
        dist[sorted_idx[0]] = float("inf")
        dist[sorted_idx[-1]] = float("inf")
        min_v = population[sorted_idx[0]].objectives[obj]
        max_v = population[sorted_idx[-1]].objectives[obj]
        denom = max(1e-12, max_v - min_v)
        for i in range(1, len(sorted_idx) - 1):
            prev_v = population[sorted_idx[i - 1]].objectives[obj]
            next_v = population[sorted_idx[i + 1]].objectives[obj]
            dist[sorted_idx[i]] += (next_v - prev_v) / denom
    return dist


def sbx_crossover(p1: Sequence[float], p2: Sequence[float], rng: random.Random, eta: float = 20.0) -> Tuple[List[float], List[float]]:
    c1, c2 = list(p1), list(p2)
    for i in range(len(p1)):
        if rng.random() > 0.5:
            continue
        u = rng.random()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return c1, c2


def polynomial_mutation(x: Sequence[float], rng: random.Random, p_mut: float, eta: float = 20.0) -> List[float]:
    out = list(x)
    for i in range(len(out)):
        if rng.random() > p_mut:
            continue
        u = rng.random()
        if u < 0.5:
            delta = (2 * u) ** (1 / (eta + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        out[i] += delta * 0.1
    return out


def evaluate_population(
    population: Sequence[Sequence[float]],
    evaluator: Callable[[Sequence[float]], EvaluationResult],
    parallel: bool,
    workers: int,
) -> List[Individual]:
    if parallel:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            evals = list(ex.map(evaluator, population))
    else:
        evals = [evaluator(x) for x in population]
    return [Individual(list(x), e.objectives, e.constraint_violation) for x, e in zip(population, evals)]


def extract_pareto(population: Sequence[Individual]) -> Tuple[List[List[float]], List[Tuple[float, ...]]]:
    fronts = fast_non_dominated_sort(population)
    if not fronts:
        return [], []
    first = fronts[0]
    return [population[i].x for i in first], [population[i].objectives for i in first]


def weighted_tchebycheff(objectives: Sequence[float], weight: Sequence[float], ideal: Sequence[float]) -> float:
    return max(w * abs(f - z) for f, w, z in zip(objectives, weight, ideal))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
