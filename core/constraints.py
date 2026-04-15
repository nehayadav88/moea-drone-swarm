"""Constraint handling for swarm path planning MOP."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

Point = Tuple[float, float]


def energy_violation(energy_used: float, battery_capacity: float) -> float:
    return max(0.0, energy_used - battery_capacity)


def path_feasibility_violation(is_collision_free: bool) -> float:
    return 0.0 if is_collision_free else 1.0


def communication_violation(connectivity_ratio: float, min_ratio: float = 1.0) -> float:
    return max(0.0, min_ratio - connectivity_ratio)


def task_completion_violation(completed_tasks: int, required_tasks: int) -> float:
    return max(0, required_tasks - completed_tasks)


def aggregate_constraint_violation(components: Dict[str, float]) -> float:
    return float(sum(max(0.0, value) for value in components.values()))


def collision_risk_penalty(min_pairwise_distance: float, safety_distance: float) -> float:
    if min_pairwise_distance >= safety_distance:
        return 0.0
    gap = max(1e-6, safety_distance - min_pairwise_distance)
    return gap / max(1e-6, safety_distance)


def sample_connectivity_ratio(adjacency_over_time: Sequence[Sequence[Sequence[int]]]) -> float:
    if not adjacency_over_time:
        return 0.0
    connected_steps = 0
    for adjacency in adjacency_over_time:
        n = len(adjacency)
        if n == 0:
            continue
        visited = {0}
        stack = [0]
        while stack:
            node = stack.pop()
            for other in range(n):
                if adjacency[node][other] and other not in visited:
                    visited.add(other)
                    stack.append(other)
        if len(visited) == n:
            connected_steps += 1
    return connected_steps / max(1, len(adjacency_over_time))
