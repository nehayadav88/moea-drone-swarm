"""Multi-objective problem definition for heterogeneous drone swarm path planning.

Decision vector (continuous encoding):
- For each drone d and waypoint k: x[d,k], y[d,k] in normalized [0,1]
- Optional task assignment variables (rounded integers) can be appended.

Objectives (minimize):
f1 = sum_d PathLength_d
f2 = sum_d FlightTime_d
f3 = sum_d Energy_d
f4 = CollisionRisk + ConstraintPenalty

Constraints:
g1_d: Energy_d - Battery_d <= 0
g2:   min_connectivity_ratio - observed_connectivity_ratio <= 0
g3:   path collision free (binary relaxed as violation count)
g4:   required_tasks - completed_tasks <= 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from core.constraints import (
    aggregate_constraint_violation,
    collision_risk_penalty,
    communication_violation,
    energy_violation,
    path_feasibility_violation,
    task_completion_violation,
)
from core.energy_model import EnergyModel, path_length
from simulation.drone import Drone
from simulation.environment import Environment2D

Point = Tuple[float, float]


@dataclass
class ProblemConfig:
    waypoints_per_drone: int = 4
    min_connectivity_ratio: float = 0.9
    safety_distance: float = 3.0
    required_tasks: int = 0
    hover_time_per_drone: float = 0.0


@dataclass
class EvaluationResult:
    objectives: Tuple[float, float, float, float]
    constraint_violation: float
    metadata: Dict[str, float]


class SwarmPathPlanningProblem:
    def __init__(
        self,
        drones: Sequence[Drone],
        environment: Environment2D,
        energy_model: EnergyModel,
        config: ProblemConfig | None = None,
    ) -> None:
        self.drones = list(drones)
        self.environment = environment
        self.energy_model = energy_model
        self.config = config or ProblemConfig()
        self.n_var = len(self.drones) * self.config.waypoints_per_drone * 2
        self.n_obj = 4

    def bounds(self) -> Tuple[List[float], List[float]]:
        lower = [0.0] * self.n_var
        upper = [1.0] * self.n_var
        return lower, upper

    def decode(self, decision_vector: Sequence[float]) -> List[List[Point]]:
        paths: List[List[Point]] = []
        idx = 0
        for drone in self.drones:
            path = [drone.start]
            for _ in range(self.config.waypoints_per_drone):
                nx = min(1.0, max(0.0, decision_vector[idx]))
                ny = min(1.0, max(0.0, decision_vector[idx + 1]))
                idx += 2
                x = self.environment.x_min + nx * (self.environment.x_max - self.environment.x_min)
                y = self.environment.y_min + ny * (self.environment.y_max - self.environment.y_min)
                path.append((x, y))
            path.append(drone.goal)
            paths.append(path)
        return paths

    def _min_pairwise_distance(self, sampled_paths: List[List[Point]]) -> float:
        if not sampled_paths or len(sampled_paths[0]) == 0:
            return 0.0
        min_dist = float("inf")
        steps = len(sampled_paths[0])
        for t in range(steps):
            for i in range(len(sampled_paths)):
                for j in range(i + 1, len(sampled_paths)):
                    p = sampled_paths[i][t]
                    q = sampled_paths[j][t]
                    dist = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
                    min_dist = min(min_dist, dist)
        return 0.0 if min_dist == float("inf") else min_dist

    def evaluate(self, decision_vector: Sequence[float]) -> EvaluationResult:
        paths = self.decode(decision_vector)
        total_distance = 0.0
        total_time = 0.0
        total_energy = 0.0
        total_tasks = 0
        violation_components: Dict[str, float] = {}

        sampled_paths = []
        adjacency_over_time = []

        for i, (drone, path) in enumerate(zip(self.drones, paths)):
            dlen = path_length(path)
            total_distance += dlen
            speed = max(1e-6, drone.speed)
            total_time += dlen / speed
            energy = self.energy_model.path_energy(
                path,
                speed=speed,
                drone_mass=drone.mass,
                payload_mass=drone.payload,
                hover_time=self.config.hover_time_per_drone,
            )
            total_energy += energy
            violation_components[f"energy_{i}"] = energy_violation(energy, drone.battery_capacity)

            is_feasible = self.environment.is_path_collision_free(path)
            violation_components[f"feasible_{i}"] = path_feasibility_violation(is_feasible)
            if drone.task_id is not None:
                total_tasks += int(path[-1] == drone.goal)

            sampled = self.environment.sample_path(path, samples_per_segment=6)
            sampled_paths.append(sampled)

        max_steps = max([len(p) for p in sampled_paths], default=0)
        aligned = [p + [p[-1]] * (max_steps - len(p)) for p in sampled_paths if p]

        for t in range(max_steps):
            positions_t = [p[t] for p in aligned]
            adjacency_over_time.append(self.environment.communication_adjacency(positions_t, [d.comm_range for d in self.drones]))

        connectivity_ratio = self.environment.connectivity_ratio_from_adjacency(adjacency_over_time)
        violation_components["comm"] = communication_violation(connectivity_ratio, self.config.min_connectivity_ratio)

        if self.config.required_tasks > 0:
            violation_components["tasks"] = task_completion_violation(total_tasks, self.config.required_tasks)

        min_dist = self._min_pairwise_distance(aligned)
        risk_penalty = collision_risk_penalty(min_dist, self.config.safety_distance)

        total_violation = aggregate_constraint_violation(violation_components)
        safety_objective = risk_penalty + total_violation

        return EvaluationResult(
            objectives=(total_distance, total_time, total_energy, safety_objective),
            constraint_violation=total_violation,
            metadata={
                "connectivity_ratio": connectivity_ratio,
                "min_pairwise_distance": min_dist,
                "risk_penalty": risk_penalty,
                "tasks_completed": float(total_tasks),
            },
        )
