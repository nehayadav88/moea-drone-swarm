"""Energy model for heterogeneous drone swarm missions."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos
from typing import Iterable, Sequence, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class EnergyModelConfig:
    """Configurable coefficients for comprehensive energy modeling."""

    base_power: float = 45.0
    distance_coeff: float = 1.0
    speed_coeff: float = 0.12
    mass_coeff: float = 0.08
    payload_coeff: float = 0.09
    hover_power: float = 28.0
    turn_coeff: float = 0.15


class EnergyModel:
    """Computes mission energy using distance, speed, mass, hovering, and turning costs."""

    def __init__(self, config: EnergyModelConfig | None = None) -> None:
        self.config = config or EnergyModelConfig()

    @staticmethod
    def _segment_length(a: Point, b: Point) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def _turn_angle(p0: Point, p1: Point, p2: Point) -> float:
        v1 = (p0[0] - p1[0], p0[1] - p1[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        n1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        n2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        dot = min(1.0, max(-1.0, dot))
        return acos(dot)

    def segment_energy(self, distance: float, speed: float, total_mass: float) -> float:
        cfg = self.config
        speed_term = 1.0 + cfg.speed_coeff * max(0.0, speed)
        mass_term = 1.0 + cfg.mass_coeff * max(0.0, total_mass)
        propulsion = cfg.base_power * cfg.distance_coeff * distance * speed_term * mass_term
        return propulsion

    def hover_energy(self, hover_time: float, total_mass: float) -> float:
        cfg = self.config
        return cfg.hover_power * hover_time * (1.0 + cfg.mass_coeff * max(0.0, total_mass))

    def turning_energy(self, path: Sequence[Point], speed: float, total_mass: float) -> float:
        if len(path) < 3:
            return 0.0
        turn_sum = 0.0
        for i in range(1, len(path) - 1):
            turn_sum += self._turn_angle(path[i - 1], path[i], path[i + 1])
        return self.config.turn_coeff * turn_sum * (1.0 + speed * 0.05) * (1.0 + total_mass * 0.03)

    def path_energy(
        self,
        path: Sequence[Point],
        speed: float,
        drone_mass: float,
        payload_mass: float,
        hover_time: float = 0.0,
    ) -> float:
        if len(path) < 2:
            return 0.0
        total_mass = drone_mass + self.config.payload_coeff * payload_mass
        distance = sum(self._segment_length(path[i], path[i + 1]) for i in range(len(path) - 1))
        flight = self.segment_energy(distance, speed, total_mass)
        hover = self.hover_energy(hover_time, total_mass)
        turns = self.turning_energy(path, speed, total_mass)
        return flight + hover + turns


def path_length(path: Iterable[Point]) -> float:
    pts = list(path)
    if len(pts) < 2:
        return 0.0
    return sum(((pts[i][0] - pts[i + 1][0]) ** 2 + (pts[i][1] - pts[i + 1][1]) ** 2) ** 0.5 for i in range(len(pts) - 1))
