"""Drone model for heterogeneous swarm simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

Point = Tuple[float, float]


@dataclass
class Drone:
    drone_id: int
    start: Point
    goal: Point
    battery_capacity: float
    speed: float
    sensing_range: float
    payload: float
    mass: float = 2.0
    comm_range: float = 20.0
    task_id: Optional[int] = None
