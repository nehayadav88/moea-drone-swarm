"""2D environment with static/dynamic obstacles and communication modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point = Tuple[float, float]


@dataclass
class CircleObstacle:
    center: Point
    radius: float
    velocity: Point = (0.0, 0.0)

    def position_at(self, t: float) -> Point:
        return (self.center[0] + self.velocity[0] * t, self.center[1] + self.velocity[1] * t)


@dataclass
class RectObstacle:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Environment2D:
    def __init__(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        static_circles: Sequence[CircleObstacle] | None = None,
        dynamic_circles: Sequence[CircleObstacle] | None = None,
        static_rects: Sequence[RectObstacle] | None = None,
    ) -> None:
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.static_circles = list(static_circles or [])
        self.dynamic_circles = list(dynamic_circles or [])
        self.static_rects = list(static_rects or [])

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def in_bounds(self, p: Point) -> bool:
        return self.x_min <= p[0] <= self.x_max and self.y_min <= p[1] <= self.y_max

    def point_collides(self, p: Point, t: float = 0.0) -> bool:
        if not self.in_bounds(p):
            return True
        for obs in self.static_circles:
            if self._distance(p, obs.center) <= obs.radius:
                return True
        for obs in self.dynamic_circles:
            if self._distance(p, obs.position_at(t)) <= obs.radius:
                return True
        for r in self.static_rects:
            if r.x_min <= p[0] <= r.x_max and r.y_min <= p[1] <= r.y_max:
                return True
        return False

    def segment_collides(self, a: Point, b: Point, t0: float = 0.0, samples: int = 20) -> bool:
        for i in range(samples + 1):
            alpha = i / samples
            p = (a[0] + alpha * (b[0] - a[0]), a[1] + alpha * (b[1] - a[1]))
            if self.point_collides(p, t=t0 + alpha):
                return True
        return False

    def is_path_collision_free(self, path: Sequence[Point]) -> bool:
        if len(path) < 2:
            return False
        for i in range(len(path) - 1):
            if self.segment_collides(path[i], path[i + 1], t0=float(i)):
                return False
        return True

    def sample_path(self, path: Sequence[Point], samples_per_segment: int = 5) -> List[Point]:
        sampled: List[Point] = []
        if len(path) < 2:
            return sampled
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            for j in range(samples_per_segment):
                alpha = j / samples_per_segment
                sampled.append((a[0] + alpha * (b[0] - a[0]), a[1] + alpha * (b[1] - a[1])))
        sampled.append(path[-1])
        return sampled

    def communication_adjacency(self, positions: Sequence[Point], comm_ranges: Sequence[float]) -> List[List[int]]:
        n = len(positions)
        adjacency = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            adjacency[i][i] = 1
            for j in range(i + 1, n):
                threshold = min(comm_ranges[i], comm_ranges[j])
                connected = int(self._distance(positions[i], positions[j]) <= threshold)
                adjacency[i][j] = connected
                adjacency[j][i] = connected
        return adjacency

    @staticmethod
    def is_graph_connected(adjacency: Sequence[Sequence[int]]) -> bool:
        n = len(adjacency)
        if n == 0:
            return True
        visited = {0}
        stack = [0]
        while stack:
            node = stack.pop()
            for nxt in range(n):
                if adjacency[node][nxt] and nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        return len(visited) == n

    def connectivity_ratio_from_adjacency(self, adjacency_over_time: Sequence[Sequence[Sequence[int]]]) -> float:
        if not adjacency_over_time:
            return 0.0
        connected = sum(1 for a in adjacency_over_time if self.is_graph_connected(a))
        return connected / len(adjacency_over_time)
