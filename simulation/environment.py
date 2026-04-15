"""2-D simulation environment with static and dynamic circular obstacles."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class Environment:
    """A rectangular 2-D environment containing circular obstacles.

    Static obstacles are fixed; dynamic obstacles move at constant velocity
    and bounce off the environment walls.

    Attributes:
        width: Environment width in metres.
        height: Environment height in metres.
        obstacles: List of (cx, cy, radius) static obstacles.
        dynamic_obstacles: List of (cx, cy, radius, vx, vy) moving obstacles.
    """

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        obstacles: List[Tuple[float, float, float]] | None = None,
        dynamic_obstacles: List[Tuple[float, float, float, float, float]] | None = None,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.obstacles: List[Tuple[float, float, float]] = list(obstacles) if obstacles else []
        self.dynamic_obstacles: List[Tuple[float, float, float, float, float]] = (
            list(dynamic_obstacles) if dynamic_obstacles else []
        )

    # ------------------------------------------------------------------
    # Random obstacle generation
    # ------------------------------------------------------------------

    def generate_random_obstacles(
        self,
        n_static: int = 15,
        n_dynamic: int = 5,
        min_radius: float = 1.0,
        max_radius: float = 5.0,
        seed: int | None = None,
    ) -> None:
        """Populate the environment with randomly placed obstacles.

        Obstacles are guaranteed not to overlap (with a margin equal to 1.0 m)
        and are kept away from the environment border by at least *max_radius*.

        Args:
            n_static: Number of static obstacles to place.
            n_dynamic: Number of dynamic obstacles to place.
            min_radius: Minimum obstacle radius.
            max_radius: Maximum obstacle radius.
            seed: Optional RNG seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        margin = max_radius  # keep obstacles away from walls
        overlap_margin = 1.0  # extra spacing between obstacles

        placed: list[tuple[float, float, float]] = []

        def _try_place(rng: np.random.Generator, radius: float) -> tuple[float, float] | None:
            for _ in range(500):
                cx = rng.uniform(margin, self.width - margin)
                cy = rng.uniform(margin, self.height - margin)
                if all(
                    np.hypot(cx - px, cy - py) > radius + pr + overlap_margin
                    for px, py, pr in placed
                ):
                    return cx, cy
            return None

        self.obstacles.clear()
        self.dynamic_obstacles.clear()

        for _ in range(n_static):
            r = rng.uniform(min_radius, max_radius)
            pos = _try_place(rng, r)
            if pos is None:
                continue
            cx, cy = pos
            placed.append((cx, cy, r))
            self.obstacles.append((cx, cy, r))

        for _ in range(n_dynamic):
            r = rng.uniform(min_radius, max_radius)
            pos = _try_place(rng, r)
            if pos is None:
                continue
            cx, cy = pos
            vx = rng.uniform(-2.0, 2.0)
            vy = rng.uniform(-2.0, 2.0)
            placed.append((cx, cy, r))
            self.dynamic_obstacles.append((cx, cy, r, vx, vy))

    # ------------------------------------------------------------------
    # Dynamic obstacle update
    # ------------------------------------------------------------------

    def update_dynamic_obstacles(self, dt: float = 1.0) -> None:
        """Advance dynamic obstacles by *dt* seconds, bouncing off walls.

        Each obstacle's centre is moved by (vx*dt, vy*dt).  If the obstacle
        would exceed the environment bounds (accounting for its radius), the
        velocity component is reversed and the position is clamped.

        Args:
            dt: Time-step in seconds.
        """
        updated: list[tuple[float, float, float, float, float]] = []
        for cx, cy, r, vx, vy in self.dynamic_obstacles:
            new_cx = cx + vx * dt
            new_cy = cy + vy * dt

            if new_cx - r < 0:
                new_cx = r
                vx = abs(vx)
            elif new_cx + r > self.width:
                new_cx = self.width - r
                vx = -abs(vx)

            if new_cy - r < 0:
                new_cy = r
                vy = abs(vy)
            elif new_cy + r > self.height:
                new_cy = self.height - r
                vy = -abs(vy)

            updated.append((new_cx, new_cy, r, vx, vy))
        self.dynamic_obstacles = updated

    # ------------------------------------------------------------------
    # Collision queries
    # ------------------------------------------------------------------

    def is_point_in_obstacle(self, x: float, y: float) -> bool:
        """Return True if the point (x, y) lies inside any obstacle."""
        for cx, cy, r in self.get_obstacle_positions():
            if np.hypot(x - cx, y - cy) <= r:
                return True
        return False

    def is_path_clear(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check whether the line segment p1→p2 is free of obstacle collisions.

        Uses analytic line–circle intersection: for each circle centred at
        *c* with radius *r*, the closest point on the segment to *c* is
        computed and its distance compared with *r*.

        Args:
            p1: Start point, shape (2,).
            p2: End point, shape (2,).

        Returns:
            True if the segment does **not** intersect any obstacle.
        """
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        d = p2 - p1
        seg_len_sq = float(np.dot(d, d))

        for cx, cy, r in self.get_obstacle_positions():
            c = np.array([cx, cy], dtype=np.float64)
            if seg_len_sq == 0.0:
                # Degenerate segment (single point)
                if np.hypot(p1[0] - cx, p1[1] - cy) <= r:
                    return False
                continue
            # Parameter t of nearest point on the infinite line
            t = float(np.dot(c - p1, d)) / seg_len_sq
            t = max(0.0, min(1.0, t))
            nearest = p1 + t * d
            if np.hypot(nearest[0] - cx, nearest[1] - cy) <= r:
                return False
        return True

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def get_obstacle_positions(self) -> List[Tuple[float, float, float]]:
        """Return (cx, cy, radius) for every obstacle (static + dynamic).

        Dynamic obstacles are returned at their current position.
        """
        positions = list(self.obstacles)
        for cx, cy, r, _vx, _vy in self.dynamic_obstacles:
            positions.append((cx, cy, r))
        return positions

    def point_to_nearest_obstacle_distance(self, x: float, y: float) -> float:
        """Return the minimum distance from (x, y) to the nearest obstacle surface.

        The distance is measured from the point to the circle boundary, i.e.
        ``dist_to_centre - radius``.  A negative value means the point is
        inside an obstacle.  Returns ``np.inf`` when there are no obstacles.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Signed distance to nearest obstacle surface.
        """
        obstacles = self.get_obstacle_positions()
        if not obstacles:
            return np.inf
        min_dist = np.inf
        for cx, cy, r in obstacles:
            dist = np.hypot(x - cx, y - cy) - r
            if dist < min_dist:
                min_dist = dist
        return float(min_dist)

    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return the environment bounds as ``((0, width), (0, height))``."""
        return ((0.0, self.width), (0.0, self.height))
