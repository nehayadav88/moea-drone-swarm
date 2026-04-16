"""3-D simulation environment with static and dynamic spherical obstacles."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class Environment3D:
    """A rectangular 3-D environment containing spherical obstacles.

    Static obstacles are fixed; dynamic obstacles move at constant velocity
    and bounce off the environment walls.

    Attributes:
        width: Environment extent along the X axis in metres.
        height: Environment extent along the Y axis in metres.
        depth: Environment extent along the Z axis in metres.
        obstacles: List of ``(cx, cy, cz, radius)`` static obstacles.
        dynamic_obstacles: List of ``(cx, cy, cz, radius, vx, vy, vz)``
            moving obstacles.
    """

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        depth: float = 50.0,
        obstacles: List[Tuple[float, float, float, float]] | None = None,
        dynamic_obstacles: (
            List[Tuple[float, float, float, float, float, float, float]] | None
        ) = None,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.depth = float(depth)
        self.obstacles: List[Tuple[float, float, float, float]] = (
            list(obstacles) if obstacles else []
        )
        self.dynamic_obstacles: List[
            Tuple[float, float, float, float, float, float, float]
        ] = list(dynamic_obstacles) if dynamic_obstacles else []

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
        """Populate the environment with randomly placed spherical obstacles.

        Obstacles are guaranteed not to overlap (with a margin of 1.0 m) and
        are kept away from the environment border by at least *max_radius*.

        Args:
            n_static: Number of static obstacles to place.
            n_dynamic: Number of dynamic obstacles to place.
            min_radius: Minimum obstacle radius.
            max_radius: Maximum obstacle radius.
            seed: Optional RNG seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        margin = max_radius
        overlap_margin = 1.0

        placed: list[tuple[float, float, float, float]] = []

        def _try_place(
            rng: np.random.Generator, radius: float
        ) -> tuple[float, float, float] | None:
            for _ in range(500):
                cx = rng.uniform(margin, self.width - margin)
                cy = rng.uniform(margin, self.height - margin)
                cz = rng.uniform(margin, self.depth - margin)
                if all(
                    np.sqrt(
                        (cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2
                    )
                    > radius + pr + overlap_margin
                    for px, py, pz, pr in placed
                ):
                    return cx, cy, cz
            return None

        self.obstacles.clear()
        self.dynamic_obstacles.clear()

        for _ in range(n_static):
            r = rng.uniform(min_radius, max_radius)
            pos = _try_place(rng, r)
            if pos is None:
                continue
            cx, cy, cz = pos
            placed.append((cx, cy, cz, r))
            self.obstacles.append((cx, cy, cz, r))

        for _ in range(n_dynamic):
            r = rng.uniform(min_radius, max_radius)
            pos = _try_place(rng, r)
            if pos is None:
                continue
            cx, cy, cz = pos
            vx = rng.uniform(-2.0, 2.0)
            vy = rng.uniform(-2.0, 2.0)
            vz = rng.uniform(-2.0, 2.0)
            placed.append((cx, cy, cz, r))
            self.dynamic_obstacles.append((cx, cy, cz, r, vx, vy, vz))

    # ------------------------------------------------------------------
    # Dynamic obstacle update
    # ------------------------------------------------------------------

    def update_dynamic_obstacles(self, dt: float = 1.0) -> None:
        """Advance dynamic obstacles by *dt* seconds, bouncing off walls.

        Each obstacle's centre is moved by ``(vx*dt, vy*dt, vz*dt)``.  If the
        obstacle would exceed the environment bounds (accounting for its
        radius), the velocity component is reversed and the position is
        clamped.

        Args:
            dt: Time-step in seconds.
        """
        updated: list[tuple[float, float, float, float, float, float, float]] = []
        for cx, cy, cz, r, vx, vy, vz in self.dynamic_obstacles:
            new_cx = cx + vx * dt
            new_cy = cy + vy * dt
            new_cz = cz + vz * dt

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

            if new_cz - r < 0:
                new_cz = r
                vz = abs(vz)
            elif new_cz + r > self.depth:
                new_cz = self.depth - r
                vz = -abs(vz)

            updated.append((new_cx, new_cy, new_cz, r, vx, vy, vz))
        self.dynamic_obstacles = updated

    # ------------------------------------------------------------------
    # Collision queries
    # ------------------------------------------------------------------

    def is_point_in_obstacle(self, x: float, y: float, z: float) -> bool:
        """Return ``True`` if the point *(x, y, z)* lies inside any obstacle."""
        for cx, cy, cz, r in self.get_obstacle_positions():
            if np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) <= r:
                return True
        return False

    def is_path_clear(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check whether the line segment *p1* → *p2* is free of obstacles.

        Uses the analytic line–sphere intersection test: for each sphere
        centred at *c* with radius *r*, the closest point on the segment to
        *c* is computed via projection, and its distance is compared with *r*.

        Args:
            p1: Start point, shape ``(3,)``.
            p2: End point, shape ``(3,)``.

        Returns:
            ``True`` if the segment does **not** intersect any obstacle.
        """
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        d = p2 - p1
        seg_len_sq = float(np.dot(d, d))

        for cx, cy, cz, r in self.get_obstacle_positions():
            c = np.array([cx, cy, cz], dtype=np.float64)
            if seg_len_sq == 0.0:
                if np.sqrt(
                    (p1[0] - cx) ** 2 + (p1[1] - cy) ** 2 + (p1[2] - cz) ** 2
                ) <= r:
                    return False
                continue
            # Parameter t of nearest point on the infinite line
            t = float(np.dot(c - p1, d)) / seg_len_sq
            t = max(0.0, min(1.0, t))
            nearest = p1 + t * d
            dist = np.sqrt(
                (nearest[0] - cx) ** 2
                + (nearest[1] - cy) ** 2
                + (nearest[2] - cz) ** 2
            )
            if dist <= r:
                return False
        return True

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def get_obstacle_positions(self) -> List[Tuple[float, float, float, float]]:
        """Return ``(cx, cy, cz, radius)`` for every obstacle (static + dynamic).

        Dynamic obstacles are returned at their current position.
        """
        positions = list(self.obstacles)
        for cx, cy, cz, r, _vx, _vy, _vz in self.dynamic_obstacles:
            positions.append((cx, cy, cz, r))
        return positions

    def point_to_nearest_obstacle_distance(
        self, x: float, y: float, z: float
    ) -> float:
        """Return the minimum distance from *(x, y, z)* to the nearest obstacle surface.

        The distance is measured from the point to the sphere boundary, i.e.
        ``dist_to_centre - radius``.  A negative value means the point is
        inside an obstacle.  Returns ``np.inf`` when there are no obstacles.

        Args:
            x: X coordinate.
            y: Y coordinate.
            z: Z coordinate.

        Returns:
            Signed distance to nearest obstacle surface.
        """
        obstacles = self.get_obstacle_positions()
        if not obstacles:
            return np.inf
        min_dist = np.inf
        for cx, cy, cz, r in obstacles:
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - r
            if dist < min_dist:
                min_dist = dist
        return float(min_dist)

    def get_bounds(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Return the environment bounds as ``((0, width), (0, height), (0, depth))``."""
        return ((0.0, self.width), (0.0, self.height), (0.0, self.depth))
