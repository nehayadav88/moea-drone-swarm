"""Path planning utilities for drone swarm waypoint paths.

All functions operate on waypoints represented as numpy arrays of shape (n, 2).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from simulation.environment import Environment


# ------------------------------------------------------------------
# Path smoothing
# ------------------------------------------------------------------


def smooth_path(
    waypoints: np.ndarray,
    environment: Environment,
    iterations: int = 50,
) -> np.ndarray:
    """Iteratively smooth a path while keeping it collision-free.

    For each interior waypoint the algorithm tries to move it toward the
    midpoint of its two neighbours.  The move is accepted only if both
    adjacent segments remain clear of obstacles.

    Args:
        waypoints: Array of shape (n, 2) — the path to smooth.
        environment: The simulation environment used for collision checks.
        iterations: Number of smoothing passes over the interior points.

    Returns:
        Smoothed waypoints as an array of shape (n, 2).
    """
    path = np.array(waypoints, dtype=np.float64)
    n = len(path)
    if n <= 2:
        return path

    for _ in range(iterations):
        for i in range(1, n - 1):
            midpoint = 0.5 * (path[i - 1] + path[i + 1])
            candidate = 0.5 * (path[i] + midpoint)  # move halfway toward midpoint
            if environment.is_path_clear(path[i - 1], candidate) and environment.is_path_clear(
                candidate, path[i + 1]
            ):
                path[i] = candidate
    return path


# ------------------------------------------------------------------
# Path repair
# ------------------------------------------------------------------


def repair_path(
    waypoints: np.ndarray,
    environment: Environment,
    max_attempts: int = 100,
) -> np.ndarray:
    """Attempt to repair a path that intersects obstacles.

    Interior waypoints that participate in a colliding segment are perturbed
    away from the nearest obstacle.  The process repeats until the path is
    collision-free or *max_attempts* is exhausted.

    Args:
        waypoints: Array of shape (n, 2).
        environment: The simulation environment.
        max_attempts: Maximum number of repair iterations.

    Returns:
        Repaired waypoints as an array of shape (n, 2).
    """
    path = np.array(waypoints, dtype=np.float64)
    n = len(path)
    if n <= 1:
        return path

    rng = np.random.default_rng(42)

    for _ in range(max_attempts):
        feasible, collision_indices = check_path_feasibility(path, environment)
        if feasible:
            break

        visited: set[int] = set()
        for seg_idx in collision_indices:
            for wp_idx in (seg_idx, seg_idx + 1):
                if wp_idx == 0 or wp_idx == n - 1:
                    continue  # don't move start/end
                if wp_idx in visited:
                    continue
                visited.add(wp_idx)

                x, y = path[wp_idx]
                obstacles = environment.get_obstacle_positions()
                if not obstacles:
                    continue

                # Find nearest obstacle centre and push waypoint away
                best_dist = np.inf
                best_cx, best_cy, best_r = 0.0, 0.0, 1.0
                for cx, cy, r in obstacles:
                    d = np.hypot(x - cx, y - cy)
                    if d < best_dist:
                        best_dist = d
                        best_cx, best_cy, best_r = cx, cy, r

                # Direction away from obstacle centre
                dx = x - best_cx
                dy = y - best_cy
                norm = np.hypot(dx, dy)
                if norm < 1e-12:
                    dx, dy = rng.normal(), rng.normal()
                    norm = np.hypot(dx, dy)
                dx /= norm
                dy /= norm

                push = best_r + 1.0  # push outside obstacle + margin
                path[wp_idx, 0] = best_cx + dx * push
                path[wp_idx, 1] = best_cy + dy * push

                # Clamp to environment bounds
                (x_lo, x_hi), (y_lo, y_hi) = environment.get_bounds()
                path[wp_idx, 0] = np.clip(path[wp_idx, 0], x_lo, x_hi)
                path[wp_idx, 1] = np.clip(path[wp_idx, 1], y_lo, y_hi)

    return path


# ------------------------------------------------------------------
# Path interpolation
# ------------------------------------------------------------------


def interpolate_path(
    waypoints: np.ndarray,
    num_points: int = 50,
) -> np.ndarray:
    """Linearly interpolate between waypoints to produce a dense path.

    Args:
        waypoints: Array of shape (n, 2).
        num_points: Total number of points in the output path.

    Returns:
        Interpolated path as an array of shape (num_points, 2).
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if len(waypoints) < 2:
        return np.tile(waypoints, (num_points, 1)) if len(waypoints) == 1 else waypoints.copy()

    # Cumulative arc-length parameterisation
    diffs = np.diff(waypoints, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_length = cum_lengths[-1]

    if total_length == 0.0:
        return np.tile(waypoints[0], (num_points, 1))

    # Evenly spaced parameter values
    t_values = np.linspace(0.0, total_length, num_points)
    result = np.empty((num_points, 2), dtype=np.float64)

    seg_idx = 0
    n_segs = len(seg_lengths)
    for i, t in enumerate(t_values):
        while seg_idx < n_segs - 1 and t > cum_lengths[seg_idx + 1]:
            seg_idx += 1
        seg_start = cum_lengths[seg_idx]
        seg_end = cum_lengths[seg_idx + 1]
        seg_len = seg_end - seg_start
        alpha = (t - seg_start) / seg_len if seg_len > 0 else 0.0
        result[i] = waypoints[seg_idx] + alpha * diffs[seg_idx]

    return result


# ------------------------------------------------------------------
# Path metrics
# ------------------------------------------------------------------


def compute_path_length(waypoints: np.ndarray) -> float:
    """Return the total Euclidean length of the path.

    Args:
        waypoints: Array of shape (n, 2).

    Returns:
        Sum of segment lengths.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if len(waypoints) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))


def compute_path_angles(waypoints: np.ndarray) -> np.ndarray:
    """Compute turning angles at each interior waypoint.

    The turning angle at waypoint *i* (1 ≤ i ≤ n−2) is the angle between
    the incoming segment (i−1 → i) and the outgoing segment (i → i+1),
    measured in radians in [0, π].

    Args:
        waypoints: Array of shape (n, 2).

    Returns:
        Array of shape (n−2,) with the turning angle at each interior point.
        Returns an empty array when there are fewer than 3 waypoints.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    n = len(waypoints)
    if n < 3:
        return np.array([], dtype=np.float64)

    angles = np.empty(n - 2, dtype=np.float64)
    for i in range(1, n - 1):
        v1 = waypoints[i] - waypoints[i - 1]
        v2 = waypoints[i + 1] - waypoints[i]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles[i - 1] = np.arccos(cos_angle)
    return angles


# ------------------------------------------------------------------
# Feasibility check
# ------------------------------------------------------------------


def check_path_feasibility(
    waypoints: np.ndarray,
    environment: Environment,
) -> Tuple[bool, List[int]]:
    """Check whether the entire path is collision-free.

    Args:
        waypoints: Array of shape (n, 2).
        environment: The simulation environment.

    Returns:
        A tuple ``(feasible, collision_segment_indices)`` where *feasible*
        is True when the path has no collisions, and
        *collision_segment_indices* lists the indices of segments that
        intersect an obstacle (segment *i* connects waypoint *i* to *i+1*).
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    collision_indices: list[int] = []
    for i in range(len(waypoints) - 1):
        if not environment.is_path_clear(waypoints[i], waypoints[i + 1]):
            collision_indices.append(i)
    return len(collision_indices) == 0, collision_indices
