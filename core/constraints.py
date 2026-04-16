"""Constraint handling utilities for drone swarm path planning.

Provides functions for collision detection, energy feasibility checks,
communication connectivity verification, and safety penalty computation
for both 2-D and 3-D environments.  All geometric computations use numpy
for efficiency.
"""

import numpy as np


def line_segment_circle_distance(p1, p2, center, radius):
    """Compute the minimum distance from a line segment to a circle boundary.

    Parameters
    ----------
    p1 : array-like, shape (2,)
        Start point of the line segment.
    p2 : array-like, shape (2,)
        End point of the line segment.
    center : array-like, shape (2,)
        Center of the circle.
    radius : float
        Radius of the circle.

    Returns
    -------
    float
        Minimum distance from the line segment to the circle boundary.
        Returns 0.0 if the segment intersects the circle.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    d = p2 - p1
    f = p1 - center

    seg_len_sq = np.dot(d, d)

    if seg_len_sq < 1e-12:
        # Degenerate segment (point)
        dist_to_center = np.linalg.norm(p1 - center)
        return max(0.0, dist_to_center - radius)

    # Parameter t for closest point on the infinite line to center
    t = -np.dot(f, d) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)

    closest = p1 + t * d
    dist_to_center = np.linalg.norm(closest - center)

    return max(0.0, dist_to_center - radius)


def check_path_collision(waypoints, obstacles, drone_radius=0.5):
    """Check if any segment of a path intersects an obstacle.

    Uses line-circle intersection tests between each consecutive pair
    of waypoints and each obstacle (expanded by drone_radius).

    Parameters
    ----------
    waypoints : array-like, shape (N, 2)
        Ordered list of 2D waypoints defining the path.
    obstacles : list of tuple
        Each obstacle is ``(cx, cy, radius)`` defining a circular obstacle.
    drone_radius : float, optional
        Effective radius of the drone for collision buffering (default 0.5).

    Returns
    -------
    has_collision : bool
        True if any segment collides with any obstacle.
    num_collisions : int
        Total number of segment–obstacle collision pairs detected.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if len(waypoints) < 2 or len(obstacles) == 0:
        return False, 0

    num_collisions = 0
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        for cx, cy, r in obstacles:
            dist = line_segment_circle_distance(p1, p2, np.array([cx, cy]), r)
            if dist <= drone_radius:
                num_collisions += 1

    return num_collisions > 0, num_collisions


def check_energy_feasibility(energy, battery_capacity):
    """Return the energy constraint violation amount.

    Parameters
    ----------
    energy : float
        Energy consumed by the drone.
    battery_capacity : float
        Maximum energy the drone's battery can provide.

    Returns
    -------
    float
        Violation amount: ``max(0, energy - battery_capacity)``.
        Zero means the constraint is satisfied.
    """
    return max(0.0, energy - battery_capacity)


def check_communication(drone_positions_at_step, comm_range):
    """Check if the communication graph among drones is connected.

    Builds an adjacency graph where two drones are connected if their
    Euclidean distance is at most ``comm_range``, then checks graph
    connectivity via BFS.

    Parameters
    ----------
    drone_positions_at_step : array-like, shape (num_drones, 2)
        Positions of all drones at a single timestep.
    comm_range : float
        Maximum communication range between two drones.

    Returns
    -------
    connected : bool
        True if the communication graph is fully connected.
    num_disconnected : int
        Number of connected components minus one. Zero means fully connected.
    """
    positions = np.asarray(drone_positions_at_step, dtype=np.float64)
    n = len(positions)

    if n <= 1:
        return True, 0

    # Build adjacency list using pairwise distances
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= comm_range:
                adjacency[i].append(j)
                adjacency[j].append(i)

    # BFS to count connected components
    visited = np.zeros(n, dtype=bool)
    num_components = 0

    for start in range(n):
        if visited[start]:
            continue
        num_components += 1
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

    return num_components == 1, num_components - 1


def compute_safety_penalty(waypoints, obstacles, safety_distance=2.0):
    """Compute a penalty for path segments that pass too close to obstacles.

    For each path segment and each obstacle, if the minimum distance from
    the segment to the obstacle boundary is less than ``safety_distance``,
    a penalty proportional to the penetration depth is accumulated.

    Parameters
    ----------
    waypoints : array-like, shape (N, 2)
        Ordered list of 2D waypoints defining the path.
    obstacles : list of tuple
        Each obstacle is ``(cx, cy, radius)`` defining a circular obstacle.
    safety_distance : float, optional
        Minimum desired clearance from obstacle boundaries (default 2.0).

    Returns
    -------
    float
        Total accumulated safety penalty. Zero if all segments maintain
        at least ``safety_distance`` clearance from all obstacles.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if len(waypoints) < 2 or len(obstacles) == 0:
        return 0.0

    penalty = 0.0
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        for cx, cy, r in obstacles:
            dist = line_segment_circle_distance(p1, p2, np.array([cx, cy]), r)
            if dist < safety_distance:
                penalty += safety_distance - dist

    return penalty


# =====================================================================
# 3-D variants (spherical obstacles)
# =====================================================================


def line_segment_sphere_distance(p1, p2, center, radius):
    """Compute minimum distance from a 3-D line segment to a sphere boundary.

    Parameters
    ----------
    p1 : array-like, shape (3,)
        Segment start.
    p2 : array-like, shape (3,)
        Segment end.
    center : array-like, shape (3,)
        Sphere centre.
    radius : float
        Sphere radius.

    Returns
    -------
    float
        Minimum distance (0 if the segment intersects the sphere).
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    d = p2 - p1
    f = p1 - center
    seg_len_sq = np.dot(d, d)

    if seg_len_sq < 1e-12:
        dist_to_center = np.linalg.norm(p1 - center)
        return max(0.0, dist_to_center - radius)

    t = -np.dot(f, d) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)

    closest = p1 + t * d
    dist_to_center = np.linalg.norm(closest - center)
    return max(0.0, dist_to_center - radius)


def check_path_collision_3d(waypoints, obstacles, drone_radius=0.5):
    """Check if a 3-D path collides with any spherical obstacle.

    Parameters
    ----------
    waypoints : array-like, shape (N, 3)
        Ordered 3-D waypoints.
    obstacles : list of tuple
        ``(cx, cy, cz, radius)`` for each spherical obstacle.
    drone_radius : float, optional
        Effective collision radius of the drone (default 0.5).

    Returns
    -------
    has_collision : bool
    num_collisions : int
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if len(waypoints) < 2 or len(obstacles) == 0:
        return False, 0

    num_collisions = 0
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        for obs in obstacles:
            cx, cy, cz, r = obs[0], obs[1], obs[2], obs[3]
            dist = line_segment_sphere_distance(
                p1, p2, np.array([cx, cy, cz]), r
            )
            if dist <= drone_radius:
                num_collisions += 1

    return num_collisions > 0, num_collisions


def compute_safety_penalty_3d(waypoints, obstacles, safety_distance=2.0):
    """Compute safety penalty for 3-D paths near spherical obstacles.

    Parameters
    ----------
    waypoints : array-like, shape (N, 3)
        Ordered 3-D waypoints.
    obstacles : list of tuple
        ``(cx, cy, cz, radius)`` for each obstacle.
    safety_distance : float, optional
        Minimum desired clearance (default 2.0).

    Returns
    -------
    float
        Total accumulated safety penalty.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if len(waypoints) < 2 or len(obstacles) == 0:
        return 0.0

    penalty = 0.0
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        for obs in obstacles:
            cx, cy, cz, r = obs[0], obs[1], obs[2], obs[3]
            dist = line_segment_sphere_distance(
                p1, p2, np.array([cx, cy, cz]), r
            )
            if dist < safety_distance:
                penalty += safety_distance - dist

    return penalty
