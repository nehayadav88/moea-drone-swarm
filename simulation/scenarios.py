"""Predefined 3-D scenario factory for drone-swarm path planning experiments.

Each scenario returns a configured :class:`Environment3D` together with a list
of :class:`Drone` instances whose start and target positions are 3-D numpy
arrays of shape ``(3,)``.

Use :func:`create_scenario` as the single entry-point, or call individual
``create_scenario_*`` builder functions directly.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from simulation.drone import Drone
from simulation.environment3d import Environment3D

# All publicly available scenario names.
_SCENARIO_NAMES: list[str] = [
    "urban_canyon",
    "dense_forest",
    "open_field",
    "corridor",
    "multi_layer",
    "search_and_rescue",
    "high_density",
    "custom",
]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def list_scenarios() -> list[str]:
    """Return a sorted list of available scenario names."""
    return sorted(_SCENARIO_NAMES)


def create_scenario(
    name: str, seed: int | None = None, **kwargs: Any
) -> tuple[Environment3D, list[Drone]]:
    """Create a named scenario.  Returns ``(environment, drones)``.

    Available scenarios: urban_canyon, dense_forest, open_field, corridor,
    multi_layer, search_and_rescue, high_density, custom.

    Args:
        name: Scenario identifier (case-insensitive, hyphens accepted).
        seed: Optional RNG seed for reproducibility.
        **kwargs: Forwarded to the underlying builder function.

    Returns:
        A tuple of (:class:`Environment3D`, list[:class:`Drone`]).

    Raises:
        ValueError: If *name* is not a recognised scenario.
    """
    normalised = name.lower().replace("-", "_")
    builders = {
        "urban_canyon": create_scenario_urban_canyon,
        "dense_forest": create_scenario_dense_forest,
        "open_field": create_scenario_open_field,
        "corridor": create_scenario_corridor,
        "multi_layer": create_scenario_multi_layer,
        "search_and_rescue": create_scenario_search_and_rescue,
        "high_density": create_scenario_high_density,
        "custom": create_scenario_custom,
    }
    if normalised not in builders:
        available = ", ".join(sorted(builders))
        raise ValueError(
            f"Unknown scenario {name!r}. Available: {available}"
        )
    return builders[normalised](seed=seed, **kwargs)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _random_drones_3d(
    rng: np.random.Generator,
    n_drones: int,
    env: Environment3D,
    *,
    start_margin: float = 5.0,
    target_margin: float = 5.0,
) -> list[Drone]:
    """Create *n_drones* with random 3-D start / target positions inside *env*.

    Drone physical parameters are heterogeneous (same ranges as
    :meth:`Drone.create_heterogeneous_swarm`).
    """
    w, h, d = env.width, env.height, env.depth
    drones: list[Drone] = []
    for i in range(n_drones):
        start = rng.uniform(
            [start_margin, start_margin, start_margin],
            [w - start_margin, h - start_margin, d - start_margin],
        ).astype(np.float64)
        target = rng.uniform(
            [target_margin, target_margin, target_margin],
            [w - target_margin, h - target_margin, d - target_margin],
        ).astype(np.float64)
        drones.append(
            Drone(
                drone_id=i,
                start_position=start,
                target_position=target,
                battery_capacity=rng.uniform(80.0, 150.0),
                max_speed=rng.uniform(10.0, 25.0),
                sensing_range=rng.uniform(15.0, 30.0),
                comm_range=rng.uniform(40.0, 80.0),
                mass=rng.uniform(1.5, 3.5),
                payload=rng.uniform(0.0, 1.0),
            )
        )
    return drones


def _place_non_overlapping(
    rng: np.random.Generator,
    count: int,
    bounds: tuple[float, float, float],
    radius_range: tuple[float, float],
    margin: float,
    overlap_margin: float = 1.0,
    placed: list[tuple[float, float, float, float]] | None = None,
) -> list[tuple[float, float, float, float]]:
    """Place *count* non-overlapping spheres within *bounds*.

    Returns list of ``(cx, cy, cz, radius)`` tuples.
    """
    w, h, d = bounds
    min_r, max_r = radius_range
    existing: list[tuple[float, float, float, float]] = list(placed) if placed else []
    result: list[tuple[float, float, float, float]] = []
    for _ in range(count):
        r = rng.uniform(min_r, max_r)
        success = False
        for _ in range(500):
            cx = rng.uniform(margin, w - margin)
            cy = rng.uniform(margin, h - margin)
            cz = rng.uniform(margin, d - margin)
            if all(
                np.sqrt((cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2)
                > r + pr + overlap_margin
                for px, py, pz, pr in existing
            ):
                existing.append((cx, cy, cz, r))
                result.append((cx, cy, cz, r))
                success = True
                break
        if not success:
            # Could not place — skip silently (mirrors Environment behaviour)
            pass
    return result


# ------------------------------------------------------------------
# Scenario builders
# ------------------------------------------------------------------


def create_scenario_urban_canyon(
    seed: int | None = None, **kwargs: Any
) -> tuple[Environment3D, list[Drone]]:
    """Tall building-like vertical columns with drones navigating between them.

    Buildings are approximated as columns of vertically stacked spheres.

    Args:
        seed: Optional RNG seed.
        **kwargs: Overrides — *n_drones* (default 8), *n_buildings* (default 8).

    Returns:
        ``(environment, drones)``
    """
    rng = np.random.default_rng(seed)
    n_drones: int = kwargs.get("n_drones", 8)
    n_buildings: int = kwargs.get("n_buildings", 8)
    width, height, depth = 120.0, 120.0, 60.0

    obstacles: list[tuple[float, float, float, float]] = []
    placed_columns: list[tuple[float, float]] = []
    col_radius = 3.5

    for _ in range(n_buildings):
        for _attempt in range(200):
            cx = rng.uniform(15.0, width - 15.0)
            cy = rng.uniform(15.0, height - 15.0)
            if all(
                np.hypot(cx - px, cy - py) > 2 * col_radius + 4.0
                for px, py in placed_columns
            ):
                placed_columns.append((cx, cy))
                # Stack spheres from ground to a random building height
                building_height = rng.uniform(25.0, depth - 5.0)
                z = col_radius
                while z < building_height:
                    obstacles.append((cx, cy, z, col_radius))
                    z += col_radius * 1.8  # slight overlap for continuity
                break

    # A few dynamic obstacles (e.g. other aircraft)
    dynamic: list[tuple[float, float, float, float, float, float, float]] = []
    for _ in range(2):
        dynamic.append((
            rng.uniform(10.0, width - 10.0),
            rng.uniform(10.0, height - 10.0),
            rng.uniform(10.0, depth - 10.0),
            2.0,
            rng.uniform(-1.5, 1.5),
            rng.uniform(-1.5, 1.5),
            rng.uniform(-0.5, 0.5),
        ))

    env = Environment3D(
        width=width, height=height, depth=depth,
        obstacles=obstacles, dynamic_obstacles=dynamic,
    )
    drones = _random_drones_3d(rng, n_drones, env)
    return env, drones


def create_scenario_dense_forest(
    seed: int | None = None, **kwargs: Any
) -> tuple[Environment3D, list[Drone]]:
    """Many small obstacles at various heights simulating trees.

    Args:
        seed: Optional RNG seed.
        **kwargs: Overrides — *n_drones* (default 5), *n_trees* (default 40).

    Returns:
        ``(environment, drones)``
    """
    rng = np.random.default_rng(seed)
    n_drones: int = kwargs.get("n_drones", 5)
    n_trees: int = kwargs.get("n_trees", 40)
    width, height, depth = 80.0, 80.0, 40.0

    obstacles = _place_non_overlapping(
        rng, n_trees,
        bounds=(width, height, depth),
        radius_range=(0.8, 2.5),
        margin=3.0,
    )

    env = Environment3D(width=width, height=height, depth=depth, obstacles=obstacles)
    drones = _random_drones_3d(rng, n_drones, env)
    return env, drones


def create_scenario_open_field(
    seed: int | None = None, **kwargs: Any
) -> tuple[Environment3D, list[Drone]]:
    """Large open area with few scattered large obstacles and long-range paths.

    Args:
        seed: Optional RNG seed.
        **kwargs: Overrides — *n_drones* (default 10), *n_obstacles* (default 6).

    Returns:
        ``(environment, drones)``
    """
    rng = np.random.default_rng(seed)
    n_drones: int = kwargs.get("n_drones", 10)
    n_obstacles: int = kwargs.get("n_obstacles", 6)
    width, height, depth = 200.0, 200.0, 60.0

    obstacles = _place_non_overlapping(
        rng, n_obstacles,
        bounds=(width, height, depth),
        radius_range=(5.0, 12.0),
        margin=15.0,
    )

    env = Environment3D(width=width, height=height, depth=depth, obstacles=obstacles)
    # Drones start on one side and target the opposite side for long range
    drones: list[Drone] = []
    for i in range(n_drones):
        start = np.array([
            rng.uniform(5.0, 30.0),
            rng.uniform(5.0, height - 5.0),
            rng.uniform(5.0, depth - 5.0),
        ], dtype=np.float64)
        target = np.array([
            rng.uniform(width - 30.0, width - 5.0),
            rng.uniform(5.0, height - 5.0),
            rng.uniform(5.0, depth - 5.0),
        ], dtype=np.float64)
        drones.append(
            Drone(
                drone_id=i,
                start_position=start,
                target_position=target,
                battery_capacity=rng.uniform(100.0, 150.0),
                max_speed=rng.uniform(12.0, 25.0),
                sensing_range=rng.uniform(20.0, 35.0),
                comm_range=rng.uniform(50.0, 100.0),
                mass=rng.uniform(1.5, 3.5),
                payload=rng.uniform(0.0, 1.0),
            )
        )
    return env, drones


def create_scenario_corridor(
    seed: int | None = None, **kwargs: Any
) -> tuple[Environment3D, list[Drone]]:
    """Narrow passage with obstacles lining both sides that drones must fly through.

    Args:
        seed: Optional RNG seed.
        **kwargs: Overrides — *n_drones* (default 4).

    Returns:
        ``(environment, drones)``
    """
    rng = np.random.default_rng(seed)
    n_drones: int = kwargs.get("n_drones", 4)
    width, height, depth = 100.0, 30.0, 30.0

    obstacles: list[tuple[float, float, float, float]] = []
    r = 3.0
    # Walls of obstacles along both Y edges
    for x in np.arange(10.0, width - 10.0, 8.0):
        for z in np.arange(r, depth, 7.0):
            obstacles.append((x, r + 0.5, z, r))          # bottom wall
            obstacles.append((x, height - r - 0.5, z, r))  # top wall

    env = Environment3D(width=width, height=height, depth=depth, obstacles=obstacles)
    # Drones start at left and target the right end of the corridor
    drones: list[Drone] = []
    mid_y = height / 2.0
    mid_z = depth / 2.0
    for i in range(n_drones):
        start = np.array([
            5.0,
            mid_y + rng.uniform(-3.0, 3.0),
            mid_z + rng.uniform(-3.0, 3.0),
        ], dtype=np.float64)
        target = np.array([
            width - 5.0,
            mid_y + rng.uniform(-3.0, 3.0),
            mid_z + rng.uniform(-3.0, 3.0),
        ], dtype=np.float64)
        drones.append(
            Drone(
                drone_id=i,
                start_position=start,
                target_position=target,
                battery_capacity=rng.uniform(80.0, 120.0),
                max_speed=rng.uniform(10.0, 18.0),
                sensing_range=rng.uniform(10.0, 20.0),
                comm_range=rng.uniform(30.0, 60.0),
                mass=rng.uniform(1.5, 3.0),
                payload=rng.uniform(0.0, 0.5),
            )
        )
    return env, drones


def create_scenario_multi_layer(
    seed: int | None = None, **kwargs: Any
) -> tuple[Environment3D, list[Drone]]:
    """Obstacles arranged at different altitude layers (ground, mid, high).

    Drones have targets at different heights to exercise vertical path planning.

    Args:
        seed: Optional RNG seed.
        **kwargs: Overrides — *n_drones* (default 8).

    Returns:
        ``(environment, drones)``
    """
    rng = np.random.default_rng(seed)
    n_drones: int = kwargs.get("n_drones", 8)
    width, height, depth = 100.0, 100.0, 60.0

    obstacles: list[tuple[float, float, float, float]] = []
    layers = [
        (5.0, 2.0, 4.0),    # ground layer: z ≈ 5, radii 2–4
        (30.0, 2.0, 3.5),   # mid layer: z ≈ 30
        (50.0, 1.5, 3.0),   # high layer: z ≈ 50
    ]
    for z_centre, r_min, r_max in layers:
        n_per_layer = 8
        for _ in range(n_per_layer):
            r = rng.uniform(r_min, r_max)
            for _attempt in range(200):
                cx = rng.uniform(r + 5.0, width - r - 5.0)
                cy = rng.uniform(r + 5.0, height - r - 5.0)
                cz = z_centre + rng.uniform(-3.0, 3.0)
                cz = np.clip(cz, r, depth - r)
                if all(
                    np.sqrt(
                        (cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2
                    )
                    > r + pr + 1.0
                    for px, py, pz, pr in obstacles
                ):
                    obstacles.append((cx, cy, float(cz), r))
                    break

    env = Environment3D(width=width, height=height, depth=depth, obstacles=obstacles)
    # Alternate drone targets between low and high altitudes
    drones: list[Drone] = []
    for i in range(n_drones):
        start_z = rng.choice([5.0, 25.0, 45.0]) + rng.uniform(-2.0, 2.0)
        target_z = rng.choice([5.0, 25.0, 45.0]) + rng.uniform(-2.0, 2.0)
        start = np.array([
            rng.uniform(5.0, width - 5.0),
            rng.uniform(5.0, height - 5.0),
            np.clip(start_z, 2.0, depth - 2.0),
        ], dtype=np.float64)
        target = np.array([
            rng.uniform(5.0, width - 5.0),
            rng.uniform(5.0, height - 5.0),
            np.clip(target_z, 2.0, depth - 2.0),
        ], dtype=np.float64)
        drones.append(
            Drone(
                drone_id=i,
                start_position=start,
                target_position=target,
                battery_capacity=rng.uniform(90.0, 150.0),
                max_speed=rng.uniform(10.0, 22.0),
                sensing_range=rng.uniform(15.0, 30.0),
                comm_range=rng.uniform(40.0, 80.0),
                mass=rng.uniform(1.5, 3.5),
                payload=rng.uniform(0.0, 1.0),
            )
        )
    return env, drones


def create_scenario_search_and_rescue(
    seed: int | None = None, **kwargs: Any
) -> tuple[Environment3D, list[Drone]]:
    """Large area with scattered obstacles and multiple target zones.

    Drones are spread across the environment and assigned to distinct target
    zones, simulating a search-and-rescue deployment.

    Args:
        seed: Optional RNG seed.
        **kwargs: Overrides — *n_drones* (default 12), *n_obstacles* (default 20).

    Returns:
        ``(environment, drones)``
    """
    rng = np.random.default_rng(seed)
    n_drones: int = kwargs.get("n_drones", 12)
    n_obstacles: int = kwargs.get("n_obstacles", 20)
    width, height, depth = 150.0, 150.0, 50.0

    obstacles = _place_non_overlapping(
        rng, n_obstacles,
        bounds=(width, height, depth),
        radius_range=(1.5, 5.0),
        margin=6.0,
    )

    # Define target zones (centres of interest in the environment)
    n_zones = 4
    zone_centres = [
        np.array([rng.uniform(20.0, width - 20.0),
                   rng.uniform(20.0, height - 20.0),
                   rng.uniform(5.0, depth - 5.0)], dtype=np.float64)
        for _ in range(n_zones)
    ]

    env = Environment3D(width=width, height=height, depth=depth, obstacles=obstacles)
    drones: list[Drone] = []
    for i in range(n_drones):
        start = np.array([
            rng.uniform(5.0, width - 5.0),
            rng.uniform(5.0, height - 5.0),
            rng.uniform(2.0, depth - 2.0),
        ], dtype=np.float64)
        # Assign drone to a random target zone with small jitter
        zone = zone_centres[i % n_zones]
        target = zone + rng.uniform(-10.0, 10.0, size=3)
        target = np.clip(target, 2.0, [width - 2.0, height - 2.0, depth - 2.0])
        drones.append(
            Drone(
                drone_id=i,
                start_position=start,
                target_position=target.astype(np.float64),
                battery_capacity=rng.uniform(100.0, 150.0),
                max_speed=rng.uniform(12.0, 25.0),
                sensing_range=rng.uniform(20.0, 35.0),
                comm_range=rng.uniform(50.0, 100.0),
                mass=rng.uniform(1.5, 3.5),
                payload=rng.uniform(0.0, 1.0),
            )
        )
    return env, drones


def create_scenario_high_density(
    seed: int | None = None, **kwargs: Any
) -> tuple[Environment3D, list[Drone]]:
    """Stress test with 20 drones and many obstacles.

    Args:
        seed: Optional RNG seed.
        **kwargs: Overrides — *n_drones* (default 20), *n_static* (default 40),
            *n_dynamic* (default 10).

    Returns:
        ``(environment, drones)``
    """
    rng = np.random.default_rng(seed)
    n_drones: int = kwargs.get("n_drones", 20)
    n_static: int = kwargs.get("n_static", 40)
    n_dynamic: int = kwargs.get("n_dynamic", 10)
    width, height, depth = 100.0, 100.0, 50.0

    env = Environment3D(width=width, height=height, depth=depth)
    env.generate_random_obstacles(
        n_static=n_static,
        n_dynamic=n_dynamic,
        min_radius=1.0,
        max_radius=4.0,
        seed=rng.integers(0, 2**31),
    )
    drones = _random_drones_3d(rng, n_drones, env)
    return env, drones


def create_scenario_custom(
    seed: int | None = None, **kwargs: Any
) -> tuple[Environment3D, list[Drone]]:
    """Fully user-configurable scenario.

    All environment and drone parameters are taken from *kwargs*.

    Keyword Args:
        width: Environment X extent (default 100).
        height: Environment Y extent (default 100).
        depth: Environment Z extent (default 50).
        obstacles: List of ``(cx, cy, cz, radius)`` static obstacles.
        dynamic_obstacles: List of ``(cx, cy, cz, radius, vx, vy, vz)``
            dynamic obstacles.
        n_random_static: If given, generate this many random static obstacles.
        n_random_dynamic: If given, generate this many random dynamic obstacles.
        n_drones: Number of random drones (default 5).  Ignored if *drones*
            is provided.
        drones: Explicit list of :class:`Drone` instances.

    Returns:
        ``(environment, drones)``
    """
    rng = np.random.default_rng(seed)
    width: float = kwargs.get("width", 100.0)
    height: float = kwargs.get("height", 100.0)
    depth: float = kwargs.get("depth", 50.0)

    env = Environment3D(
        width=width,
        height=height,
        depth=depth,
        obstacles=kwargs.get("obstacles"),
        dynamic_obstacles=kwargs.get("dynamic_obstacles"),
    )

    n_random_static = kwargs.get("n_random_static")
    n_random_dynamic = kwargs.get("n_random_dynamic")
    if n_random_static is not None or n_random_dynamic is not None:
        env.generate_random_obstacles(
            n_static=n_random_static or 0,
            n_dynamic=n_random_dynamic or 0,
            seed=rng.integers(0, 2**31),
        )

    if "drones" in kwargs:
        drones: list[Drone] = kwargs["drones"]
    else:
        n_drones: int = kwargs.get("n_drones", 5)
        drones = _random_drones_3d(rng, n_drones, env)

    return env, drones
