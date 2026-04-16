"""Multi-objective problem definition for drone swarm path planning.

Defines continuous multi-objective optimisation problems (MOPs) where
decision variables encode intermediate waypoints for each drone and the
objectives capture path length, flight time, energy, and safety.

Provides both 2-D (:class:`DroneSwarmMOP`) and 3-D (:class:`DroneSwarmMOP3D`)
formulations.
"""

import numpy as np

from .energy_model import EnergyModel
from .constraints import (
    check_path_collision,
    check_path_collision_3d,
    check_energy_feasibility,
    check_communication,
    compute_safety_penalty,
    compute_safety_penalty_3d,
)


class DroneSwarmMOP:
    """Multi-objective drone swarm path-planning problem.

    Decision variables
    ------------------
    A flat array of ``(x, y)`` intermediate waypoints for every drone.
    For *D* drones each with *W* intermediate waypoints the total number
    of decision variables is ``D * W * 2``.

    Objectives (to minimise)
    ------------------------
    1. **Total path length** – sum of Euclidean segment lengths across all drones.
    2. **Total flight time** – sum of ``path_length / speed`` per drone.
    3. **Total energy consumption** – computed via :class:`EnergyModel`.
    4. **Safety penalty** – accumulated proximity penalty to obstacles.

    Constraints
    -----------
    - Energy feasibility per drone (energy ≤ battery capacity).
    - Path feasibility (no obstacle collisions).
    - Communication connectivity at each waypoint index.

    Parameters
    ----------
    environment : object
        Must expose:

        - ``width`` (float): environment X dimension.
        - ``height`` (float): environment Y dimension.
        - ``obstacles`` (list of tuple): each ``(cx, cy, radius)``.

    drones : list of dict
        Each dict must contain:

        - ``start`` (tuple): ``(x, y)`` start position.
        - ``target`` (tuple): ``(x, y)`` target position.
        - ``mass`` (float): drone mass in kg.
        - ``payload`` (float): payload mass in kg.
        - ``speed`` (float): cruise speed in m/s.
        - ``battery_capacity`` (float): max energy in joules.

    task_positions : list of tuple
        Task/target positions (informational; per-drone targets come from
        the *drones* list).
    config : dict, optional
        Optional overrides:

        - ``num_waypoints_per_drone`` (int, default 5)
        - ``safety_distance`` (float, default 2.0)
        - ``comm_range`` (float, default 50.0)
        - ``drone_radius`` (float, default 0.5)
        - Any ``EnergyModel`` kwargs under an ``"energy_model"`` sub-dict.
    """

    def __init__(self, environment, drones, task_positions, config=None):
        self.environment = environment
        self.drones = drones
        self.task_positions = task_positions
        self.config = config or {}

        self.num_waypoints_per_drone = self.config.get("num_waypoints_per_drone", 5)
        self.safety_distance = self.config.get("safety_distance", 2.0)
        self.comm_range = self.config.get("comm_range", 50.0)
        self.drone_radius = self.config.get("drone_radius", 0.5)

        energy_kwargs = self.config.get("energy_model", {})
        self.energy_model = EnergyModel(**energy_kwargs)

    # ------------------------------------------------------------------
    # Problem dimensions
    # ------------------------------------------------------------------

    @property
    def n_var(self):
        """Total number of decision variables."""
        return len(self.drones) * self.num_waypoints_per_drone * 2

    @property
    def n_obj(self):
        """Number of objectives."""
        return 4

    @property
    def bounds(self):
        """Lower and upper bounds for every decision variable.

        Returns
        -------
        lower : ndarray, shape (n_var,)
        upper : ndarray, shape (n_var,)
        """
        lower = np.zeros(self.n_var)
        upper = np.zeros(self.n_var)
        for i in range(self.n_var):
            if i % 2 == 0:
                lower[i] = 0.0
                upper[i] = self.environment.width
            else:
                lower[i] = 0.0
                upper[i] = self.environment.height
        return lower, upper

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_solution(self, x):
        """Convert a flat decision-variable array into per-drone waypoint lists.

        Each drone's full path is ``[start] + intermediate_waypoints + [target]``.

        Parameters
        ----------
        x : array-like, shape (n_var,)
            Flat array of decision variables.

        Returns
        -------
        list of ndarray
            One ``(num_waypoints_per_drone + 2, 2)`` array per drone.
        """
        x = np.asarray(x, dtype=np.float64)
        n_wp = self.num_waypoints_per_drone
        paths = []

        for d_idx, drone in enumerate(self.drones):
            offset = d_idx * n_wp * 2
            raw = x[offset: offset + n_wp * 2].reshape(n_wp, 2)

            start = np.array(drone["start"], dtype=np.float64).reshape(1, 2)
            target = np.array(drone["target"], dtype=np.float64).reshape(1, 2)

            full_path = np.vstack([start, raw, target])
            paths.append(full_path)

        return paths

    # ------------------------------------------------------------------
    # Objectives
    # ------------------------------------------------------------------

    def evaluate(self, x):
        """Evaluate objectives for a candidate solution.

        Parameters
        ----------
        x : array-like, shape (n_var,)
            Decision variable vector.

        Returns
        -------
        tuple of float
            ``(total_path_length, total_flight_time, total_energy, safety_penalty)``
        """
        paths = self.decode_solution(x)
        obstacles = self.environment.obstacles

        total_length = 0.0
        total_time = 0.0
        total_energy = 0.0
        total_safety = 0.0

        for d_idx, (path, drone) in enumerate(zip(paths, self.drones)):
            segments = np.diff(path, axis=0)
            seg_lengths = np.linalg.norm(segments, axis=1)
            path_length = np.sum(seg_lengths)

            total_length += path_length
            total_time += path_length / max(drone["speed"], 1e-9)

            drone_params = {
                "mass": drone["mass"],
                "payload": drone["payload"],
                "speed": drone["speed"],
            }
            total_energy += self.energy_model.compute_path_energy(path, drone_params)

            total_safety += compute_safety_penalty(
                path, obstacles, self.safety_distance
            )

        return total_length, total_time, total_energy, total_safety

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def evaluate_constraints(self, x):
        """Evaluate constraint violations for a candidate solution.

        Parameters
        ----------
        x : array-like, shape (n_var,)
            Decision variable vector.

        Returns
        -------
        list of float
            Constraint violations (all ≥ 0; 0 means satisfied):

            - One energy-feasibility violation per drone.
            - One collision-count violation per drone.
            - One communication-connectivity violation per waypoint index.
        """
        paths = self.decode_solution(x)
        obstacles = self.environment.obstacles
        violations = []

        # --- Energy feasibility per drone ---
        for path, drone in zip(paths, self.drones):
            drone_params = {
                "mass": drone["mass"],
                "payload": drone["payload"],
                "speed": drone["speed"],
            }
            energy = self.energy_model.compute_path_energy(path, drone_params)
            violations.append(
                check_energy_feasibility(energy, drone["battery_capacity"])
            )

        # --- Path collision feasibility per drone ---
        for path in paths:
            _, n_col = check_path_collision(path, obstacles, self.drone_radius)
            violations.append(float(n_col))

        # --- Communication connectivity at each waypoint index ---
        # All drones have the same number of path points
        n_points = self.num_waypoints_per_drone + 2  # start + waypoints + target
        for step in range(n_points):
            positions = np.array(
                [paths[d][step] for d in range(len(self.drones))]
            )
            _, n_disc = check_communication(positions, self.comm_range)
            violations.append(float(n_disc))

        return violations


class DroneSwarmMOP3D:
    """3-D multi-objective drone swarm path-planning problem.

    Identical formulation to :class:`DroneSwarmMOP` but uses 3-D waypoints
    ``(x, y, z)`` and spherical obstacles.

    Decision variables
    ------------------
    A flat array of ``(x, y, z)`` intermediate waypoints for every drone.
    For *D* drones each with *W* intermediate waypoints the total number
    of decision variables is ``D * W * 3``.

    Objectives (to minimise)
    ------------------------
    1. **Total path length** – sum of 3-D Euclidean segment lengths.
    2. **Total flight time** – sum of ``path_length / speed`` per drone.
    3. **Total energy consumption** – computed via :class:`EnergyModel`.
    4. **Safety penalty** – proximity penalty to spherical obstacles.

    Parameters
    ----------
    environment : Environment3D
        A 3-D environment with ``width``, ``height``, ``depth``, and
        ``obstacles`` (list of ``(cx, cy, cz, radius)``).
    drones : list of dict
        Each dict must contain ``start`` (3-tuple), ``target`` (3-tuple),
        ``mass``, ``payload``, ``speed``, ``battery_capacity``.
    task_positions : list of tuple
        Informational task positions.
    config : dict, optional
        Overrides for ``num_waypoints_per_drone``, ``safety_distance``,
        ``comm_range``, ``drone_radius``, and ``energy_model`` sub-dict.
    """

    def __init__(self, environment, drones, task_positions, config=None):
        self.environment = environment
        self.drones = drones
        self.task_positions = task_positions
        self.config = config or {}

        self.num_waypoints_per_drone = self.config.get("num_waypoints_per_drone", 5)
        self.safety_distance = self.config.get("safety_distance", 2.0)
        self.comm_range = self.config.get("comm_range", 50.0)
        self.drone_radius = self.config.get("drone_radius", 0.5)

        energy_kwargs = self.config.get("energy_model", {})
        self.energy_model = EnergyModel(**energy_kwargs)

    # ------------------------------------------------------------------
    # Problem dimensions
    # ------------------------------------------------------------------

    @property
    def n_var(self):
        """Total number of decision variables (D * W * 3)."""
        return len(self.drones) * self.num_waypoints_per_drone * 3

    @property
    def n_obj(self):
        """Number of objectives."""
        return 4

    @property
    def bounds(self):
        """Lower and upper bounds for every decision variable.

        Returns
        -------
        lower : ndarray, shape (n_var,)
        upper : ndarray, shape (n_var,)
        """
        lower = np.zeros(self.n_var)
        upper = np.zeros(self.n_var)
        dims = [self.environment.width, self.environment.height,
                self.environment.depth]
        for i in range(self.n_var):
            dim_idx = i % 3
            lower[i] = 0.0
            upper[i] = dims[dim_idx]
        return lower, upper

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_solution(self, x):
        """Convert a flat decision-variable array into per-drone 3-D paths.

        Parameters
        ----------
        x : array-like, shape (n_var,)

        Returns
        -------
        list of ndarray
            One ``(num_waypoints_per_drone + 2, 3)`` array per drone.
        """
        x = np.asarray(x, dtype=np.float64)
        n_wp = self.num_waypoints_per_drone
        paths = []

        for d_idx, drone in enumerate(self.drones):
            offset = d_idx * n_wp * 3
            raw = x[offset: offset + n_wp * 3].reshape(n_wp, 3)

            start = np.array(drone["start"], dtype=np.float64).reshape(1, 3)
            target = np.array(drone["target"], dtype=np.float64).reshape(1, 3)

            full_path = np.vstack([start, raw, target])
            paths.append(full_path)

        return paths

    # ------------------------------------------------------------------
    # Objectives
    # ------------------------------------------------------------------

    def evaluate(self, x):
        """Evaluate objectives for a candidate solution.

        Returns
        -------
        tuple of float
            ``(total_path_length, total_flight_time, total_energy, safety_penalty)``
        """
        paths = self.decode_solution(x)
        obstacles = self.environment.obstacles

        total_length = 0.0
        total_time = 0.0
        total_energy = 0.0
        total_safety = 0.0

        for d_idx, (path, drone) in enumerate(zip(paths, self.drones)):
            segments = np.diff(path, axis=0)
            seg_lengths = np.linalg.norm(segments, axis=1)
            path_length = float(np.sum(seg_lengths))

            total_length += path_length
            total_time += path_length / max(drone["speed"], 1e-9)

            drone_params = {
                "mass": drone["mass"],
                "payload": drone["payload"],
                "speed": drone["speed"],
            }
            total_energy += self.energy_model.compute_path_energy(path, drone_params)

            total_safety += compute_safety_penalty_3d(
                path, obstacles, self.safety_distance
            )

        return total_length, total_time, total_energy, total_safety

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def evaluate_constraints(self, x):
        """Evaluate constraint violations.

        Returns
        -------
        list of float
            Constraint violations (≥ 0; 0 = satisfied).
        """
        paths = self.decode_solution(x)
        obstacles = self.environment.obstacles
        violations = []

        # Energy feasibility per drone
        for path, drone in zip(paths, self.drones):
            drone_params = {
                "mass": drone["mass"],
                "payload": drone["payload"],
                "speed": drone["speed"],
            }
            energy = self.energy_model.compute_path_energy(path, drone_params)
            violations.append(
                check_energy_feasibility(energy, drone["battery_capacity"])
            )

        # Path collision feasibility per drone
        for path in paths:
            _, n_col = check_path_collision_3d(path, obstacles, self.drone_radius)
            violations.append(float(n_col))

        # Communication connectivity at each waypoint index
        n_points = self.num_waypoints_per_drone + 2
        for step in range(n_points):
            positions = np.array(
                [paths[d][step] for d in range(len(self.drones))]
            )
            _, n_disc = check_communication(positions, self.comm_range)
            violations.append(float(n_disc))

        return violations
