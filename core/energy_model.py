"""Realistic, modular energy consumption model for heterogeneous drones.

Models energy expenditure from distance traveled, hovering, turning,
payload transport, and aerodynamic drag.  All parameters are configurable
to support heterogeneous drone fleets.
"""

import numpy as np


class EnergyModel:
    """Configurable energy model for multi-rotor drones.

    Energy components
    -----------------
    - **Distance**: ``E_dist = k_dist * mass * distance * speed_factor``
    - **Hover**: ``E_hover = k_hover * mass * hover_time``
    - **Turning**: ``E_turn = k_turn * sum(|angle_changes|)``
    - **Payload**: ``E_payload = k_payload * payload_weight * distance``
    - **Drag**: ``E_drag = k_drag * speed^2 * distance``

    Total energy is the sum of all components.

    Parameters
    ----------
    k_dist : float
        Distance energy coefficient (default 0.5 J/(kg·m)).
    k_hover : float
        Hover energy coefficient (default 15.0 J/(kg·s)).
    k_turn : float
        Turning energy coefficient (default 2.0 J/rad).
    k_payload : float
        Payload energy coefficient (default 0.3 J/(kg·m)).
    k_drag : float
        Aerodynamic drag coefficient (default 0.02 J·s²/(m³)).
    speed_factor_base : float
        Baseline speed factor (default 1.0).  The effective speed factor
        increases linearly with speed: ``speed_factor_base + 0.1 * speed``.
    """

    def __init__(self, **kwargs):
        self.k_dist = kwargs.get("k_dist", 0.5)
        self.k_hover = kwargs.get("k_hover", 15.0)
        self.k_turn = kwargs.get("k_turn", 2.0)
        self.k_payload = kwargs.get("k_payload", 0.3)
        self.k_drag = kwargs.get("k_drag", 0.02)
        self.speed_factor_base = kwargs.get("speed_factor_base", 1.0)

    # ------------------------------------------------------------------
    # Individual energy components
    # ------------------------------------------------------------------

    def energy_distance(self, mass, distance, speed):
        """Energy from distance traveled.

        Parameters
        ----------
        mass : float
            Drone mass in kg (excluding payload).
        distance : float
            Total distance traveled in metres.
        speed : float
            Cruise speed in m/s.

        Returns
        -------
        float
            Energy in joules.
        """
        speed_factor = self.speed_factor_base + 0.1 * speed
        return self.k_dist * mass * distance * speed_factor

    def energy_hover(self, mass, hover_time):
        """Energy from hovering.

        Parameters
        ----------
        mass : float
            Drone mass in kg.
        hover_time : float
            Total hover duration in seconds.

        Returns
        -------
        float
            Energy in joules.
        """
        return self.k_hover * mass * hover_time

    def energy_turning(self, angle_changes):
        """Energy from turning manoeuvres.

        Parameters
        ----------
        angle_changes : array-like
            Absolute heading changes at each waypoint in radians.

        Returns
        -------
        float
            Energy in joules.
        """
        angle_changes = np.asarray(angle_changes, dtype=np.float64)
        return self.k_turn * np.sum(np.abs(angle_changes))

    def energy_payload(self, payload_weight, distance):
        """Extra energy from carrying a payload.

        Parameters
        ----------
        payload_weight : float
            Payload mass in kg.
        distance : float
            Distance traveled while carrying the payload in metres.

        Returns
        -------
        float
            Energy in joules.
        """
        return self.k_payload * payload_weight * distance

    def energy_drag(self, speed, distance):
        """Energy from speed-dependent aerodynamic drag.

        Parameters
        ----------
        speed : float
            Cruise speed in m/s.
        distance : float
            Distance traveled in metres.

        Returns
        -------
        float
            Energy in joules.
        """
        return self.k_drag * speed ** 2 * distance

    # ------------------------------------------------------------------
    # Path-level computation
    # ------------------------------------------------------------------

    def compute_path_energy(self, waypoints, drone_params):
        """Compute total energy consumption for a drone following a path.

        Parameters
        ----------
        waypoints : array-like, shape (N, 2)
            Ordered 2D waypoints (including start and target).
        drone_params : dict
            Must contain:

            - ``mass`` (float): drone mass in kg.
            - ``payload`` (float): payload mass in kg.
            - ``speed`` (float): cruise speed in m/s.

            Optional:

            - ``hover_time`` (float): total hover time in seconds (default 0).

        Returns
        -------
        float
            Total energy in joules.
        """
        waypoints = np.asarray(waypoints, dtype=np.float64)
        mass = drone_params["mass"]
        payload = drone_params["payload"]
        speed = drone_params["speed"]
        hover_time = drone_params.get("hover_time", 0.0)

        if len(waypoints) < 2:
            return self.energy_hover(mass, hover_time)

        # Segment vectors and distances
        segments = np.diff(waypoints, axis=0)
        seg_lengths = np.linalg.norm(segments, axis=1)
        total_distance = np.sum(seg_lengths)

        # Heading angles and turning costs
        angles = np.arctan2(segments[:, 1], segments[:, 0])
        if len(angles) >= 2:
            raw_diffs = np.diff(angles)
            # Wrap to [-pi, pi]
            angle_changes = (raw_diffs + np.pi) % (2 * np.pi) - np.pi
        else:
            angle_changes = np.array([], dtype=np.float64)

        e_dist = self.energy_distance(mass, total_distance, speed)
        e_hover = self.energy_hover(mass, hover_time)
        e_turn = self.energy_turning(angle_changes)
        e_payload = self.energy_payload(payload, total_distance)
        e_drag = self.energy_drag(speed, total_distance)

        return e_dist + e_hover + e_turn + e_payload + e_drag

    @staticmethod
    def is_feasible(energy, battery_capacity):
        """Check whether the energy requirement is within battery capacity.

        Parameters
        ----------
        energy : float
            Required energy in joules.
        battery_capacity : float
            Battery capacity in joules.

        Returns
        -------
        bool
            True if ``energy <= battery_capacity``.
        """
        return energy <= battery_capacity
