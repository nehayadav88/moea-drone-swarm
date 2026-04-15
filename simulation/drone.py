"""Heterogeneous drone model for multi-objective swarm path planning."""

from __future__ import annotations

import numpy as np


class Drone:
    """A single drone with configurable physical and operational parameters.

    Attributes:
        drone_id: Unique integer identifier.
        start_position: 2-D start coordinates (x, y).
        target_position: 2-D goal coordinates (x, y).
        battery_capacity: Battery energy in Wh.
        max_speed: Maximum flight speed in m/s.
        sensing_range: Sensor reach in metres.
        comm_range: Communication reach in metres.
        mass: Airframe mass in kg.
        payload: Payload mass in kg.
        drone_radius: Collision radius in metres.
    """

    def __init__(
        self,
        drone_id: int,
        start_position: np.ndarray | None = None,
        target_position: np.ndarray | None = None,
        battery_capacity: float = 100.0,
        max_speed: float = 15.0,
        sensing_range: float = 20.0,
        comm_range: float = 50.0,
        mass: float = 2.0,
        payload: float = 0.0,
        drone_radius: float = 0.5,
    ) -> None:
        self.drone_id = drone_id
        self.start_position = (
            np.asarray(start_position, dtype=np.float64)
            if start_position is not None
            else np.zeros(2, dtype=np.float64)
        )
        self.target_position = (
            np.asarray(target_position, dtype=np.float64)
            if target_position is not None
            else np.zeros(2, dtype=np.float64)
        )
        self.battery_capacity = float(battery_capacity)
        self.max_speed = float(max_speed)
        self.sensing_range = float(sensing_range)
        self.comm_range = float(comm_range)
        self.mass = float(mass)
        self.payload = float(payload)
        self.drone_radius = float(drone_radius)

    def get_params(self) -> dict:
        """Return a dictionary of all drone parameters."""
        return {
            "drone_id": self.drone_id,
            "start_position": self.start_position.copy(),
            "target_position": self.target_position.copy(),
            "battery_capacity": self.battery_capacity,
            "max_speed": self.max_speed,
            "sensing_range": self.sensing_range,
            "comm_range": self.comm_range,
            "mass": self.mass,
            "payload": self.payload,
            "drone_radius": self.drone_radius,
        }

    @classmethod
    def create_heterogeneous_swarm(
        cls,
        n_drones: int,
        env_width: float,
        env_height: float,
        seed: int | None = None,
    ) -> list[Drone]:
        """Create *n_drones* with randomised, heterogeneous parameters.

        Parameters are sampled uniformly from the following ranges:
            battery_capacity : [80, 150] Wh
            max_speed        : [10, 25]  m/s
            sensing_range    : [15, 30]  m
            comm_range       : [40, 80]  m
            mass             : [1.5, 3.5] kg
            payload          : [0, 1.0]  kg

        Start and target positions are placed randomly inside the given
        environment bounds.

        Args:
            n_drones: Number of drones to create.
            env_width: Width of the environment.
            env_height: Height of the environment.
            seed: Optional RNG seed for reproducibility.

        Returns:
            List of Drone instances.
        """
        rng = np.random.default_rng(seed)
        drones: list[Drone] = []
        for i in range(n_drones):
            start = rng.uniform([0.0, 0.0], [env_width, env_height]).astype(
                np.float64
            )
            target = rng.uniform([0.0, 0.0], [env_width, env_height]).astype(
                np.float64
            )
            drones.append(
                cls(
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

    def __repr__(self) -> str:
        return (
            f"Drone(id={self.drone_id}, "
            f"start={self.start_position}, "
            f"target={self.target_position}, "
            f"battery={self.battery_capacity:.1f}Wh, "
            f"speed={self.max_speed:.1f}m/s, "
            f"mass={self.mass:.2f}kg)"
        )
