import unittest

from core.energy_model import EnergyModel
from core.problem_definition import ProblemConfig, SwarmPathPlanningProblem
from simulation.drone import Drone
from simulation.environment import CircleObstacle, Environment2D


class CoreAndSimulationTests(unittest.TestCase):
    def test_energy_increases_with_payload(self):
        model = EnergyModel()
        path = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        e1 = model.path_energy(path, speed=10.0, drone_mass=2.0, payload_mass=0.2)
        e2 = model.path_energy(path, speed=10.0, drone_mass=2.0, payload_mass=2.0)
        self.assertGreater(e2, e1)

    def test_environment_collision_and_connectivity(self):
        env = Environment2D(0, 0, 100, 100, static_circles=[CircleObstacle((50, 50), 5)])
        self.assertTrue(env.segment_collides((45, 50), (55, 50)))
        adj = env.communication_adjacency([(0, 0), (5, 0), (20, 0)], [10, 10, 10])
        self.assertFalse(env.is_graph_connected(adj))

    def test_problem_evaluate_outputs_objectives(self):
        drones = [
            Drone(0, (5, 5), (90, 90), battery_capacity=1_000_000, speed=10, sensing_range=10, payload=1.0),
            Drone(1, (10, 10), (85, 85), battery_capacity=1_000_000, speed=12, sensing_range=12, payload=0.8),
        ]
        env = Environment2D(0, 0, 100, 100)
        problem = SwarmPathPlanningProblem(drones, env, EnergyModel(), ProblemConfig(waypoints_per_drone=2))
        x = [0.2, 0.2, 0.4, 0.4, 0.3, 0.3, 0.6, 0.6]
        result = problem.evaluate(x)
        self.assertEqual(len(result.objectives), 4)
        self.assertGreaterEqual(result.objectives[0], 0.0)


if __name__ == "__main__":
    unittest.main()
