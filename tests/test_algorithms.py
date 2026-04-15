import unittest

from algorithms.nsga2 import NSGA2, NSGA2Config
from core.energy_model import EnergyModel
from core.problem_definition import ProblemConfig, SwarmPathPlanningProblem
from simulation.drone import Drone
from simulation.environment import Environment2D


class AlgorithmSmokeTests(unittest.TestCase):
    def test_nsga2_smoke(self):
        drones = [
            Drone(0, (0, 0), (20, 20), battery_capacity=1_000_000, speed=10, sensing_range=10, payload=1.0),
            Drone(1, (0, 20), (20, 0), battery_capacity=1_000_000, speed=10, sensing_range=10, payload=1.0),
        ]
        env = Environment2D(0, 0, 30, 30)
        problem = SwarmPathPlanningProblem(drones, env, EnergyModel(), ProblemConfig(waypoints_per_drone=2))
        algo = NSGA2(NSGA2Config(population_size=16, generations=3, seed=1, parallel_evaluation=False, workers=1))
        result = algo.run(problem)
        self.assertGreater(len(result.pareto_solutions), 0)
        self.assertGreater(len(result.history), 0)


if __name__ == "__main__":
    unittest.main()
