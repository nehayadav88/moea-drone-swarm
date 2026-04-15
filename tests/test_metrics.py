import unittest

from metrics.statistics import improvement_rate


class MetricsTests(unittest.TestCase):
    def test_improvement_rate_minimize_and_maximize(self):
        self.assertAlmostEqual(improvement_rate(8.0, 10.0), 20.0)
        self.assertAlmostEqual(improvement_rate(12.0, 10.0, maximize=True), 20.0)


if __name__ == "__main__":
    unittest.main()
