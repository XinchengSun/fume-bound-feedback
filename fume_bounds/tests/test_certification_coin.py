from fractions import Fraction as Q
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "certification"))
from scalar_gold import coin_bounds, project_coin, run


class CoinOracleTest(unittest.TestCase):
    def test_locked_grid(self):
        r = run()
        self.assertEqual(r["endpoint_cases"], 222)
        self.assertEqual(len(r["exact_coverage_grid"]), 192)

    def test_invalid_counts(self):
        for n, k in [(0, 1), (-1, 0), (3, 4), (3, -1)]:
            with self.subTest(n=n, k=k), self.assertRaises(ValueError):
                coin_bounds(Q(1, 2), n, k, Q(1, 20))

    def test_signed_and_zero_directions(self):
        q, delta = Q(1, 3), Q(1, 20)
        low, high = coin_bounds(q, 7, 2, delta)
        self.assertEqual(project_coin(q, 7, 2, delta, -1), (1-high, 1-low))
        self.assertEqual(coin_bounds(1-q, 7, 5, delta), (1-high, 1-low))
        self.assertEqual(project_coin(q, 7, 2, delta, 0), (Q(1, 2), Q(1, 2)))


if __name__ == "__main__":
    unittest.main()
