"""
Unit tests for Exact SMTT Dynamic Programming Solver
"""

import unittest
import numpy as np

from src.exact_solver import solve_smtt_exact, solve_smtt_brute_force
from src.baselines import compute_total_tardiness


class TestExactSolver(unittest.TestCase):

    def test_dp_matches_brute_force(self):
        rng = np.random.default_rng(12345)
        for n in [3, 4, 5, 6, 7]:
            p = rng.integers(1, 30, size=n)
            d = rng.integers(5, 100, size=n)

            bf_tard, bf_seq = solve_smtt_brute_force(p, d)
            dp_tard, dp_seq = solve_smtt_exact(p, d)

            self.assertEqual(bf_tard, dp_tard, f"Mismatch between DP and BF for n={n}")
            # Verify sequence tardiness matches reported optimal tardiness
            computed_dp_tard = compute_total_tardiness(p, d, dp_seq)
            self.assertEqual(dp_tard, computed_dp_tard)
            computed_bf_tard = compute_total_tardiness(p, d, bf_seq)
            self.assertEqual(bf_tard, computed_bf_tard)

    def test_edge_cases(self):
        # Empty
        p0 = np.array([], dtype=int)
        d0 = np.array([], dtype=int)
        t0, seq0 = solve_smtt_exact(p0, d0)
        self.assertEqual(t0, 0)
        self.assertEqual(len(seq0), 0)

        # Single job
        p1 = np.array([10], dtype=int)
        d1 = np.array([5], dtype=int)
        t1, seq1 = solve_smtt_exact(p1, d1)
        self.assertEqual(t1, 5)
        self.assertEqual(list(seq1), [0])


if __name__ == "__main__":
    unittest.main()
