"""
Unit tests for SMTT baseline algorithms
"""

import unittest
import numpy as np

from src.baselines import (
    algo_spt,
    algo_edd,
    algo_mdd,
    algo_panneerselvam,
    algo_psk,
    algo_mddc,
    algo_augmented_mddc,
    algo_amddc_apd,
    compute_total_tardiness,
)


class TestBaselines(unittest.TestCase):

    def setUp(self):
        # Known small instance
        self.p = np.array([10, 15, 20, 5, 8], dtype=int)
        self.d = np.array([12, 25, 30, 8, 18], dtype=int)
        self.n = len(self.p)

    def test_permutation_validity(self):
        algos = [
            algo_spt,
            algo_edd,
            algo_mdd,
            algo_panneerselvam,
            algo_psk,
            algo_mddc,
            algo_augmented_mddc,
            algo_amddc_apd,
        ]
        for algo in algos:
            seq = algo(self.p, self.d)
            self.assertEqual(len(seq), self.n, f"{algo.__name__} output length mismatch")
            self.assertEqual(set(seq), set(range(self.n)), f"{algo.__name__} is not a valid permutation")

    def test_compute_tardiness(self):
        seq = np.array([0, 1, 2, 3, 4])
        # c = [10, 25, 45, 50, 58]
        # d = [12, 25, 30, 8, 18]
        # tard = [0, 0, 15, 42, 40] = 97
        tard = compute_total_tardiness(self.p, self.d, seq)
        self.assertEqual(tard, 97)

    def test_amddc_apd_monotonicity(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            p = rng.integers(1, 100, size=15)
            d = rng.integers(10, 400, size=15)
            seq_base = algo_augmented_mddc(p, d)
            tard_base = compute_total_tardiness(p, d, seq_base)
            seq_apd = algo_amddc_apd(p, d)
            tard_apd = compute_total_tardiness(p, d, seq_apd)
            self.assertLessEqual(tard_apd, tard_base, "APD resulted in higher tardiness than starting schedule!")


if __name__ == "__main__":
    unittest.main()
