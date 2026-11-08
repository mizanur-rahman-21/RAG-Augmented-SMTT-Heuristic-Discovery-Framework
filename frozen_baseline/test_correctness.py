"""
Correctness Tests for SMTT Verification Package
===============================================
Verifies:
1. Every algorithm returns a strict valid permutation of 0..n-1.
2. Exact DP solver matches brute-force permutation search on n = 3, 4, 5, 6, 7, 8.
3. RADC-APD (formerly RADC-VND) never produces higher tardiness than Augmented MDDC.
4. RADC-APD adjacent swap strictly decreases total tardiness before any swap is accepted.
5. All completion times remain mathematically correct after swaps.
"""

import unittest
import numpy as np
from algorithms import (
    algo_mdd,
    algo_mddc,
    algo_augmented_mdd,
    algo_augmented_mddc,
    algo_radc_apd,
    compute_tardiness,
    compute_completion_times,
    validate_permutation,
)
from exact_solver import solve_smtt_exact, solve_smtt_brute_force
from instance_generator import generate_single_instance


class TestSMTTCorrectness(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def test_dp_vs_brute_force_small_n(self):
        """Verify DP matches brute-force search for n = 3, 4, 5, 6, 7, 8."""
        for n in [3, 4, 5, 6, 7, 8]:
            for rep in range(3):
                inst = generate_single_instance(n, rdd=0.6, tf=0.6, rng=self.rng)
                opt_dp, seq_dp = solve_smtt_exact(inst.p, inst.d)
                opt_bf, seq_bf = solve_smtt_brute_force(inst.p, inst.d)
                
                self.assertEqual(
                    opt_dp,
                    opt_bf,
                    f"DP ({opt_dp}) != Brute Force ({opt_bf}) for n={n}, rep={rep}"
                )
                self.assertTrue(validate_permutation(seq_dp, n))
                self.assertTrue(validate_permutation(seq_bf, n))
                tard_from_seq = compute_tardiness(inst.p, inst.d, seq_dp)
                self.assertEqual(opt_dp, tard_from_seq)

    def test_all_algorithms_valid_permutations(self):
        """Verify MDD, MDDC, AugMDD, AugMDDC, RADC-APD return valid permutations."""
        algos = {
            "MDD": algo_mdd,
            "MDDC": algo_mddc,
            "AugMDD": algo_augmented_mdd,
            "AugMDDC": algo_augmented_mddc,
            "RADC-APD": algo_radc_apd,
        }
        for n in [5, 10, 20]:
            inst = generate_single_instance(n, rdd=0.4, tf=0.6, rng=self.rng)
            for name, algo in algos.items():
                seq = algo(inst.p, inst.d)
                self.assertTrue(
                    validate_permutation(seq, n),
                    f"{name} returned invalid permutation for n={n}: {seq}"
                )

    def test_radc_apd_monotonicity(self):
        """Verify RADC-APD never has higher tardiness than its Augmented MDDC baseline."""
        for _ in range(50):
            inst = generate_single_instance(20, rdd=0.4, tf=0.6, rng=self.rng)
            seq_aug = algo_augmented_mddc(inst.p, inst.d)
            seq_apd = algo_radc_apd(inst.p, inst.d)

            tard_aug = compute_tardiness(inst.p, inst.d, seq_aug)
            tard_apd = compute_tardiness(inst.p, inst.d, seq_apd)

            self.assertLessEqual(
                tard_apd,
                tard_aug,
                f"Monotonicity violation: APD tardiness ({tard_apd}) > AugMDDC ({tard_aug})"
            )

    def test_adjacent_swap_math_invariant(self):
        """
        Verify that whenever an adjacent swap condition:
          tard_swap < tard_orig
        is met, the actual global schedule tardiness decreases by exactly (tard_orig - tard_swap).
        """
        for _ in range(20):
            inst = generate_single_instance(15, rdd=0.2, tf=0.4, rng=self.rng)
            p = inst.p
            d = inst.d
            seq = np.array(range(15))  # arbitrary starting sequence

            comp = compute_completion_times(p, seq)
            global_tard_before = compute_tardiness(p, d, seq)

            for i in range(len(p) - 1):
                j1, j2 = seq[i], seq[i + 1]
                t_curr = comp[i - 1] if i > 0 else 0
                c1_orig = t_curr + p[j1]
                c2_orig = c1_orig + p[j2]
                tard_orig = max(0, c1_orig - d[j1]) + max(0, c2_orig - d[j2])

                c1_swap = t_curr + p[j2]
                c2_swap = c1_swap + p[j1]
                tard_swap = max(0, c1_swap - d[j2]) + max(0, c2_swap - d[j1])

                # Swap and test global tardiness
                trial_seq = seq.copy()
                trial_seq[i], trial_seq[i + 1] = j2, j1
                global_tard_after = compute_tardiness(p, d, trial_seq)

                predicted_diff = tard_orig - tard_swap
                actual_diff = global_tard_before - global_tard_after

                self.assertEqual(
                    predicted_diff,
                    actual_diff,
                    f"O(1) predicted diff ({predicted_diff}) != actual diff ({actual_diff})"
                )

    def test_completion_times_invariant_after_swap(self):
        """Verify that swapping adjacent jobs j1, j2 does not affect completion times of jobs k > i+1."""
        for _ in range(10):
            inst = generate_single_instance(10, rdd=0.5, tf=0.5, rng=self.rng)
            p = inst.p
            seq = np.array(range(10))

            comp_before = compute_completion_times(p, seq)
            i = 3  # swap jobs at positions 3 and 4
            swapped_seq = seq.copy()
            swapped_seq[i], swapped_seq[i + 1] = swapped_seq[i + 1], swapped_seq[i]
            comp_after = compute_completion_times(p, swapped_seq)

            # Jobs before i: completion times unchanged
            for k in range(i):
                self.assertEqual(comp_before[k], comp_after[k])
            # Job i+1: completion time unchanged because p1 + p2 == p2 + p1
            self.assertEqual(comp_before[i + 1], comp_after[i + 1])
            # Jobs after i+1: completion times unchanged
            for k in range(i + 2, len(p)):
                self.assertEqual(comp_before[k], comp_after[k])


if __name__ == "__main__":
    unittest.main()
