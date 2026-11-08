import unittest
import numpy as np
from audit_algorithms import (
    augmented_mddc_reference_sequential,
    augmented_mddc_reference_best,
    algo_current_augmented_mddc,
    apply_apd,
    compute_tardiness,
    validate_permutation,
)


class TestAuditAlgorithms(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(999)

    def test_permutations_and_monotonicity(self):
        for _ in range(30):
            p = self.rng.integers(1, 101, size=20)
            P = int(np.sum(p))
            d = self.rng.integers(int(P * 0.2), int(P * 0.8) + 1, size=20)

            s_curr = algo_current_augmented_mddc(p, d)
            s_ref_best = augmented_mddc_reference_best(p, d)
            s_ref_seq = augmented_mddc_reference_sequential(p, d)

            self.assertTrue(validate_permutation(s_curr, 20))
            self.assertTrue(validate_permutation(s_ref_best, 20))
            self.assertTrue(validate_permutation(s_ref_seq, 20))

            # APD on each
            s_curr_apd = apply_apd(p, d, s_curr)
            s_ref_best_apd = apply_apd(p, d, s_ref_best)
            s_ref_seq_apd = apply_apd(p, d, s_ref_seq)

            self.assertTrue(validate_permutation(s_curr_apd, 20))
            self.assertTrue(validate_permutation(s_ref_best_apd, 20))
            self.assertTrue(validate_permutation(s_ref_seq_apd, 20))

            # Monotonicity
            self.assertLessEqual(compute_tardiness(p, d, s_curr_apd), compute_tardiness(p, d, s_curr))
            self.assertLessEqual(compute_tardiness(p, d, s_ref_best_apd), compute_tardiness(p, d, s_ref_best))
            self.assertLessEqual(compute_tardiness(p, d, s_ref_seq_apd), compute_tardiness(p, d, s_ref_seq))


if __name__ == "__main__":
    unittest.main()
