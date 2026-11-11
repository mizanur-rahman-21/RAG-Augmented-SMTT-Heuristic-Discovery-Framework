"""
Unit tests for AST sandbox scanner and novelty audit
"""

import unittest
import numpy as np

from src.evaluator import scan_code_ast, safe_execute_candidate, audit_candidate_novelty_strengthened
from src.baselines import algo_mdd, algo_augmented_mddc, algo_amddc_apd
from src.instance_generator import generate_single_instance


class TestSandboxAndAudit(unittest.TestCase):

    def test_ast_scanner_rejections(self):
        unsafe_snippets = [
            "import os\ndef custom_heuristic(p, d):\n    return np.argsort(d)",
            "from subprocess import Popen\ndef custom_heuristic(p, d):\n    return np.argsort(d)",
            "def custom_heuristic(p, d):\n    open('secret.txt', 'w')\n    return np.argsort(d)",
            "def custom_heuristic(p, d):\n    eval('2+2')\n    return np.argsort(d)",
            "def custom_heuristic(p, d):\n    exec('x=1')\n    return np.argsort(d)",
            "def custom_heuristic(p, d):\n    try:\n        x = 1\n    except:\n        pass\n    return np.argsort(d)",
        ]
        for snippet in unsafe_snippets:
            is_safe, msg = scan_code_ast(snippet)
            self.assertFalse(is_safe, f"Failed to reject unsafe snippet: {snippet}")

    def test_ast_scanner_allows_clean_code(self):
        clean_snippet = """
import numpy as np
def custom_heuristic(p, d):
    n = len(p)
    return np.argsort(d)
"""
        is_safe, msg = scan_code_ast(clean_snippet)
        self.assertTrue(is_safe, f"Wrongly rejected safe snippet: {msg}")

    def test_safe_execute_success(self):
        code = """
def custom_heuristic(p, d):
    return np.argsort(p)
"""
        is_safe, fn, msg = safe_execute_candidate(code)
        self.assertTrue(is_safe)
        self.assertIsNotNone(fn)

        p = np.array([20, 10, 30])
        d = np.array([50, 50, 50])
        seq = fn(p, d)
        self.assertEqual(list(seq), [1, 0, 2])

    def test_novelty_audit_clone_detection(self):
        # A function that mimics algo_mdd exactly
        clone_code = """
def custom_heuristic(p, d):
    n = len(p)
    U = list(range(n))
    S = []
    t = 0
    while U:
        best_j = -1
        best_val = 10**18
        for j in U:
            v = max(t + p[j], d[j])
            if v < best_val:
                best_val = v
                best_j = j
        U.remove(best_j)
        S.append(best_j)
        t += p[best_j]
    return np.array(S, dtype=int)
"""
        is_safe, fn, _ = safe_execute_candidate(clone_code)
        self.assertTrue(is_safe)

        rng = np.random.default_rng(42)
        audit_instances = [
            generate_single_instance(n_jobs=10, rdd=0.4, tf=0.4, rng=rng)
            for _ in range(5)
        ]
        baselines = {"MDD": algo_mdd, "Augmented MDDC": algo_augmented_mddc}

        result = audit_candidate_novelty_strengthened(
            code_str=clone_code,
            fn=fn,
            audit_instances=audit_instances,
            baselines=baselines,
        )
        self.assertEqual(result["classification"], "CLONE")
        self.assertEqual(result["closest_baseline"], "MDD")
        self.assertFalse(result["is_novel"])


if __name__ == "__main__":
    unittest.main()
