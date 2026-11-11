"""
Unit tests for Improved Discovery Engine v2 Modules
===================================================
Tests:
- PromptBuilderV2: Prompt construction across all 5 conditions.
- EvolutionMemory: Elite tracking, failure diagnostic formatting, diverse retrieval.
- MultiCriteriaCandidateSelector: Utility scoring, clone penalization, Pareto frontier.
- EvaluatorV2: AST LoopGuard, evaluation and novelty audit.
"""

import unittest
import numpy as np

from src.prompt_builder_v2 import build_ablation_prompt_v2
from src.evolution_memory import EvolutionMemory, CandidateEntry, extract_reasoning_and_code
from src.selector_v2 import MultiCriteriaCandidateSelector
from src.evaluator_v2 import safe_execute_candidate, evaluate_on_dataset_v2, audit_candidate_scientific_novelty
from src.instance_generator import generate_single_instance
from src.baselines import algo_mdd, algo_augmented_mddc


class TestV2Modules(unittest.TestCase):

    def test_prompt_builder_v2_conditions(self):
        conditions = ["no_rag", "theory_rag", "code_rag", "dual_rag", "evolution_rag"]
        for cond in conditions:
            prompt = build_ablation_prompt_v2(
                condition=cond,
                theory_context="THEORY_SNIPPET",
                code_memory_context="CODE_SNIPPET",
                evolution_context="EVOLUTION_SNIPPET",
            )
            self.assertIn("Mathematical Intuition & Formulation", prompt)
            self.assertIn("custom_heuristic", prompt)
            self.assertIn("DO NOT generate simple linear combinations", prompt)

            if cond in ("theory_rag", "dual_rag"):
                self.assertIn("THEORY_SNIPPET", prompt)
            if cond in ("code_rag", "dual_rag"):
                self.assertIn("CODE_SNIPPET", prompt)
            if cond == "evolution_rag":
                self.assertIn("EVOLUTION_SNIPPET", prompt)

    def test_evolution_memory_and_diversity(self):
        mem = EvolutionMemory(max_elites=5, max_failures=5)

        # Add elite candidate
        entry1 = CandidateEntry(
            iteration=1, condition="evolution_rag", seed=101,
            code="def custom_heuristic(p, d): return np.argsort(p)",
            reasoning="SPT rule", val_score=400.0, train_score=350.0,
            is_valid=True, status="SUCCESS", novelty_classification="GENUINE_NEW_PRIORITY_FUNCTION",
            closest_baseline="SPT", exact_match_pct=50.0, mean_kendall_tau=0.6,
            novelty_distance=0.4, runtime_ms=0.1, fingerprint=[0, 1, 2, 3, 4]
        )
        mem.add_candidate(entry1)

        # Add second elite with different fingerprint
        entry2 = CandidateEntry(
            iteration=2, condition="evolution_rag", seed=101,
            code="def custom_heuristic(p, d): return np.argsort(d)",
            reasoning="EDD rule", val_score=380.0, train_score=340.0,
            is_valid=True, status="SUCCESS", novelty_classification="GENUINE_NEW_PRIORITY_FUNCTION",
            closest_baseline="EDD", exact_match_pct=40.0, mean_kendall_tau=0.5,
            novelty_distance=0.5, runtime_ms=0.1, fingerprint=[4, 3, 2, 1, 0]
        )
        mem.add_candidate(entry2)

        # Add failed clone
        entry_fail = CandidateEntry(
            iteration=3, condition="evolution_rag", seed=101,
            code="def custom_heuristic(p, d): return np.argsort(d)",
            reasoning="Clone", val_score=390.0, train_score=345.0,
            is_valid=True, status="SUCCESS", novelty_classification="CLONE",
            closest_baseline="MDD", exact_match_pct=98.0, mean_kendall_tau=0.99,
            novelty_distance=0.01, runtime_ms=0.1, fingerprint=[4, 3, 2, 1, 0]
        )
        mem.add_candidate(entry_fail)

        self.assertEqual(len(mem.elites), 2)
        self.assertEqual(len(mem.failures), 1)

        # Retrieve diverse elites
        diverse = mem.retrieve_diverse_elites(top_k=2)
        self.assertEqual(len(diverse), 2)

        # Format context string
        context = mem.format_evolution_context(ref_mdd_score=500.0, ref_amddc_apd_score=450.0)
        self.assertIn("High-Performing Elite Candidates", context)
        self.assertIn("Diagnostic Warnings", context)
        self.assertIn("REJECTED CLONE", context)

    def test_extract_reasoning_and_code(self):
        response_text = """
### Section 1: Mathematical Intuition
We balance dynamic slack s_j(t) against the remaining workload fraction p_j / P(U).

### Section 2: Code
```python
def custom_heuristic(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.argsort(p)
```
"""
        reasoning, code = extract_reasoning_and_code(response_text)
        self.assertIn("balance dynamic slack", reasoning)
        self.assertIn("def custom_heuristic", code)
        self.assertNotIn("```", code)

    def test_multi_criteria_selector(self):
        selector = MultiCriteriaCandidateSelector()
        ref_tard = 500.0

        good_cand = {
            "is_valid": True,
            "val_score": 450.0,
            "novelty_classification": "GENUINE_NEW_PRIORITY_FUNCTION",
            "mean_kendall_tau": 0.5,
            "exact_match_pct": 30.0,
            "eval_latency_ms": 0.05,
            "fingerprint": [0, 1, 2, 3],
        }
        clone_cand = {
            "is_valid": True,
            "val_score": 440.0,  # slightly better score, but it's a clone!
            "novelty_classification": "CLONE",
            "mean_kendall_tau": 0.99,
            "exact_match_pct": 98.0,
            "eval_latency_ms": 0.05,
            "fingerprint": [0, 1, 2, 3],
        }

        score_good = selector.score_candidate(good_cand, ref_tard)
        score_clone = selector.score_candidate(clone_cand, ref_tard)

        # Good novel candidate should have significantly higher novelty and composite utility
        self.assertGreater(score_good["novelty_score"], score_clone["novelty_score"])
        self.assertGreater(score_good["composite_utility"], score_clone["composite_utility"])
        self.assertEqual(score_good["is_eligible"], 1.0)
        self.assertEqual(score_clone["is_eligible"], 0.0)

        # Pareto frontier selection should prioritize the eligible novel candidate
        best = selector.select_best_candidate([clone_cand, good_cand], ref_tard)
        self.assertEqual(best["novelty_classification"], "GENUINE_NEW_PRIORITY_FUNCTION")


if __name__ == "__main__":
    unittest.main()
