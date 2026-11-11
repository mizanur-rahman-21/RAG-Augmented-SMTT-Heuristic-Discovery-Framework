"""
Multi-Criteria Candidate Selector (v2)
======================================
Implements rigorous multi-attribute candidate selection for SMTT heuristic discovery.

Selection Attributes:
1. Validation Tardiness Quality (Lower tardiness = higher performance score).
2. Structural & Behavioral Novelty (Kendall tau divergence from all published baselines).
3. Computational Efficiency (Execution latency penalty for high-order polynomial complexity).
4. Population Diversity (Dissimilarity to previously selected best candidates).
5. Pareto Frontier Identification (Multi-objective non-dominated sorting).

Prevents myopic selection of disguised baseline clones that marginally reduce tardiness
at the expense of architectural novelty and runtime efficiency.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats


class MultiCriteriaCandidateSelector:
    """
    Selects optimal discovery candidates using multi-criteria utility and Pareto dominance.
    """

    def __init__(
        self,
        weight_performance: float = 0.50,
        weight_novelty: float = 0.30,
        weight_efficiency: float = 0.10,
        weight_diversity: float = 0.10,
        max_acceptable_runtime_ms: float = 5.0,
    ):
        self.w_perf = weight_performance
        self.w_nov = weight_novelty
        self.w_eff = weight_efficiency
        self.w_div = weight_diversity
        self.max_runtime_ms = max_acceptable_runtime_ms

    def score_candidate(
        self,
        candidate: Dict[str, Any],
        ref_baseline_tardiness: float,
        selected_population: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """
        Computes normalized multi-attribute scores for a single candidate.
        All scores are in [0.0, 1.0+] where higher is better.
        """
        # 1. Validity gate
        if not candidate.get("is_valid", False) or candidate.get("val_score", float("inf")) >= 1e8:
            return {
                "performance_score": 0.0,
                "novelty_score": 0.0,
                "efficiency_score": 0.0,
                "diversity_score": 0.0,
                "composite_utility": 0.0,
                "is_eligible": 0.0,
            }

        val_score = float(candidate["val_score"])
        ref_tard = max(1.0, float(ref_baseline_tardiness))

        # Performance score: ratio of reference to candidate (1.0 = matches baseline, >1.0 = beats baseline)
        perf_score = ref_tard / max(1.0, val_score)

        # 2. Novelty score based on Kendall tau distance from baselines
        # novelty_classification: "CLONE", "KNOWN_LOCAL_SEARCH_COMPOSITION", "GENUINE_NEW_STRUCTURE"
        cls_type = candidate.get("novelty_classification", "UNKNOWN")
        mean_tau = candidate.get("mean_kendall_tau", 1.0)
        exact_match = candidate.get("exact_match_pct", 100.0)

        # Base distance from closest baseline
        base_dist = max(0.0, 1.0 - mean_tau)

        if cls_type == "CLONE" or exact_match >= 95.0:
            nov_score = base_dist * 0.05  # Severe penalty for clones
        elif cls_type == "KNOWN_LOCAL_SEARCH_COMPOSITION":
            nov_score = base_dist * 0.30  # Heavy penalty for repackaging known search
        elif cls_type == "GENUINE_NEW_STRUCTURE":
            nov_score = 0.40 + 0.60 * min(1.0, base_dist)
        else:
            nov_score = base_dist * 0.20

        # 3. Efficiency score
        runtime_ms = candidate.get("inference_time_ms", 0.0) # evaluation time or per-instance time
        # Per-instance execution speed if available
        inst_runtime = candidate.get("eval_latency_ms", 0.05)
        eff_score = 1.0 / (1.0 + (inst_runtime / max(0.01, self.max_runtime_ms)))

        # 4. Diversity score relative to selected population
        div_score = 1.0
        if selected_population and "fingerprint" in candidate and candidate["fingerprint"]:
            cand_fp = candidate["fingerprint"]
            div_list = []
            for other in selected_population:
                if "fingerprint" in other and other["fingerprint"]:
                    tau, _ = stats.kendalltau(cand_fp, other["fingerprint"])
                    d = 1.0 - (1.0 if np.isnan(tau) else tau)
                    div_list.append(d)
            if div_list:
                div_score = float(np.mean(div_list))

        # Composite multi-criteria utility
        composite = (
            self.w_perf * perf_score
            + self.w_nov * nov_score
            + self.w_eff * eff_score
            + self.w_div * div_score
        )

        return {
            "performance_score": float(perf_score),
            "novelty_score": float(nov_score),
            "efficiency_score": float(eff_score),
            "diversity_score": float(div_score),
            "composite_utility": float(composite),
            "is_eligible": 1.0 if (cls_type != "CLONE" and perf_score > 0.5) else 0.0,
        }

    def compute_pareto_frontier(
        self,
        candidates: List[Dict[str, Any]],
        ref_baseline_tardiness: float,
    ) -> List[Dict[str, Any]]:
        """
        Computes the non-dominated Pareto frontier across two primary objectives:
        Objective 1: Min Validation Tardiness (or Max Performance Score)
        Objective 2: Max Novelty Score
        """
        valid_cands = []
        for c in candidates:
            if c.get("is_valid", False) and c.get("val_score", float("inf")) < 1e8:
                scores = self.score_candidate(c, ref_baseline_tardiness)
                c_copy = dict(c)
                c_copy.update(scores)
                valid_cands.append(c_copy)

        if not valid_cands:
            return []

        # Find Pareto-optimal candidates:
        # Candidate A dominates B if A.perf >= B.perf and A.nov >= B.nov (with at least one strict)
        pareto_front = []
        for i, c_a in enumerate(valid_cands):
            dominated = False
            for j, c_b in enumerate(valid_cands):
                if i == j:
                    continue
                # c_b dominates c_a if:
                if (
                    c_b["performance_score"] >= c_a["performance_score"]
                    and c_b["novelty_score"] >= c_a["novelty_score"]
                    and (
                        c_b["performance_score"] > c_a["performance_score"]
                        or c_b["novelty_score"] > c_a["novelty_score"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(c_a)

        # Sort Pareto front by composite utility
        pareto_front.sort(key=lambda x: x["composite_utility"], reverse=True)
        return pareto_front

    def select_best_candidate(
        self,
        candidates: List[Dict[str, Any]],
        ref_baseline_tardiness: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Selects the single best candidate from a discovery run.
        Prioritizes eligible non-clone candidates on the Pareto frontier with highest utility.
        """
        pareto = self.compute_pareto_frontier(candidates, ref_baseline_tardiness)
        if not pareto:
            return None

        # Filter for eligible (non-clone) Pareto candidates
        eligible = [c for c in pareto if c.get("is_eligible", 0.0) > 0.5]
        if eligible:
            return eligible[0]

        # If all candidates are clones or marginal, pick highest composite utility from Pareto
        return pareto[0]
