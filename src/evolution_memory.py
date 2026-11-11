"""
Evolutionary Candidate Memory & Population Manager (v2)
=======================================================
Implements an advanced evolutionary memory bank for LLM-based heuristic search:
1. Dual Memory Pools:
   - Elite Memory: High-performing, syntactically sound, novel candidates.
   - Failure & Negative Memory: Failed candidates (clones, timeouts, execution errors,
     suboptimal heuristics) with specific diagnosis of why they failed.
2. Behavioral Diversity-Preserving Selection:
   - Prevents population collapse into single-equation variations by selecting
     exemplars that are mutually diverse according to rank correlation.
3. Mutation & Evolution Directives:
   - Synthesizes actionable evolutionary prompts that provide both positive exemplar code
     and negative guidance ("what failed and why").
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class CandidateEntry:
    iteration: int
    condition: str
    seed: int
    code: str
    reasoning: str
    val_score: float
    train_score: float
    is_valid: bool
    status: str  # "SUCCESS", "SYNTAX_ERROR", "TIMEOUT", "CLONE", "HIGH_TARDINESS"
    novelty_classification: str
    closest_baseline: str
    exact_match_pct: float
    mean_kendall_tau: float
    novelty_distance: float  # 1.0 - mean_kendall_tau (higher = more novel)
    runtime_ms: float
    fingerprint: Optional[List[int]] = None  # Permutation on benchmark anchor instance


def extract_reasoning_and_code(raw_response: str) -> Tuple[str, str]:
    """
    Separates the mathematical reasoning section from the Python code block.
    """
    # 1. Extract Python code
    code_pattern = r"```(?:python)?\s*(.*?)\s*```"
    code_matches = re.findall(code_pattern, raw_response, re.DOTALL)
    
    extracted_code = ""
    if code_matches:
        for block in code_matches:
            if "def custom_heuristic" in block:
                extracted_code = block.strip()
                break
        if not extracted_code:
            extracted_code = code_matches[0].strip()
    else:
        # Fallback to function body search
        if "def custom_heuristic" in raw_response:
            lines = raw_response.splitlines()
            fn_lines = []
            capturing = False
            for line in lines:
                if "def custom_heuristic" in line:
                    capturing = True
                if capturing:
                    fn_lines.append(line)
            extracted_code = "\n".join(fn_lines).strip()
        else:
            extracted_code = raw_response.strip()

    # 2. Extract Reasoning (text outside the code block)
    text_without_code = re.sub(code_pattern, "", raw_response, flags=re.DOTALL).strip()
    
    # Clean up markdown headers
    reasoning_lines = []
    for line in text_without_code.splitlines():
        line_clean = line.strip()
        if line_clean and not line_clean.startswith("```"):
            reasoning_lines.append(line)
    
    reasoning = "\n".join(reasoning_lines).strip()
    if not reasoning:
        reasoning = "No explicit theoretical intuition provided."

    return reasoning, extracted_code


class EvolutionMemory:
    """
    Population memory bank managing elites, failures, and diversity-preserving retrieval.
    """

    def __init__(self, max_elites: int = 15, max_failures: int = 15):
        self.max_elites = max_elites
        self.max_failures = max_failures
        self.elites: List[CandidateEntry] = []
        self.failures: List[CandidateEntry] = []
        self.all_history: List[CandidateEntry] = []

    def add_candidate(self, entry: CandidateEntry):
        """Adds a evaluated candidate entry to the appropriate memory tier."""
        self.all_history.append(entry)

        if entry.is_valid and entry.status == "SUCCESS":
            # Add to elite pool if not a clone
            if entry.novelty_classification != "CLONE":
                self.elites.append(entry)
                # Sort elites by validation score (ascending: lower tardiness is better)
                self.elites.sort(key=lambda x: x.val_score)
                if len(self.elites) > self.max_elites:
                    self.elites.pop()  # Keep only top elites
            else:
                # Clones are considered redundant attempts
                entry.status = "CLONE"
                self.failures.append(entry)
                if len(self.failures) > self.max_failures:
                    self.failures.pop(0)
        else:
            self.failures.append(entry)
            if len(self.failures) > self.max_failures:
                self.failures.pop(0)

    def retrieve_diverse_elites(self, top_k: int = 2) -> List[CandidateEntry]:
        """
        Retrieves top_k elites that maximize both performance and behavioral diversity.
        Uses greedy maximal diversity selection.
        """
        if not self.elites:
            return []
        if len(self.elites) <= top_k:
            return list(self.elites)

        # Start with the best-performing elite
        selected = [self.elites[0]]
        candidates = list(self.elites[1:])

        while len(selected) < top_k and candidates:
            # Pick candidate with best combined score: low tardiness + distance from selected
            best_cand = None
            best_utility = -float("inf")

            for cand in candidates:
                # Compute average distance to currently selected elites
                distances = []
                for s in selected:
                    if cand.fingerprint is not None and s.fingerprint is not None:
                        tau, _ = stats.kendalltau(cand.fingerprint, s.fingerprint)
                        dist = 1.0 - (1.0 if np.isnan(tau) else tau)
                    else:
                        dist = 0.5
                    distances.append(dist)
                
                avg_dist = float(np.mean(distances)) if distances else 0.5
                
                # Relative tardiness ratio compared to best
                perf_ratio = selected[0].val_score / max(1.0, cand.val_score)
                # Combined utility: 70% performance, 30% diversity
                utility = 0.7 * perf_ratio + 0.3 * avg_dist

                if utility > best_utility:
                    best_utility = utility
                    best_cand = cand

            if best_cand:
                selected.append(best_cand)
                candidates.remove(best_cand)
            else:
                break

        return selected

    def format_evolution_context(
        self,
        ref_mdd_score: float,
        ref_amddc_apd_score: float,
        top_k_elites: int = 2,
        top_k_failures: int = 2,
    ) -> str:
        """
        Synthesizes an evolutionary prompt section combining:
        1. Performance benchmarks.
        2. Diverse elite exemplars with code & intuition.
        3. Diagnosis of failed/rejected approaches.
        4. Targeted evolutionary operators (crossover / mutation recommendations).
        """
        sections = ["### Evolutionary Population Feedback & Search Guidance:"]

        # 1. Performance Horizon
        sections.append(
            f"**Current Benchmark Tardiness Horizons (Validation Set)**:\n"
            f"- Standard MDD Baseline: `{ref_mdd_score:.2f}`\n"
            f"- State-of-the-Art AMDDC + APD Target: `{ref_amddc_apd_score:.2f}`\n"
            f"Goal: Discover a pure constructive rule achieving lower total tardiness without local search."
        )

        # 2. Elite Exemplars
        elites = self.retrieve_diverse_elites(top_k=top_k_elites)
        if elites:
            sections.append("#### High-Performing Elite Candidates Discovered So Far:")
            for i, elite in enumerate(elites, 1):
                sections.append(
                    f"--- Elite Program #{i} (Val Tardiness: {elite.val_score:.2f} | Novelty Distance: {elite.novelty_distance:.3f}) ---\n"
                    f"Theoretical Intuition: {elite.reasoning[:300]}...\n"
                    f"```python\n{elite.code.strip()}\n```"
                )
        else:
            sections.append("#### Elite Pool: Initial population in progress (no dominant elites discovered yet).")

        # 3. Negative Feedback / Failed Attempts
        failures = [f for f in self.failures if f.status in ("CLONE", "TIMEOUT", "HIGH_TARDINESS")][-top_k_failures:]
        if failures:
            sections.append("#### Diagnostic Warnings from Suboptimal / Rejected Attempts:")
            for fail in failures:
                if fail.status == "CLONE":
                    sections.append(
                        f"- REJECTED CLONE: Candidate produced schedules matching {fail.exact_match_pct:.1f}% "
                        f"with existing baseline {fail.closest_baseline}. Reason: Pure cosmetic variation of known rule."
                    )
                elif fail.status == "TIMEOUT":
                    sections.append(
                        "- REJECTED TIMEOUT: Candidate exceeded execution time limits. Reason: Excessive lookahead "
                        "or unpruned loop complexity. Keep dispatching to O(n^2)."
                    )
                else:
                    sections.append(
                        f"- SUBOPTIMAL EXPLORATION: Attempt with Val Tardiness = {fail.val_score:.2f} "
                        f"(poor performance). Reason: Priority index failed to balance urgent vs non-urgent jobs."
                    )

        # 4. Evolutionary Mutation Directives
        sections.append(
            "#### Suggested Evolutionary Mutation & Crossover Directions:\n"
            "1. **Crossover**: Synthesize complementary ideas: combine an effective slack-ratio index from one elite "
            "with a dynamic bottleneck opportunity penalty.\n"
            "2. **Nonlinear Mutation**: If linear scaling of processing time or slack causes instability, introduce a smooth "
            "transition function (e.g. sigmoid or hyperbolic tangent) governed by remaining workload fraction p_j / P(U).\n"
            "3. **State Interaction**: Explore how the variance of processing times in remaining set U should modify "
            "the urgency threshold."
        )

        return "\n\n".join(sections)
