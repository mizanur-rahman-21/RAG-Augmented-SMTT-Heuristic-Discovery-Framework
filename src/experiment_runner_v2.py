"""
Advanced Experiment Runner v2 with Evolution-RAG & Multi-Criteria Selection
==========================================================================
Executes controlled discovery runs across 5 conditions:
1. no_rag
2. theory_rag
3. code_rag
4. dual_rag
5. evolution_rag (New)

Orchestrates:
- Theory retrieval (theory_kb/)
- Evolution population memory & targeted mutation directives
- Extraction of mathematical reasoning + Python implementation
- Sandbox execution & loop guard protection
- Multi-tier scientific novelty audit
- Multi-criteria candidate scoring & Pareto frontier identification
- Cryptographic provenance logging (SHA-256)
"""

import os
import json
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
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
)
from src.evaluator_v2 import (
    safe_execute_candidate,
    evaluate_on_dataset_v2,
    audit_candidate_scientific_novelty,
)
from src.retrieval import TheoryRetriever, CodeMemoryRetriever
from src.prompt_builder_v2 import build_ablation_prompt_v2
from src.evolution_memory import EvolutionMemory, CandidateEntry, extract_reasoning_and_code
from src.selector_v2 import MultiCriteriaCandidateSelector
from src.llm_client import LLMClient


ALL_AUDIT_BASELINES: Dict[str, Any] = {
    "SPT": algo_spt,
    "EDD": algo_edd,
    "Panneerselvam": algo_panneerselvam,
    "PSK": algo_psk,
    "MDD": algo_mdd,
    "MDDC": algo_mddc,
    "Augmented MDDC": algo_augmented_mddc,
    "AMDDC + APD": algo_amddc_apd,
}


class ExperimentRunnerV2:
    """Manages execution of Improved Discovery Engine v2."""

    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: LLMClient,
        project_root: Optional[Path] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.root = project_root or Path(__file__).resolve().parent.parent

        kb_dir = self.root / "theory_kb"
        self.theory_retriever = TheoryRetriever(str(kb_dir))

        self.results_dir = self.root / "results"
        self.logs_dir = self.root / "logs"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.baselines = ALL_AUDIT_BASELINES
        self.selector = MultiCriteriaCandidateSelector()

    def run_discovery_session_v2(
        self,
        conditions: List[str],
        discovery_seeds: List[int],
        iterations_per_run: int,
        train_instances: list,
        val_instances: list,
        audit_instances: list,
        provenance_log_file: str = "v2_candidates_provenance.jsonl",
        summary_csv_file: str = "v2_discovery_summary.csv",
    ) -> List[Dict[str, Any]]:
        """
        Runs the full discovery loop across conditions and random seeds.
        """
        provenance_log_path = self.logs_dir / provenance_log_file
        summary_csv_path = self.results_dir / summary_csv_file

        # Precompute baseline validation tardiness
        baseline_val_scores = {}
        for b_name, b_fn in self.baselines.items():
            valid, mean_t, _, _ = evaluate_on_dataset_v2(b_fn, val_instances)
            baseline_val_scores[b_name] = mean_t

        ref_mdd_score = baseline_val_scores.get("MDD", 1.0)
        ref_amddc_apd_score = baseline_val_scores.get("AMDDC + APD", 1.0)

        anchor_instance = audit_instances[0] if audit_instances else train_instances[0]

        all_candidate_records = []
        run_summaries = []

        total_experiments = len(conditions) * len(discovery_seeds) * iterations_per_run
        exp_idx = 0

        print("=" * 80)
        print("Starting SMTT Heuristic Discovery Session v2 (Improved Engine)")
        print(f"Conditions: {conditions}")
        print(f"Discovery Seeds: {discovery_seeds}")
        print(f"Iterations per run: {iterations_per_run}")
        print(f"Total Planned LLM Inferences: {total_experiments}")
        print(f"Reference Horizons - MDD: {ref_mdd_score:.2f} | AMDDC + APD: {ref_amddc_apd_score:.2f}")
        print("=" * 80)

        with open(provenance_log_path, "a", encoding="utf-8") as prov_f:
            for cond in conditions:
                for seed in discovery_seeds:
                    code_memory = CodeMemoryRetriever()
                    evolution_memory = EvolutionMemory()
                    run_candidates: List[Dict[str, Any]] = []

                    for it in range(1, iterations_per_run + 1):
                        exp_idx += 1
                        print(f"\n[{exp_idx}/{total_experiments}] Condition: {cond.upper()} | Seed: {seed} | Iter: {it}")

                        # 1. Retrieve Condition-Specific Context
                        theory_context = ""
                        if cond in ("theory_rag", "dual_rag"):
                            docs = self.theory_retriever.retrieve(top_k=2)
                            theory_context = self.theory_retriever.format_for_prompt(docs)

                        code_context = ""
                        if cond in ("code_rag", "dual_rag"):
                            elites = code_memory.retrieve(top_k=2)
                            code_context = code_memory.format_for_prompt(elites)

                        evolution_context = ""
                        if cond == "evolution_rag":
                            evolution_context = evolution_memory.format_evolution_context(
                                ref_mdd_score=ref_mdd_score,
                                ref_amddc_apd_score=ref_amddc_apd_score,
                                top_k_elites=2,
                                top_k_failures=2,
                            )

                        # 2. Build Enhanced v2 Prompt
                        prompt = build_ablation_prompt_v2(
                            condition=cond,
                            theory_context=theory_context,
                            code_memory_context=code_context,
                            evolution_context=evolution_context,
                        )

                        # 3. LLM Inference (Live, with designated random seed & hashing)
                        try:
                            gen_result = self.llm_client.generate(prompt, seed=seed)
                        except Exception as e:
                            print(f"  [ERROR] LLM generation failed: {e}")
                            raise

                        # 4. Extract Mathematical Reasoning & Python Implementation
                        reasoning, code = extract_reasoning_and_code(gen_result.raw_response)

                        # 5. Sandbox Compilation & AST Loop-Guard Transformation
                        is_valid, fn, err_msg = safe_execute_candidate(code)

                        train_score = float("inf")
                        val_score = float("inf")
                        train_time_ms = 0.0
                        eval_status = "SYNTAX_ERROR" if not is_valid else "UNKNOWN"

                        novelty_info = {
                            "classification": "INVALID" if not is_valid else "UNKNOWN",
                            "closest_baseline": "None",
                            "exact_match_pct": 0.0,
                            "mean_kendall_tau": 0.0,
                            "literature_novelty_distance": 0.0,
                            "is_novel": False,
                            "fingerprint": None,
                            "reason": err_msg if not is_valid else "",
                        }

                        if is_valid and fn is not None:
                            # Evaluate on training split
                            timeout_s = float(self.config.get("timeout_per_candidate_s", 2.0))
                            train_success, train_score, train_time_ms, t_msg = evaluate_on_dataset_v2(
                                fn, train_instances, timeout_s=timeout_s
                            )

                            if train_success:
                                # Evaluate on validation split
                                val_success, val_score, val_time_ms, v_msg = evaluate_on_dataset_v2(
                                    fn, val_instances, timeout_s=timeout_s
                                )

                                if val_success:
                                    eval_status = "SUCCESS"
                                    # Novelty audit against all 8 baselines
                                    novelty_info = audit_candidate_scientific_novelty(
                                        code_str=code,
                                        fn=fn,
                                        audit_instances=audit_instances,
                                        baselines=self.baselines,
                                        anchor_instance=anchor_instance,
                                    )

                                    # Update legacy CodeMemory
                                    code_memory.add_candidate(
                                        code=code,
                                        val_score=val_score,
                                        train_score=train_score,
                                        is_novel=novelty_info.get("is_novel", False),
                                        iteration=it,
                                        condition=cond,
                                    )
                                else:
                                    eval_status = "VAL_EVAL_FAILED"
                            else:
                                eval_status = "TRAIN_EVAL_FAILED"

                        # Register candidate in EvolutionMemory
                        entry = CandidateEntry(
                            iteration=it,
                            condition=cond,
                            seed=seed,
                            code=code,
                            reasoning=reasoning,
                            val_score=val_score,
                            train_score=train_score,
                            is_valid=is_valid and eval_status == "SUCCESS",
                            status=eval_status,
                            novelty_classification=novelty_info.get("classification", "INVALID"),
                            closest_baseline=novelty_info.get("closest_baseline", "None"),
                            exact_match_pct=novelty_info.get("exact_match_pct", 0.0),
                            mean_kendall_tau=novelty_info.get("mean_kendall_tau", 0.0),
                            novelty_distance=novelty_info.get("literature_novelty_distance", 0.0),
                            runtime_ms=train_time_ms,
                            fingerprint=novelty_info.get("fingerprint"),
                        )
                        evolution_memory.add_candidate(entry)

                        # Determine performance tier
                        perf_tier = "INVALID"
                        if eval_status == "SUCCESS":
                            if val_score < ref_amddc_apd_score:
                                perf_tier = "BEATS_AMDDC_APD"
                            elif val_score < baseline_val_scores.get("Augmented MDDC", ref_mdd_score):
                                perf_tier = "BEATS_AUGMENTED_MDDC"
                            elif val_score < ref_mdd_score:
                                perf_tier = "BEATS_MDD"
                            else:
                                perf_tier = "SUB_MDD"

                        # Record candidate
                        record = {
                            "condition": cond,
                            "seed": seed,
                            "iteration": it,
                            "is_valid": is_valid,
                            "eval_status": eval_status,
                            "performance_tier": perf_tier,
                            "train_score": train_score,
                            "val_score": val_score,
                            "novelty_classification": novelty_info.get("classification"),
                            "closest_baseline": novelty_info.get("closest_baseline"),
                            "exact_match_pct": novelty_info.get("exact_match_pct"),
                            "mean_kendall_tau": novelty_info.get("mean_kendall_tau"),
                            "literature_novelty_distance": novelty_info.get("literature_novelty_distance"),
                            "is_novel": novelty_info.get("is_novel", False),
                            "novelty_reason": novelty_info.get("reason"),
                            "beats_mdd": val_score < ref_mdd_score if is_valid else False,
                            "beats_amddc_apd": val_score < ref_amddc_apd_score if is_valid else False,
                            "llm_model": gen_result.model,
                            "llm_backend": gen_result.backend,
                            "timestamp_utc": gen_result.timestamp_utc,
                            "prompt_hash": gen_result.prompt_hash,
                            "response_hash": gen_result.response_hash,
                            "generated_tokens": gen_result.generated_tokens,
                            "inference_time_ms": gen_result.inference_time_ms,
                            "fingerprint": novelty_info.get("fingerprint"),
                            "reasoning": reasoning,
                            "code": code,
                        }

                        # Compute multi-criteria scores
                        mc_scores = self.selector.score_candidate(record, ref_mdd_score)
                        record.update(mc_scores)

                        run_candidates.append(record)
                        all_candidate_records.append(record)

                        prov_f.write(json.dumps(record) + "\n")
                        prov_f.flush()

                        print(f"  Status: {eval_status} | Tier: {perf_tier} | "
                              f"Novelty: {novelty_info.get('classification')} | "
                              f"Val Tard: {val_score:.2f} | Utility: {mc_scores['composite_utility']:.3f}")

                    # Select best candidate for this run using multi-criteria selector
                    best_cand = self.selector.select_best_candidate(run_candidates, ref_mdd_score)
                    pareto_front = self.selector.compute_pareto_frontier(run_candidates, ref_mdd_score)

                    run_summaries.append({
                        "condition": cond,
                        "seed": seed,
                        "candidates_generated": len(run_candidates),
                        "pareto_frontier_size": len(pareto_front),
                        "best_candidate_val_tardiness": best_cand.get("val_score") if best_cand else float("inf"),
                        "best_candidate_novelty": best_cand.get("novelty_classification") if best_cand else "None",
                        "best_candidate_utility": best_cand.get("composite_utility") if best_cand else 0.0,
                        "beats_amddc_apd": best_cand.get("beats_amddc_apd", False) if best_cand else False,
                        "best_code": best_cand.get("code") if best_cand else "",
                        "best_reasoning": best_cand.get("reasoning") if best_cand else "",
                    })

        # Write summary CSV
        if all_candidate_records:
            fieldnames = [
                "condition", "seed", "iteration", "eval_status", "performance_tier",
                "train_score", "val_score", "novelty_classification", "closest_baseline",
                "exact_match_pct", "mean_kendall_tau", "literature_novelty_distance",
                "is_novel", "composite_utility", "performance_score", "novelty_score",
                "efficiency_score", "diversity_score", "beats_mdd", "beats_amddc_apd",
                "llm_model", "inference_time_ms", "generated_tokens", "prompt_hash", "response_hash"
            ]
            with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for rec in all_candidate_records:
                    writer.writerow(rec)

        print("\n" + "=" * 80)
        print("Discovery Session v2 Completed Successfully.")
        print(f"Provenance logged to: {provenance_log_path}")
        print(f"Summary CSV written to: {summary_csv_path}")
        print("=" * 80)

        return all_candidate_records
