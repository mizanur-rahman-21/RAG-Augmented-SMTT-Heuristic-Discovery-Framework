"""
Experiment Runner for Controlled SMTT Heuristic Discovery
=========================================================
Executes controlled discovery runs across the 4 conditions:
- no_rag
- theory_rag
- code_rag
- dual_rag

Maintains strict separation:
- Train dataset: for scoring and candidate feedback
- Validation dataset: for selecting the best candidate per run
- Audit dataset: for novelty and equivalence classification
- Test dataset: NEVER exposed during discovery

Outputs:
- JSONL stream of candidate generations with full cryptographic provenance
- Summary statistics across conditions and seeds
- Best discovered candidates saved to disk
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
from src.evaluator import (
    safe_execute_candidate,
    evaluate_on_dataset,
    audit_candidate_novelty_strengthened,
)
from src.retrieval import TheoryRetriever, CodeMemoryRetriever
from src.prompt_builder import build_ablation_prompt
from src.llm_client import LLMClient, LLMGenerationResult


DEFAULT_BASELINES = {
    "SPT": algo_spt,
    "EDD": algo_edd,
    "MDD": algo_mdd,
    "Panneerselvam": algo_panneerselvam,
    "PSK": algo_psk,
    "MDDC": algo_mddc,
    "Augmented MDDC": algo_augmented_mddc,
    "AMDDC + APD": algo_amddc_apd,
}


class ExperimentRunner:
    """Manages execution of controlled SMTT discovery experiments."""

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

        self.baselines = DEFAULT_BASELINES

    def run_discovery_session(
        self,
        conditions: List[str],
        discovery_seeds: List[int],
        iterations_per_run: int,
        train_instances: list,
        val_instances: list,
        audit_instances: list,
        provenance_log_file: str = "candidates_provenance.jsonl",
        summary_csv_file: str = "discovery_summary.csv",
    ) -> List[Dict[str, Any]]:
        """
        Executes discovery loop over conditions and random seeds.
        """
        provenance_log_path = self.logs_dir / provenance_log_file
        summary_csv_path = self.results_dir / summary_csv_file

        # Precompute baseline validation tardiness for comparison
        baseline_val_scores = {}
        for b_name, b_fn in self.baselines.items():
            valid, mean_t, _ = evaluate_on_dataset(b_fn, val_instances)
            baseline_val_scores[b_name] = mean_t

        ref_mdd_score = baseline_val_scores.get("MDD", 1.0)
        ref_amddc_apd_score = baseline_val_scores.get("AMDDC + APD", 1.0)

        all_candidate_records = []
        run_summaries = []

        total_experiments = len(conditions) * len(discovery_seeds) * iterations_per_run
        exp_idx = 0

        print("=" * 80)
        print("Starting SMTT Heuristic Discovery Session")
        print(f"Conditions: {conditions}")
        print(f"Seeds: {discovery_seeds}")
        print(f"Iterations per run: {iterations_per_run}")
        print(f"Total LLM calls planned: {total_experiments}")
        print(f"Reference Val Tardiness - MDD: {ref_mdd_score:.2f}, AMDDC + APD: {ref_amddc_apd_score:.2f}")
        print("=" * 80)

        with open(provenance_log_path, "a", encoding="utf-8") as prov_f:
            for cond in conditions:
                for seed in discovery_seeds:
                    code_memory = CodeMemoryRetriever()
                    best_run_val_score = float("inf")
                    best_run_candidate = None

                    for it in range(1, iterations_per_run + 1):
                        exp_idx += 1
                        print(f"[{exp_idx}/{total_experiments}] Cond: {cond} | Seed: {seed} | Iter: {it}")

                        # 1. Retrieve context
                        theory_context = ""
                        if cond in ("theory_rag", "dual_rag"):
                            docs = self.theory_retriever.retrieve(top_k=2)
                            theory_context = self.theory_retriever.format_for_prompt(docs)

                        code_context = ""
                        if cond in ("code_rag", "dual_rag"):
                            elites = code_memory.retrieve(top_k=2)
                            code_context = code_memory.format_for_prompt(elites)

                        # 2. Build prompt
                        prompt = build_ablation_prompt(
                            condition=cond,
                            theory_context=theory_context,
                            code_memory_context=code_context,
                        )

                        # 3. LLM Inference (Real, with full provenance)
                        t0 = time.perf_counter()
                        try:
                            gen_result = self.llm_client.generate(prompt)
                        except Exception as e:
                            print(f"  [ERROR] LLM generation failed: {e}")
                            raise

                        # 4. Safe sandbox evaluation
                        code = gen_result.extracted_code
                        is_valid, fn, err_msg = safe_execute_candidate(code)

                        train_score = float("inf")
                        val_score = float("inf")
                        train_time_ms = 0.0
                        novelty_info = {
                            "classification": "INVALID" if not is_valid else "UNKNOWN",
                            "closest_baseline": "None",
                            "exact_match_pct": 0.0,
                            "mean_kendall_tau": 0.0,
                            "is_novel": False,
                            "reason": err_msg if not is_valid else "",
                        }

                        if is_valid and fn is not None:
                            # Evaluate on training split
                            valid_eval, train_score, train_time_ms = evaluate_on_dataset(
                                fn, train_instances, timeout_s=self.config.get("timeout_s", 2.0)
                            )
                            if valid_eval:
                                # Evaluate on validation split
                                _, val_score, _ = evaluate_on_dataset(
                                    fn, val_instances, timeout_s=self.config.get("timeout_s", 2.0)
                                )
                                # Novelty audit
                                novelty_info = audit_candidate_novelty_strengthened(
                                    code_str=code,
                                    fn=fn,
                                    audit_instances=audit_instances,
                                    baselines=self.baselines,
                                )

                                # Add to code memory if valid
                                code_memory.add_candidate(
                                    code=code,
                                    val_score=val_score,
                                    train_score=train_score,
                                    is_novel=novelty_info.get("is_novel", False),
                                    iteration=it,
                                    condition=cond,
                                )

                                if val_score < best_run_val_score:
                                    best_run_val_score = val_score
                                    best_run_candidate = code

                        # Record candidate
                        record = {
                            "condition": cond,
                            "seed": seed,
                            "iteration": it,
                            "is_valid": is_valid,
                            "train_score": train_score,
                            "val_score": val_score,
                            "novelty_classification": novelty_info.get("classification"),
                            "closest_baseline": novelty_info.get("closest_baseline"),
                            "exact_match_pct": novelty_info.get("exact_match_pct"),
                            "mean_kendall_tau": novelty_info.get("mean_kendall_tau"),
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
                            "code": code,
                        }

                        all_candidate_records.append(record)
                        prov_f.write(json.dumps(record) + "\n")
                        prov_f.flush()

                        status_str = "VALID" if is_valid else "INVALID"
                        print(f"  Result: {status_str} | Novelty: {novelty_info.get('classification')} | Val Tard: {val_score:.2f}")

                    run_summaries.append({
                        "condition": cond,
                        "seed": seed,
                        "best_val_score": best_run_val_score,
                        "beats_amddc_apd": best_run_val_score < ref_amddc_apd_score,
                        "best_code": best_run_candidate,
                    })

        # Write summary CSV
        if all_candidate_records:
            fieldnames = [
                "condition", "seed", "iteration", "is_valid", "train_score",
                "val_score", "novelty_classification", "closest_baseline",
                "exact_match_pct", "mean_kendall_tau", "is_novel",
                "beats_mdd", "beats_amddc_apd", "llm_model", "inference_time_ms",
                "generated_tokens", "prompt_hash", "response_hash"
            ]
            with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for rec in all_candidate_records:
                    writer.writerow(rec)

        print("\n" + "=" * 80)
        print("Discovery Session Completed Successfully.")
        print(f"Detailed cryptographic provenance logged to: {provenance_log_path}")
        print(f"Candidate summary CSV written to: {summary_csv_path}")
        print("=" * 80)

        return all_candidate_records
