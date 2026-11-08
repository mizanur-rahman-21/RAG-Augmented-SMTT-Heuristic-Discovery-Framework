"""
SMTT Heuristic Discovery - Full RAG Ablation Experiment
======================================================
Executes the comprehensive, statistically powered ablation study:
- 4 conditions: No-RAG, Theory-RAG, Code-Memory RAG, Dual-RAG
- 5 independent discovery seeds (101..105)
- 25 iterations per seed
- Total 500 real LLM inferences
- Full 800-instance training set (Seed 10001)
- Full 800-instance validation set (Seed 20002)
- 40-instance novelty screening set (Seed 99999)

Post-Discovery Evaluation:
- Selects exactly one best candidate per condition based strictly on validation score.
- Evaluates these 4 selected candidates against the 4 frozen test benchmark seeds
  (30003, 31001, 31002, 31003; total 3,200 instances) to compute final optimality gaps.

Note: Requires confirmation flag `--confirm` to avoid accidental compute costs.
"""

import sys
import os
import argparse
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.instance_generator import generate_benchmark_suite
from src.llm_client import create_llm_client
from src.experiment_runner import ExperimentRunner


def load_config():
    config_file = PROJECT_ROOT / "config.yaml"
    if not config_file.exists():
        config_file = PROJECT_ROOT / "config.example.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Full SMTT RAG Ablation Study Runner")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Mandatory confirmation flag to execute the full 500-call experiment.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SMTT RAG Ablation Full Study (500 Real LLM Inferences)")
    print("=" * 80)

    if not args.confirm:
        print("\n[SAFETY NOTICE] This script executes 500 real LLM generation calls,")
        print("evaluating each against 800 training and 800 validation instances.")
        print("Depending on your GPU/backend, this may take several hours.")
        print("\nTo execute, pass the explicit confirmation flag:")
        print("  python run_full_ablation.py --confirm")
        sys.exit(0)

    config = load_config()
    client = create_llm_client(config)

    # Health check
    health = client.health_check()
    if not health.get("online"):
        print(f"\n[ERROR] LLM Backend is OFFLINE: {health.get('error')}")
        print("Please start Ollama before running this script:")
        print("  1. ollama serve")
        print(f"  2. ollama pull {client.model}")
        sys.exit(1)

    if not health.get("target_model_present"):
        print(f"\n[ERROR] Model '{client.model}' not available on backend.")
        print(f"Please run: ollama pull {client.model}")
        sys.exit(1)

    full_cfg = config.get("full_ablation", {})
    seeds_cfg = config.get("seeds", {})
    conditions = config.get("conditions", ["no_rag", "theory_rag", "code_rag", "dual_rag"])

    discovery_seeds = full_cfg.get("discovery_seeds", [101, 102, 103, 104, 105])
    iterations = full_cfg.get("iterations_per_run", 25)
    n_train = full_cfg.get("train_instances", 800)
    n_val = full_cfg.get("val_instances", 800)
    n_audit = full_cfg.get("audit_instances", 40)

    print(f"\nGenerating full benchmark datasets...")
    train_insts = generate_benchmark_suite(n_jobs=20, instances_per_class=40, seed=seeds_cfg.get("train_seed", 10001))[:n_train]
    val_insts = generate_benchmark_suite(n_jobs=20, instances_per_class=40, seed=seeds_cfg.get("val_seed", 20002))[:n_val]
    audit_insts = generate_benchmark_suite(n_jobs=20, instances_per_class=40, seed=seeds_cfg.get("audit_seed", 99999))[:n_audit]

    runner = ExperimentRunner(config=config, llm_client=client, project_root=PROJECT_ROOT)

    results = runner.run_discovery_session(
        conditions=conditions,
        discovery_seeds=discovery_seeds,
        iterations_per_run=iterations,
        train_instances=train_insts,
        val_instances=val_insts,
        audit_instances=audit_insts,
        provenance_log_file="full_ablation_provenance.jsonl",
        summary_csv_file="full_ablation_summary.csv",
    )

    print(f"\nFull ablation study completed successfully. Generated {len(results)} candidate records.")


if __name__ == "__main__":
    main()
