"""
Discovery Engine v2 - Rapid Smoke Test
======================================
Executes a minimal, single-call verification of the entire v2 pipeline:
1. Loads v2 configuration.
2. Checks Ollama backend and Intel Arc GPU availability.
3. Generates 10 train, 10 val, and 5 audit benchmark instances.
4. Performs 1 real LLM generation under Evolution-RAG.
5. Extracts mathematical intuition + Python heuristic.
6. Compiles with LoopGuard protection and evaluates on SMTT instances.
7. Performs multi-tier novelty audit and multi-criteria utility scoring.
8. Logs full cryptographic provenance.

Usage:
  python run_smoke_v2.py
"""

import sys
import os
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.instance_generator import generate_benchmark_suite
from src.llm_client import create_llm_client
from src.experiment_runner_v2 import ExperimentRunnerV2


def load_config():
    config_file = PROJECT_ROOT / "config.v2.yaml"
    if not config_file.exists():
        config_file = PROJECT_ROOT / "config.v2.example.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 80)
    print("SMTT Improved Discovery Engine v2 - Smoke Test")
    print("=" * 80)

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

    smoke_cfg = config.get("smoke_test", {})
    seeds_cfg = config.get("seeds", {})

    n_train = smoke_cfg.get("train_instances", 10)
    n_val = smoke_cfg.get("val_instances", 10)
    n_audit = smoke_cfg.get("audit_instances", 5)

    print(f"\nGenerating small smoke datasets (Train: {n_train}, Val: {n_val}, Audit: {n_audit})...")
    train_insts = generate_benchmark_suite(n_jobs=20, instances_per_class=40, seed=seeds_cfg.get("train_seed", 10001))[:n_train]
    val_insts = generate_benchmark_suite(n_jobs=20, instances_per_class=40, seed=seeds_cfg.get("val_seed", 20002))[:n_val]
    audit_insts = generate_benchmark_suite(n_jobs=20, instances_per_class=40, seed=seeds_cfg.get("audit_seed", 99999))[:n_audit]

    runner = ExperimentRunnerV2(config=config, llm_client=client, project_root=PROJECT_ROOT)

    results = runner.run_discovery_session_v2(
        conditions=["evolution_rag"],
        discovery_seeds=[101],
        iterations_per_run=1,
        train_instances=train_insts,
        val_instances=val_insts,
        audit_instances=audit_insts,
        provenance_log_file="smoke_v2_provenance.jsonl",
        summary_csv_file="smoke_v2_summary.csv",
    )

    print(f"\nSmoke test complete! Result record count: {len(results)}")
    if results:
        rec = results[0]
        print("\n--- Smoke Test Verification Highlights ---")
        print(f"Status:             {rec['eval_status']}")
        print(f"Novelty:            {rec['novelty_classification']}")
        print(f"Validation Tard:    {rec['val_score']:.2f}")
        print(f"Composite Utility:  {rec['composite_utility']:.3f}")
        print(f"Inference Latency:  {rec['inference_time_ms']:.1f}ms")
        print(f"Extracted Reasoning (first 150 chars):\n  {rec['reasoning'][:150]}...")
        print(f"Prompt SHA-256:     {rec['prompt_hash'][:16]}...")
        print(f"Response SHA-256:   {rec['response_hash'][:16]}...")


if __name__ == "__main__":
    main()
