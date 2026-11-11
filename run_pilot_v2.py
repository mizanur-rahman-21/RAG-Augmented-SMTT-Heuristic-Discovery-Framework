"""
Discovery Engine v2 - Pilot Ablation Run
========================================
Runs controlled pilot study across all 5 conditions:
1. No-RAG
2. Theory-RAG
3. Code-Memory RAG
4. Dual-RAG
5. Evolution-RAG

Default configuration:
- 2 discovery seeds (101, 102)
- 3 iterations per seed
- Total 30 real LLM inferences
- Train subset: 50 instances (Seed 10001)
- Validation subset: 50 instances (Seed 20002)
- Audit subset: 20 instances (Seed 99999)

Outputs:
- logs/pilot_v2_provenance.jsonl
- results/pilot_v2_summary.csv
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
    print("SMTT RAG Ablation Pilot Study v2 (5 Conditions, Real LLM Inference)")
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

    pilot_cfg = config.get("pilot_v2", {})
    seeds_cfg = config.get("seeds", {})
    conditions = config.get("conditions", ["no_rag", "theory_rag", "code_rag", "dual_rag", "evolution_rag"])

    discovery_seeds = pilot_cfg.get("discovery_seeds", [101, 102])
    iterations = pilot_cfg.get("iterations_per_run", 3)
    n_train = pilot_cfg.get("train_instances", 50)
    n_val = pilot_cfg.get("val_instances", 50)
    n_audit = pilot_cfg.get("audit_instances", 20)

    print(f"\nGenerating datasets (Train Seed: {seeds_cfg.get('train_seed')}, Val Seed: {seeds_cfg.get('val_seed')})...")
    train_insts = generate_benchmark_suite(n_jobs=20, instances_per_class=40, seed=seeds_cfg.get("train_seed", 10001))[:n_train]
    val_insts = generate_benchmark_suite(n_jobs=20, instances_per_class=40, seed=seeds_cfg.get("val_seed", 20002))[:n_val]
    audit_insts = generate_benchmark_suite(n_jobs=20, instances_per_class=40, seed=seeds_cfg.get("audit_seed", 99999))[:n_audit]

    runner = ExperimentRunnerV2(config=config, llm_client=client, project_root=PROJECT_ROOT)

    results = runner.run_discovery_session_v2(
        conditions=conditions,
        discovery_seeds=discovery_seeds,
        iterations_per_run=iterations,
        train_instances=train_insts,
        val_instances=val_insts,
        audit_instances=audit_insts,
        provenance_log_file="pilot_v2_provenance.jsonl",
        summary_csv_file="pilot_v2_summary.csv",
    )

    print(f"\nPilot v2 run complete. Generated {len(results)} candidate records.")


if __name__ == "__main__":
    main()
