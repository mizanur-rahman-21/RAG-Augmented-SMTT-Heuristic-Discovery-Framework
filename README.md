# SMTT Heuristic Discovery with RAG-Augmented LLMs

[![CI](https://github.com/placeholder/smtt-rag-heuristic-discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/placeholder/smtt-rag-heuristic-discovery/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A rigorous, reproducible research framework exploring whether Retrieval-Augmented Generation (RAG)—via scheduling theory literature and code memory—enables Large Language Models to discover novel, high-performance dispatching heuristics for the Single-Machine Total Tardiness Problem ($1||\sum T_j$).

---

## Table of Contents

- [Research Question](#research-question)
- [Scientific Rigor & Limitations](#scientific-rigor--limitations)
- [Repository Structure](#repository-structure)
- [Hardware & Software Prerequisites](#hardware--software-prerequisites)
- [Installation & Quick Start](#installation--quick-start)
- [Ollama & LLM Backend Setup](#ollama--llm-backend-setup)
- [Running Experiments](#running-experiments)
  - [1. Preflight Diagnostics](#1-preflight-diagnostics)
  - [2. Pilot Study](#2-pilot-study)
  - [3. Full Ablation Study](#3-full-ablation-study)
- [Experimental Protocol & Data Isolation](#experimental-protocol--data-isolation)
- [Cryptographic Provenance](#cryptographic-provenance)
- [Reproducibility & Verification](#reproducibility--verification)
- [Citation](#citation)

---

## Research Question

> **Does Theory-RAG, Code-Memory RAG, or Dual-RAG improve the probability, quality, novelty, and sample efficiency of discovering valid SMTT heuristics compared with No-RAG LLM discovery?**

To answer this, we conduct a controlled four-way ablation study:
1. **No-RAG**: Base task specification and MDDC seed heuristic only.
2. **Theory-RAG**: Base specification + excerpts from classical scheduling literature (Emmons 1969; Baker & Bertrand 1982; Potts & Van Wassenhove 1982; Bean & Hall 1985).
3. **Code-Memory RAG**: Base specification + memory of valid, elite candidate heuristics discovered in preceding iterations.
4. **Dual-RAG**: Simultaneous retrieval of theoretical domain knowledge and past elite code implementations.

---

## Scientific Rigor & Limitations

### 1. Honest Framing of Baselines
The repository includes our frozen, independently audited benchmark implementation: **Augmented MDDC + Adjacent Pairwise Descent (AMDDC + APD)**. 
- *Important*: The descent phase in AMDDC + APD corresponds to the classical Adjacent Pairwise Interchange (API) neighborhood formalized by Emmons (1969) and Baker (1974). 
- It is an empirical local search enhancement that achieves state-of-the-art results across 3,200 held-out Potts & Van Wassenhove benchmark instances (347 wins, 2,853 ties, 0 losses against Augmented MDDC; Wilcoxon $p < 10^{-12}$).
- **It is NOT a newly discovered priority dispatching rule.** The RAG discovery experiment is evaluated separately and under strict fairness constraints.

### 2. Fair Prompting Constraints
To prevent circular evaluation, prompts across all conditions:
- **Never** mention "Adjacent Pairwise Descent", "API", or the AMDDC + APD final algorithm.
- Keep the backbone LLM, decoding temperature, context window, execution timeouts, and instance distribution identical across all four conditions.
- Strictly isolate training, validation, novelty audit, and locked test datasets.

---

## Repository Structure

```text
smtt-rag-heuristic-discovery/
├── README.md                      # Complete project documentation
├── LICENSE                        # MIT License
├── CITATION.cff                   # Citation metadata
├── CONTRIBUTING.md                # Contribution guidelines
├── RELEASE_CHECKLIST.md           # Pre-release checklist
├── requirements.txt               # Base dependencies
├── requirements-lock.txt          # Pinned exact versions
├── config.example.yaml            # Template experiment configuration
├── setup.ps1                      # Windows PowerShell automated setup
├── run_preflight.py               # Preflight environment and LLM diagnostic
├── run_pilot.py                   # Pilot ablation experiment (24 LLM calls)
├── run_full_ablation.py           # Full ablation experiment (500 LLM calls)
├── src/                           # Core research modules
│   ├── __init__.py
│   ├── baselines.py               # Exact implementations of 8 scheduling heuristics
│   ├── exact_solver.py            # Exact Bitmask DP & Brute-force solvers
│   ├── instance_generator.py      # Potts & Van Wassenhove (1982) benchmark generator
│   ├── evaluator.py               # AST sandbox scanner & 4-tier novelty audit
│   ├── retrieval.py               # Theory & Code-Memory RAG retrievers
│   ├── prompt_builder.py          # Controlled prompt construction
│   ├── llm_client.py              # LLM client with full cryptographic provenance
│   └── experiment_runner.py       # Orchestration loop for discovery experiments
├── tests/                         # Comprehensive unit test suite
│   ├── test_baselines.py          # Permutation validity & monotonicity tests
│   ├── test_exact_solver.py       # DP vs Brute-force equivalence (n <= 8)
│   ├── test_sandbox_and_audit.py  # AST safety & clone detection tests
│   └── test_llm_client.py         # Response parsing & offline handling
├── theory_kb/                     # Domain knowledge base
│   ├── baker_bertrand_1982.txt
│   ├── bean_hall_1985_relocation.txt
│   ├── emmons_1969_dominance.txt
│   └── potts_wassenhove_1982.txt
├── frozen_baseline/               # Complete audited 3,200-instance baseline artifact
│   ├── algorithms.py
│   ├── exact_solver.py
│   ├── verify_benchmarks.py
│   ├── SHA256SUMS.txt
│   └── results/
├── docs/                          # Detailed experiment documentation
│   └── EXPERIMENT_PROTOCOL.md
├── results/                       # CSV summaries and generated reports (.gitkeep)
├── logs/                          # JSONL cryptographic provenance logs (.gitkeep)
└── data/                          # Pre-generated benchmark datasets (.gitkeep)
```

---

## Hardware & Software Prerequisites

- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python**: 3.10, 3.11, or 3.12 (with NumPy, SciPy, Numba, PyYAML, Requests).
- **RAM**: Minimum 8 GB recommended (16 GB for running 7B LLM models locally).
- **GPU (Optional)**: NVIDIA GPU with CUDA for local Ollama acceleration (CPU inference also supported).

---

## Installation & Quick Start

### Windows (PowerShell)
Clone the repository and run the setup script:
```powershell
git clone <repo-url>
cd smtt-rag-heuristic-discovery
.\setup.ps1
```

### Manual Installation
```bash
git clone <repo-url>
cd smtt-rag-heuristic-discovery

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install exact pinned dependencies
pip install -r requirements-lock.txt

# Run verification tests
python -m unittest discover -s tests -p "test_*.py"
```

---

## Ollama & LLM Backend Setup

By default, the framework utilizes **Ollama** for zero-cost, local LLM inference.

1. **Install Ollama**: Download from [ollama.com](https://ollama.com).
2. **Start the Ollama daemon**:
   ```bash
   ollama serve
   ```
3. **Download the recommended model**:
   ```bash
   ollama pull qwen2.5-coder:7b-instruct-q4_K_M
   ```
4. **Configuration**:
   Copy `config.example.yaml` to `config.yaml`:
   ```bash
   cp config.example.yaml config.yaml
   ```
   Modify `config.yaml` to change the model, endpoint, or decoding parameters. (OpenAI-compatible endpoints such as vLLM or LocalAI are also supported).

---

## Running Experiments

### 1. Preflight Diagnostics
Before running any experiment, execute the preflight script to ensure your environment, Ollama connectivity, model availability, AST sandbox, and provenance logging are all operating correctly:
```bash
python run_preflight.py
```
*Note: If Ollama is offline or the model is missing, this script fails cleanly with clear remediation instructions.*

### 2. Pilot Study
Run a controlled pilot (4 conditions $\times$ 2 seeds $\times$ 3 iterations = 24 real LLM inferences):
```bash
python run_pilot.py
```
- Trains on 50 instances from Seed 10001.
- Validates on 50 instances from Seed 20002.
- Logs full cryptographic provenance to `logs/pilot_candidates_provenance.jsonl`.
- Outputs summary table to `results/pilot_summary.csv`.

### 3. Full Ablation Study
To launch the full 500-call ablation experiment (4 conditions $\times$ 5 seeds $\times$ 25 iterations):
```bash
python run_full_ablation.py --confirm
```
*(The `--confirm` flag prevents accidental execution of long-running GPU/CPU experiments).*

---

## Experimental Protocol & Data Isolation

To prevent data contamination and overfitting, the benchmark suite follows Potts & Van Wassenhove (1982) with strict separation:

| Dataset | Seed | Instances | Purpose |
| :--- | :--- | :--- | :--- |
| **Train Set** | `10001` | 800 | Evaluating candidates during discovery & in-prompt memory scoring |
| **Validation Set** | `20002` | 800 | Selecting the single best candidate per condition/seed |
| **Audit Set** | `99999` | 40 | Fast equivalence and novelty classification |
| **Held-Out Test Set** | `30003`, `31001`, `31002`, `31003` | 3,200 | **LOCKED**. Evaluated exactly once after all discovery is finished |

---

## Cryptographic Provenance

Every candidate heuristic generated through this framework is recorded in a tamper-evident JSONL stream containing:
- **`model` & `backend`**: Exact LLM checkpoint and serving mechanism.
- **`timestamp_utc`**: Precise ISO-8601 UTC timestamp.
- **`prompt_hash`**: SHA-256 digest of the exact prompt string delivered to the LLM.
- **`response_hash`**: SHA-256 digest of the raw string returned by the LLM.
- **`generated_tokens`**: Token count reported by the backend.
- **`inference_time_ms`**: Measured wall-clock latency.
- **`extracted_code`**: Cleaned executable Python code.
- **`novelty_classification`**: Class label (`CLONE`, `KNOWN_LOCAL_SEARCH_COMPOSITION`, or `GENUINE_NEW_STRUCTURE`).

Zero mock, template, or fallback synthetic generation is permitted in production experiment execution.

---

## Reproducibility & Verification

To verify the frozen baseline results:
```bash
cd frozen_baseline
python verify_benchmarks.py
```
To run full unit tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## Citation

If you use this codebase or baseline implementations in your research, please cite:

```bibtex
@software{smtt_rag_heuristic_discovery_2026,
  author = {SMTT Research Contributors},
  title = {RAG-Augmented LLM Heuristic Discovery for Single-Machine Total Tardiness Scheduling},
  year = {2026},
  url = {https://github.com/placeholder/smtt-rag-heuristic-discovery}
}
```
