# Pre-Release & Reproducibility Checklist

Before tagging or releasing any version of this repository, ensure all following checks pass:

- [ ] **Unit Tests**: All unit tests in `tests/` pass without errors (`python -m unittest discover -s tests -p "test_*.py"`).
- [ ] **Exact DP Solver Equivalence**: DP matches brute-force on instances with $n \in [3, 8]$.
- [ ] **Baseline Monotonicity**: AMDDC + APD strictly non-worsening on all test instances.
- [ ] **Path Portability**: No hard-coded absolute paths (e.g. `C:\Users\...` or `D:\Thesis\...`) exist in repository source code.
- [ ] **Dependency Pinning**: All dependencies in `requirements-lock.txt` match active environment.
- [ ] **Cryptographic Integrity**: SHA-256 hashes of frozen baseline data and results verified against `frozen_baseline/SHA256SUMS.txt`.
- [ ] **Zero Synthetic Fallbacks**: Confirm that `run_pilot.py` and `run_full_ablation.py` contain zero template generators or mock LLM clients.
- [ ] **Preflight Pass**: `run_preflight.py` exits cleanly (or gracefully reports connection failure if Ollama is not launched).
- [ ] **Clean Git State**: No credentials, `.env` files, temporary artifacts, or personal paths staged for commit.
