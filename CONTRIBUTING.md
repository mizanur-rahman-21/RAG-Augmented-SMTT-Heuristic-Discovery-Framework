# Contributing Guidelines

Thank you for your interest in contributing to the SMTT RAG Heuristic Discovery framework.

## Code of Conduct & Rigor Standards

This repository prioritizes scientific reproducibility and absolute honesty:
1. **Never fabricate benchmark numbers**: All empirical outcomes must come directly from runnable code and traceable data.
2. **Provenance integrity**: Never bypass the LLM client with hard-coded heuristics or synthetic template fallbacks in the experiment path.
3. **Accurate attribution**: New heuristic compositions using existing local search concepts (e.g. adjacent pairwise interchange or Bean & Hall relocation) must be classified as `KNOWN_LOCAL_SEARCH_COMPOSITION`, not novel dispatching discoveries.

## Development Workflow

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. Set up the local environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1
   pip install -r requirements-lock.txt
   ```
3. Run existing tests to ensure baseline passes:
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   ```
4. Implement your changes, ensuring new tests are added for novel functionality.
5. Ensure linting passes and submit a pull request with a detailed description of your changes.
