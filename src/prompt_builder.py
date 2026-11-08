"""
Prompt Builder for Controlled RAG Ablation Study
================================================
Builds strictly controlled prompts across the 4 conditions:
1. No-RAG: Base specification + Seed heuristic only.
2. Theory-RAG: Base specification + Seed heuristic + Theoretical literature context.
3. Code-Memory RAG: Base specification + Seed heuristic + Past discovered elite programs.
4. Dual-RAG: Base specification + Seed heuristic + Theory + Discovered code.

Strict Fairness Rules:
- No mention of 'Adjacent Pairwise Descent', 'API', or the final algorithm code.
- Identical base problem specification and seed heuristic across all conditions.
"""

from typing import Optional


BASE_PROBLEM_DESCRIPTION = """
You are an expert Operations Research scientist designing a heuristic algorithm for the Single-Machine Total Tardiness Problem (SMTT, notation: 1 || sum Tj).

Given:
- Processing times array p of length n (integer or float)
- Due dates array d of length n (integer or float)

Objective:
- Return a sequence (permutation) array of job indices (0, 1, ..., n-1) minimizing total tardiness:
  Tardiness = sum_{j=0}^{n-1} max(0, C_j - d_j)
  where C_j is the completion time of job j in the sequence.

Implementation Constraints:
- Provide a complete Python function:
  def custom_heuristic(p: np.ndarray, d: np.ndarray) -> np.ndarray:
- Must return a valid permutation of range(len(p)).
- Do not import external modules; numpy is already provided as np.
- Do not use file I/O, networking, system calls, exec, or eval.
"""

SEED_HEURISTIC_TEMPLATE = """
### Base Seed Heuristic (Modified Due Date Dynamic Rule):
```python
def custom_heuristic(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = len(p)
    current_time = 0.0
    U = list(np.argsort(p))
    S = []
    while U:
        pU = p[U].astype(float)
        dU = d[U].astype(float)
        mu = np.maximum(pU * 1.1 + current_time, dU)
        denom_rho = current_time + np.max(pU)
        rho = np.minimum(pU / (denom_rho if denom_rho > 0 else 1.0), 1.0)
        theta = (rho ** 2) / (1.0 + rho ** 2)
        mu = mu * (1.0 + theta)
        denom_sigma = current_time + np.mean(pU)
        sigma = pU / (denom_sigma if denom_sigma > 0 else 1.0)
        mu = mu + sigma
        best_pos = int(np.argmin(mu))
        best_job = U.pop(best_pos)
        S.append(best_job)
        current_time += float(p[best_job])
    return np.array(S, dtype=int)
```
"""


def build_ablation_prompt(
    condition: str,
    theory_context: str = "",
    code_memory_context: str = "",
) -> str:
    """
    Constructs a controlled prompt for the given condition.
    """
    parts = [BASE_PROBLEM_DESCRIPTION.strip(), SEED_HEURISTIC_TEMPLATE.strip()]

    if condition in ("theory_rag", "dual_rag") and theory_context.strip():
        parts.append(theory_context.strip())

    if condition in ("code_rag", "dual_rag") and code_memory_context.strip():
        parts.append(code_memory_context.strip())

    instructions = """
### Instruction:
Propose an improved heuristic for the Single-Machine Total Tardiness Problem.
You may redesign the dynamic priority index, incorporate lookahead, or introduce sequencing logic.
Return your answer with valid Python code enclosed in a ```python ... ``` block.
The function must be named `custom_heuristic(p: np.ndarray, d: np.ndarray) -> np.ndarray`.
"""
    parts.append(instructions.strip())

    return "\n\n".join(parts)
