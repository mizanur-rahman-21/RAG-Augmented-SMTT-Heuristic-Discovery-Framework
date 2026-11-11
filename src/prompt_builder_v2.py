r"""
Prompt Builder v2 for Advanced SMTT Heuristic Discovery
======================================================
Engineered for deep Operations Research heuristic discovery on 1 || \sum T_j.

Key Capabilities:
1. State-Dependent SMTT Feature Space:
   - Machine State: t (current time), P(U) (remaining workload), |U| (remaining job count).
   - Job State: p_j (processing time), d_j (due date), s_j(t) = d_j - (t + p_j) (dynamic slack).
   - Urgency & Contention: s_j(t)/p_j (slack ratio), p_j/P(U) (workload fraction),
     bottleneck indicator, quantile of d_j in remaining set.
   - Dominance & Interaction: Emmons (1969) dominance relations, Fisher's dual-index regimes,
     nonlinear transitions between slack-dominated (early) and tardiness-dominated (late) phases.

2. Mandatory Mathematical Reasoning Requirement:
   - Demands formal mathematical formulation and intuition BEFORE the Python code block.

3. Strict Anti-Triviality Constraints:
   - Explicitly forbids trivial convex combinations (w1*p + w2*d).
   - Forbids simple perturbation of MDDC constants.
   - Forbids embedding local search / adjacent pairwise descent (belongs in search layer).
   - Forbids superficial renaming.
"""

from typing import Optional, Dict, Any, List


BASE_PROBLEM_DESCRIPTION_V2 = """
You are a Principal Operations Research Scientist and Algorithmist specializing in single-machine scheduling theory and automated heuristic discovery.

### Problem Formulation:
Problem: Single-Machine Total Tardiness (SMTT, Graham notation: 1 || \\sum T_j).
Inputs:
  - Vector p of length n: Processing times p_j > 0 for jobs j in {0, 1, ..., n-1}.
  - Vector d of length n: Due dates d_j >= 0 for jobs j in {0, 1, ..., n-1}.
Objective:
  - Find a permutation schedule S = (j_1, j_2, ..., j_n) of all jobs minimizing total tardiness:
    Total Tardiness = \\sum_{k=1}^n \\max(0, C_{j_k} - d_{j_k})
    where completion time C_{j_k} = \\sum_{i=1}^k p_{j_i}.
Characteristics:
  - Strongly NP-hard (Du and Leung, 1990).
  - Regular measure of performance (optimal schedule has no idle time).

### Relevant Theoretical Mechanics & State Variables:
At scheduling step k, let:
  - S: sequence of already scheduled jobs.
  - current_time t: \\sum_{i \\in S} p_i (starts at 0.0).
  - U: set of unscheduled jobs {0, 1, ..., n-1} \\ S.
  - Workload remaining: P(U) = \\sum_{j \\in U} p_j.
  - Workload fraction: \\omega_j = p_j / P(U).
  - Completion time if job j is scheduled next: C_j(t) = t + p_j.
  - Dynamic Slack: s_j(t) = d_j - (t + p_j).
    * If s_j(t) < 0: Job j is already guaranteed to be tardy.
    * If s_j(t) >= 0: Job j can still complete on time.
  - Critical Slack Ratio: \\kappa_j(t) = s_j(t) / p_j.
  - Lookahead Tardiness Impact: If job j is scheduled next, the start times of all remaining
    jobs in U \\ {j} are delayed by exactly p_j.

### Mathematical Directions for Genuinely Novel Priority Rules:
Explore rich, nonlinear scheduling mechanics rather than trivial linear scoring:
1. Regime-Switching Dynamics:
   Smooth or piecewise transitions between:
   - Urgent phase (s_j(t) <= 0): Prioritize by Modified Smith / SPT-like ratios (p_j / \\max(1, tardiness penalty)).
   - Non-urgent phase (s_j(t) > 0): Prioritize by modified EDD or slack safety margins.
2. Contention & Opportunity Cost:
   Penalize jobs with large p_j that push multiple jobs in U with small positive slack into tardiness.
   Opportunity cost estimate: \\sum_{i \\in U, i \\neq j} \\max(0, (t + p_j + p_i) - d_i).
3. Non-linear Normalization:
   Use sigmoid, softmax, or hyperbolic tangent scalings over relative slack:
   \\sigma(s_j(t) / (\\text{mean}(s(U)) + \\epsilon)).
4. Joint Distribution Interaction:
   Interactions between processing time quantile in U and due-date quantile in U.

### Strict Negative Constraints (What NOT to do):
- DO NOT generate simple linear combinations: score = a * p + b * d.
- DO NOT re-implement classical MDD: score = max(t + p_j, d_j).
- DO NOT simply tweak constant coefficients of MDDC (e.g., replacing 1.1 with 1.2 or rho**2 with rho**3).
- DO NOT implement Adjacent Pairwise Descent (APD/API) or Bean-Hall relocation loops. The function must be a pure, clean, constructive dynamic dispatching heuristic operating in O(n^2) time.
- DO NOT use external Python modules (only numpy as np and standard Python math/builtins are allowed).
"""

OUTPUT_FORMAT_REQUIREMENT = """
### Required Output Structure:
Your response MUST contain exactly two sections:

#### Section 1: Mathematical Intuition & Formulation
Provide a concise theoretical explanation:
1. Name of your heuristic concept.
2. Formal mathematical equations for your dynamic priority index score(j, t, U).
3. Operations Research rationale: Explain why the interaction between p_j, d_j, t, and U reduces total tardiness.

#### Section 2: Executable Python Implementation
Enclose the complete Python function in a single ```python ... ``` code block.
The function signature must be:
```python
def custom_heuristic(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = len(p)
    # ... your algorithm ...
    return np.array(schedule, dtype=int)
```
The returned array must be a valid permutation of range(n).
"""


def build_ablation_prompt_v2(
    condition: str,
    theory_context: str = "",
    code_memory_context: str = "",
    evolution_context: str = "",
) -> str:
    """
    Constructs an enhanced v2 prompt for the specified experimental condition.
    
    Conditions supported:
    - 'no_rag': Formal OR problem formulation + state space + strict negative constraints.
    - 'theory_rag': no_rag + targeted theoretical excerpts from scheduling literature.
    - 'code_rag': no_rag + elite discovered code exemplars with performance metrics.
    - 'dual_rag': no_rag + theoretical literature + elite discovered code.
    - 'evolution_rag': no_rag + elite memory + failed attempts memory + explicit mutation directives.
    """
    sections = [BASE_PROBLEM_DESCRIPTION_V2.strip()]

    # Inject condition-specific context
    if condition in ("theory_rag", "dual_rag") and theory_context.strip():
        sections.append(theory_context.strip())

    if condition in ("code_rag", "dual_rag") and code_memory_context.strip():
        sections.append(code_memory_context.strip())

    if condition == "evolution_rag" and evolution_context.strip():
        sections.append(evolution_context.strip())

    sections.append(OUTPUT_FORMAT_REQUIREMENT.strip())

    return "\n\n" + ("\n\n" + "=" * 70 + "\n\n").join(sections) + "\n\n"
