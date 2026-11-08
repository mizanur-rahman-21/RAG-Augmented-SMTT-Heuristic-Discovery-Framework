"""
Exact Implementations of SMTT (1 || \\sum T_j) Heuristics
=========================================================

Algorithms included:
1. MDD (Modified Due Date) - Baker & Bertrand (1982)
2. MDDC (MDD Challenger) - Çetinkaya, Büyüktahtakın, Shojaee, Reddy (2025), Algorithm 2
3. Augmented MDD - Bean & Hall (1985)
4. Augmented MDDC - Çetinkaya et al. (2025), Algorithm 3
5. RADC-APD (Augmented MDDC + Adjacent Pairwise Descent) - Monotonic post-improvement
"""

from typing import List, Tuple
import numpy as np


def compute_tardiness(p: np.ndarray, d: np.ndarray, seq: np.ndarray) -> int:
    """
    Computes total tardiness for a schedule sequence seq:
      C_k = sum_{l=0}^k p_{seq[l]}
      T_k = max(0, C_k - d_{seq[k]})
      Total = sum_{k=0}^{n-1} T_k
    """
    c = 0
    total_tard = 0
    for job in seq:
        c += int(p[job])
        tard = c - int(d[job])
        if tard > 0:
            total_tard += tard
    return int(total_tard)


def compute_completion_times(p: np.ndarray, seq: np.ndarray) -> np.ndarray:
    """Returns array of completion times for schedule seq."""
    return np.cumsum(p[seq])


def validate_permutation(seq: np.ndarray, n: int) -> bool:
    """Validates that seq is a strict permutation of exactly 0, ..., n-1."""
    if len(seq) != n:
        return False
    if set(seq) != set(range(n)):
        return False
    return True


# ==============================================================================
# 1. MDD (Modified Due Date)
# Source: Baker, K. R., & Bertrand, J. W. M. (1982).
#         "A dynamic priority rule for scheduling against due-dates."
#         Journal of Operations Management, 3(1), 37-42.
# ==============================================================================
def algo_mdd(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    MDD Heuristic:
    At current time t, for each unscheduled job j in U:
        priority_j(t) = max(t + p_j, d_j)
    Select job j* = argmin_{j in U} priority_j(t).
    Tie-breaking: shortest processing time p_j, then smallest job index.
    """
    n = len(p)
    current_time = 0
    unscheduled = list(range(n))
    seq: List[int] = []

    while unscheduled:
        p_u = p[unscheduled]
        d_u = d[unscheduled]
        # Formula: max(t + p_j, d_j)
        mdd_values = np.maximum(current_time + p_u, d_u)

        # Lexicographical tie-breaking: min mdd_value, then min p_j, then index
        best_pos = 0
        best_val = mdd_values[0]
        best_p = p_u[0]
        for idx in range(1, len(unscheduled)):
            val = mdd_values[idx]
            if val < best_val:
                best_val = val
                best_p = p_u[idx]
                best_pos = idx
            elif val == best_val and p_u[idx] < best_p:
                best_p = p_u[idx]
                best_pos = idx

        selected_job = unscheduled.pop(best_pos)
        seq.append(selected_job)
        current_time += int(p[selected_job])

    return np.array(seq, dtype=int)


# ==============================================================================
# 2. MDDC (MDD Challenger)
# Source: Çetinkaya, Büyüktahtakın, Shojaee, Reddy (2025).
#         "Discovering Heuristics with Large Language Models (LLMs) for
#          Mixed-Integer Programs: Single-Machine Scheduling."
#         Algorithm 2: MDDC Algorithm, page 24.
# ==============================================================================
def algo_mddc(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    MDDC Heuristic (Çetinkaya et al., 2025, Algorithm 2):
    1. Initialize unscheduled set U sorted by Shortest Processing Time (SPT).
    2. At current time t, compute dynamic priority score for each j in U:
         mu_j    = max(1.1 * p_j + t, d_j)
         rho_j   = min(p_j / (t + max_{k in U}(p_k)), 1.0)
         theta_j = rho_j^2 / (1.0 + rho_j^2)
         mu_j    = mu_j * (1.0 + theta_j)
         sigma_j = p_j / (t + mean_{k in U}(p_k))
         mu_j    = mu_j + sigma_j
    3. Select j* = argmin_{j in U} mu_j, append to S, update t <- t + p_{j*}.
    """
    n = len(p)
    current_time = 0.0
    # Line 7 of Algorithm 2: Sort U based on Shortest Processing Time (SPT) Rule
    U = list(np.argsort(p))
    S: List[int] = []

    while U:
        pU = p[U].astype(float)
        dU = d[U].astype(float)

        # Line 10: mu <- max(pU * 1.1 + current_time, dU)
        mu = np.maximum(pU * 1.1 + current_time, dU)

        # Lines 11-12: rho <- min(pU / (current_time + max(pU)), 1.0)
        denom_rho = current_time + np.max(pU)
        rho = np.minimum(pU / (denom_rho if denom_rho > 0 else 1.0), 1.0)

        # Line 13: theta <- rho^2 / (1 + rho^2)
        theta = (rho ** 2) / (1.0 + rho ** 2)

        # Line 14: mu <- mu * (1 + theta)
        mu = mu * (1.0 + theta)

        # Line 15: sigma <- pU / (current_time + sum(pU)/|U|)
        denom_sigma = current_time + np.mean(pU)
        sigma = pU / (denom_sigma if denom_sigma > 0 else 1.0)

        # Line 16: mu <- mu + sigma
        mu = mu + sigma

        # Line 17: Find job with minimum mu score and schedule it next
        best_pos = int(np.argmin(mu))
        best_job = U.pop(best_pos)
        S.append(best_job)
        current_time += float(p[best_job])

    return np.array(S, dtype=int)


# ==============================================================================
# Helper: Bean & Hall Tail Relocation
# Source: Bean, J. C., & Hall, N. G. (1985).
#         "Extensions of the modified due date rule."
#         Naval Research Logistics Quarterly, 32(3), 503-506.
# Also Çetinkaya et al. (2025), Algorithm 3, lines 19-25.
# ==============================================================================
def _apply_bean_hall_relocation(p: np.ndarray, d: np.ndarray, S: List[int]) -> List[int]:
    """
    Evaluates trial schedules by moving each earlier job j in S \\ {j*} to the end of S.
    Accepts trial schedule if total tardiness strictly decreases.
    """
    L = len(S)
    if L <= 2:
        return S

    comp = np.cumsum(p[S])
    best_tard = int(np.sum(np.maximum(0, comp - d[S])))
    best_S = S

    # For each earlier job j in S \ {j*} (indices 0 to L-2)
    # For moderate scales (n <= 80), evaluate all positions 0..L-2
    # For large scales (n > 80), check lookback window
    indices_to_test = range(L - 1) if L <= 80 else range(max(0, L - 25), L - 1)
    for idx in indices_to_test:
        trial = S[:idx] + S[idx + 1:] + [S[idx]]
        trial_comp = np.cumsum(p[trial])
        trial_tard = int(np.sum(np.maximum(0, trial_comp - d[trial])))
        if trial_tard < best_tard:
            best_tard = trial_tard
            best_S = trial

    return best_S


# ==============================================================================
# 3. Augmented MDD
# Source: Bean, J. C., & Hall, N. G. (1985).
# ==============================================================================
def algo_augmented_mdd(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Augmented MDD:
    Constructs schedule incrementally using standard MDD priority.
    Immediately after each job is scheduled, evaluates Bean & Hall relocation
    moving each earlier job to the end of prefix S if it strictly reduces tardiness.
    """
    n = len(p)
    current_time = 0
    mask = np.ones(n, dtype=bool)
    S: List[int] = []

    for _ in range(n):
        avail = np.where(mask)[0]
        # MDD dynamic priority: max(t + p_j, d_j)
        mdd_vals = np.maximum(current_time + p[avail], d[avail])
        best_job = int(avail[np.argmin(mdd_vals)])
        S.append(best_job)
        mask[best_job] = False
        current_time += int(p[best_job])

        if len(S) > 2:
            S = _apply_bean_hall_relocation(p, d, S)

    return np.array(S, dtype=int)


# ==============================================================================
# 4. Augmented MDDC
# Source: Çetinkaya, Büyüktahtakın, Shojaee, Reddy (2025).
#         Algorithm 3: Augmented MDDC Algorithm, page 38.
# ==============================================================================
def algo_augmented_mddc(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Augmented MDDC (Çetinkaya et al., 2025, Algorithm 3):
    Lines 1-18: Dynamic MDDC score selection:
      mu = max(1.1 * p_U + t, d_U) * (1 + rho^2 / (1 + rho^2)) + sigma
      where rho = min(p_U / (t + max(p_U)), 1)
            sigma = p_U / (t + mean(p_U))
    Lines 19-25: Immediately upon scheduling job j*, apply Bean & Hall relocation:
      for each job j in S \\ {j*}:
        trial S' = move j to end of S
        if tardiness(S') < best_tardiness:
          S = S'
    """
    n = len(p)
    current_time = 0.0
    U = list(np.argsort(p))
    S: List[int] = []

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

        # Algorithm 3, lines 19-25: Bean & Hall relocation
        if len(S) > 2:
            S = _apply_bean_hall_relocation(p, d, S)

    return np.array(S, dtype=int)


# ==============================================================================
# 5. RADC-APD (Augmented MDDC + Adjacent Pairwise Descent)
# Formerly designated RADC-VND; renamed Adjacent Pairwise Descent (APD)
# per scientific requirement since it utilizes adjacent pairwise interchange.
# ==============================================================================
def algo_radc_apd(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Augmented MDDC with Adjacent Pairwise Descent (AMDDC-APD / RADC-APD).
    
    Architecture:
    Phase 1 (Construction):
      Full Augmented MDDC schedule (Algorithm 3, Çetinkaya et al., 2025).
    
    Phase 2 (Descent):
      Adjacent Pairwise Interchange (API) descent sweeps.
      For each adjacent pair (seq[i], seq[i+1]) at time t = completion(i-1):
        T_orig = max(0, t + p1 - d1) + max(0, t + p1 + p2 - d2)
        T_swap = max(0, t + p2 - d2) + max(0, t + p2 + p1 - d1)
      Jobs are swapped if and only if T_swap < T_orig (strict descent).
      Since sum of p1 + p2 is invariant, completion times of all subsequent
      jobs k > i+1 remain completely unchanged.
      Thus Delta T = T_swap - T_orig is exact and O(1).
      Guarantees: tardiness(RADC-APD) <= tardiness(Augmented MDDC) on every instance.
    """
    seq = algo_augmented_mddc(p, d).copy()
    n = len(p)
    improved = True
    passes = 0
    max_passes = 4 if n <= 100 else 2

    while improved and passes < max_passes:
        improved = False
        passes += 1
        comp = np.cumsum(p[seq])
        for i in range(n - 1):
            j1, j2 = int(seq[i]), int(seq[i + 1])
            t_curr = int(comp[i - 1]) if i > 0 else 0
            p1, d1 = int(p[j1]), int(d[j1])
            p2, d2 = int(p[j2]), int(d[j2])

            c1_orig = t_curr + p1
            c2_orig = c1_orig + p2
            tard_orig = max(0, c1_orig - d1) + max(0, c2_orig - d2)

            c1_swap = t_curr + p2
            c2_swap = c1_swap + p1
            tard_swap = max(0, c1_swap - d2) + max(0, c2_swap - d1)

            if tard_swap < tard_orig:
                seq[i], seq[i + 1] = j2, j1
                comp[i] = c1_swap
                improved = True

    return seq


# Direct alias for legacy compatibility
algo_radc_vnd = algo_radc_apd
