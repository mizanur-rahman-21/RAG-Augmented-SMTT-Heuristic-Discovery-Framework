"""
Independent Baseline Audit: Implementations of Augmented MDDC and APD
=====================================================================

This module provides two independent, rigorously documented implementations of
Augmented MDDC directly from the paper pseudocode (Çetinkaya et al. 2025, Algorithm 3),
documenting both defensible interpretations of lines 19-25:

Interpretation 1 (Sequential / First-Improvement):
  Lines 19-25 state:
    for each job j in S \\ {j*} do
      Compute current total tardiness; set best tardiness <- value
      Construct trial schedule S' by moving job j to the end of S
      Compute tardiness of S' as tardiness
      if tardiness < best tardiness then
        S <- S'
        best tardiness <- tardiness
  In this interpretation, S is replaced immediately during the loop, meaning
  subsequent jobs in S \\ {j*} are evaluated relative to the newly modified S.

Interpretation 2 (Best-Improvement / Bean & Hall 1985):
  In Bean & Hall (1985)'s original MDD extension, all jobs in S \\ {j*} are tested
  as trial relocations from the baseline schedule, and the single candidate that yields
  the lowest overall tardiness is selected (if strictly better than the baseline).

Also includes:
  - current `algo_augmented_mddc`
  - `apply_apd`: Pure Adjacent Pairwise Descent (API sweeps)
"""

from typing import List, Tuple
import numpy as np


def compute_tardiness(p: np.ndarray, d: np.ndarray, seq: np.ndarray) -> int:
    """Computes exact total tardiness for sequence seq."""
    c = 0
    tard = 0
    for job in seq:
        c += int(p[job])
        diff = c - int(d[job])
        if diff > 0:
            tard += diff
    return int(tard)


def validate_permutation(seq: np.ndarray, n: int) -> bool:
    """Verifies that seq is a strict permutation of 0..n-1."""
    return len(seq) == n and set(seq) == set(range(n))


# ==============================================================================
# 1. Reference Implementation (Interpretation 1: Sequential Update)
# Directly from Algorithm 3 of Çetinkaya et al. (2025)
# ==============================================================================
def augmented_mddc_reference_sequential(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Augmented MDDC - Reference Implementation (Sequential / First-Improvement).
    Direct line-by-line translation of Algorithm 3 (Çetinkaya et al., 2025, page 38).
    
    Lines 19-25 are implemented as immediate sequential replacement:
    Whenever a trial relocation of job j to the end improves tardiness,
    S is immediately updated to S', and subsequent jobs are tested against this new S.
    """
    n = len(p)
    current_time = 0.0
    # Line 6: Sort U based on Shortest Processing Time (SPT) rule
    U = list(np.argsort(p))
    S: List[int] = []

    while U:
        pU = p[U].astype(float)
        dU = d[U].astype(float)

        # Line 9: mu <- max(pU * 1.1 + current_time, dU)
        mu = np.maximum(pU * 1.1 + current_time, dU)

        # Lines 10-12: rho and theta
        denom_rho = current_time + np.max(pU)
        rho = np.minimum(pU / (denom_rho if denom_rho > 0 else 1.0), 1.0)
        theta = (rho ** 2) / (1.0 + rho ** 2)
        mu = mu * (1.0 + theta)

        # Lines 13-14: sigma and final mu
        denom_sigma = current_time + np.mean(pU)
        sigma = pU / (denom_sigma if denom_sigma > 0 else 1.0)
        mu = mu + sigma

        # Line 15-17: Select job j* with minimum mu and append to S
        best_pos = int(np.argmin(mu))
        j_star = U.pop(best_pos)
        S.append(j_star)
        current_time += float(p[j_star])

        # Lines 19-25: Sequential Bean & Hall relocation
        if len(S) > 2:
            candidates = [job for job in S if job != j_star]
            for cand in candidates:
                idx = S.index(cand)
                trial = S[:idx] + S[idx + 1:] + [cand]
                c_orig = np.cumsum(p[S])
                t_orig = int(np.sum(np.maximum(0, c_orig - d[S])))
                c_trial = np.cumsum(p[trial])
                t_trial = int(np.sum(np.maximum(0, c_trial - d[trial])))
                if t_trial < t_orig:
                    S = trial  # Immediate replacement (Line 24)

    return np.array(S, dtype=int)


# ==============================================================================
# 2. Reference Implementation (Interpretation 2: Best-Improvement)
# Bean & Hall (1985) original formulation of relocation
# ==============================================================================
def augmented_mddc_reference_best(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Augmented MDDC - Reference Implementation (Best-Improvement).
    Evaluates all earlier jobs in S \\ {j*}, and relocates the single job
    that produces the minimum tardiness (if strictly better than baseline).
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
        j_star = U.pop(best_pos)
        S.append(j_star)
        current_time += float(p[j_star])

        # Best-improvement relocation
        if len(S) > 2:
            c_base = np.cumsum(p[S])
            best_tard = int(np.sum(np.maximum(0, c_base - d[S])))
            best_S = S
            candidates = [job for job in S if job != j_star]
            for cand in candidates:
                idx = S.index(cand)
                trial = S[:idx] + S[idx + 1:] + [cand]
                c_trial = np.cumsum(p[trial])
                t_trial = int(np.sum(np.maximum(0, c_trial - d[trial])))
                if t_trial < best_tard:
                    best_tard = t_trial
                    best_S = trial
            S = best_S

    return np.array(S, dtype=int)


# Primary alias for reference baseline (defaults to Best-Improvement per Bean & Hall 1985)
augmented_mddc_reference = augmented_mddc_reference_best


# ==============================================================================
# 3. Current Implementation from verification/algorithms.py
# ==============================================================================
def _current_bean_hall_relocation(p: np.ndarray, d: np.ndarray, S: List[int]) -> List[int]:
    L = len(S)
    if L <= 2:
        return S
    comp = np.cumsum(p[S])
    best_tard = int(np.sum(np.maximum(0, comp - d[S])))
    best_S = S
    indices_to_test = range(L - 1) if L <= 80 else range(max(0, L - 25), L - 1)
    for idx in indices_to_test:
        trial = S[:idx] + S[idx + 1:] + [S[idx]]
        trial_comp = np.cumsum(p[trial])
        trial_tard = int(np.sum(np.maximum(0, trial_comp - d[trial])))
        if trial_tard < best_tard:
            best_tard = trial_tard
            best_S = trial
    return best_S


def algo_current_augmented_mddc(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Current implementation from verification/algorithms.py."""
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

        if len(S) > 2:
            S = _current_bean_hall_relocation(p, d, S)

    return np.array(S, dtype=int)


# ==============================================================================
# 4. Adjacent Pairwise Descent (APD)
# Standalone local-search module
# ==============================================================================
def apply_apd(p: np.ndarray, d: np.ndarray, start_seq: np.ndarray) -> np.ndarray:
    """
    Adjacent Pairwise Descent (APD) starting from any feasible schedule start_seq.
    Sweeps adjacent pairs and swaps (seq[i], seq[i+1]) iff tardiness strictly decreases.
    Exact O(1) evaluation per adjacent pair.
    """
    seq = start_seq.copy()
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


def algo_current_augmented_mddc_plus_apd(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Augmented MDDC (Current) + Adjacent Pairwise Descent."""
    start = algo_current_augmented_mddc(p, d)
    return apply_apd(p, d, start)


def algo_reference_augmented_mddc_plus_apd(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Augmented MDDC (Reference Best) + Adjacent Pairwise Descent."""
    start = augmented_mddc_reference_best(p, d)
    return apply_apd(p, d, start)


def algo_sequential_augmented_mddc_plus_apd(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Augmented MDDC (Reference Sequential) + Adjacent Pairwise Descent."""
    start = augmented_mddc_reference_sequential(p, d)
    return apply_apd(p, d, start)
