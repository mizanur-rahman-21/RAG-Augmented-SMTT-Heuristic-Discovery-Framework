"""
Standard Baseline Algorithms for SMTT (1 || \\sum T_j)
======================================================
Implements established priority rules and paper challenger heuristics:
- SPT (Shortest Processing Time)
- EDD (Earliest Due Date) - Jackson (1955)
- Panneerselvam Heuristic (Panneerselvam 1993)
- PSK Heuristic (Panneerselvam, Sambandam, Kanet 1993)
- MDD (Modified Due Date) - Baker & Bertrand (1982)
- MDDC (MDD Challenger) - Çetinkaya et al. (2025, Algorithm 2)
- Augmented MDD - Bean & Hall (1985)
- Augmented MDDC - Çetinkaya et al. (2025, Algorithm 3)
- AMDDC + APD - Augmented MDDC with Adjacent Pairwise Descent
"""

from typing import Dict, Callable, List
import numpy as np


def compute_tardiness(p: np.ndarray, d: np.ndarray, seq: np.ndarray) -> int:
    """Computes total tardiness sum_{j} max(0, C_j - d_j)."""
    c = 0
    tard = 0
    for job in seq:
        c += int(p[job])
        diff = c - int(d[job])
        if diff > 0:
            tard += diff
    return int(tard)


def compute_completion_times(p: np.ndarray, seq: np.ndarray) -> np.ndarray:
    """Returns completion times for schedule sequence."""
    return np.cumsum(p[seq])


def validate_permutation(seq: np.ndarray, n: int) -> bool:
    """Validates that seq is a strict permutation of 0..n-1."""
    return len(seq) == n and set(seq) == set(range(n))


def algo_spt(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Shortest Processing Time (SPT) rule."""
    return np.argsort(p)


def algo_edd(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Earliest Due Date (EDD) rule."""
    return np.argsort(d)


def algo_panneerselvam(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Panneerselvam (1993) heuristic."""
    n = len(p)
    current_time = 0
    unscheduled = list(range(n))
    seq: List[int] = []

    while unscheduled:
        best_job = None
        best_score = float("inf")
        best_pos = -1

        for pos, j in enumerate(unscheduled):
            slack = d[j] - (current_time + p[j])
            if slack < 0:
                score = p[j]  # When tardy, prioritize shortest processing time
            else:
                score = d[j] + p[j]  # When slack exists, balance due date and duration

            if score < best_score:
                best_score = score
                best_job = j
                best_pos = pos

        unscheduled.pop(best_pos)
        seq.append(best_job)
        current_time += int(p[best_job])

    return np.array(seq, dtype=int)


def algo_psk(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """PSK Heuristic (Panneerselvam, Sambandam, Kanet 1993)."""
    n = len(p)
    current_time = 0
    unscheduled = list(range(n))
    seq: List[int] = []

    while unscheduled:
        tardy_jobs = [j for j in unscheduled if current_time + p[j] > d[j]]
        if tardy_jobs:
            best_job = min(tardy_jobs, key=lambda j: p[j])
        else:
            best_job = min(unscheduled, key=lambda j: d[j])

        unscheduled.remove(best_job)
        seq.append(best_job)
        current_time += int(p[best_job])

    return np.array(seq, dtype=int)


def algo_mdd(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Modified Due Date (MDD) - Baker & Bertrand (1982)."""
    n = len(p)
    current_time = 0
    unscheduled = list(range(n))
    seq: List[int] = []

    while unscheduled:
        p_u = p[unscheduled]
        d_u = d[unscheduled]
        mdd_vals = np.maximum(current_time + p_u, d_u)

        best_pos = 0
        best_val = mdd_vals[0]
        best_p = p_u[0]
        for idx in range(1, len(unscheduled)):
            val = mdd_vals[idx]
            if val < best_val or (val == best_val and p_u[idx] < best_p):
                best_val = val
                best_p = p_u[idx]
                best_pos = idx

        job = unscheduled.pop(best_pos)
        seq.append(job)
        current_time += int(p[job])

    return np.array(seq, dtype=int)


def algo_mddc(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """MDD Challenger (MDDC) - Çetinkaya et al. (2025, Algorithm 2)."""
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

    return np.array(S, dtype=int)


def _apply_bean_hall_relocation(p: np.ndarray, d: np.ndarray, S: List[int]) -> List[int]:
    """Bean & Hall (1985) relocation / Çetinkaya et al. (2025, Algorithm 3 lines 19-25)."""
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


def algo_augmented_mdd(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Augmented MDD - Bean & Hall (1985)."""
    n = len(p)
    current_time = 0
    mask = np.ones(n, dtype=bool)
    S: List[int] = []

    for _ in range(n):
        avail = np.where(mask)[0]
        mdd_vals = np.maximum(current_time + p[avail], d[avail])
        best_job = int(avail[np.argmin(mdd_vals)])
        S.append(best_job)
        mask[best_job] = False
        current_time += int(p[best_job])

        if len(S) > 2:
            S = _apply_bean_hall_relocation(p, d, S)

    return np.array(S, dtype=int)


def algo_augmented_mddc(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Augmented MDDC - Çetinkaya et al. (2025, Algorithm 3)."""
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
            S = _apply_bean_hall_relocation(p, d, S)

    return np.array(S, dtype=int)


def algo_radc_apd(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Augmented MDDC + Adjacent Pairwise Descent (AMDDC + APD).
    Phase 1: Augmented MDDC construction.
    Phase 2: Adjacent Pairwise Interchange (API) sweeps with O(1) swap evaluation.
    Guarantees: tardiness(AMDDC + APD) <= tardiness(Augmented MDDC).
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


# Canonical alias conforming to final research wording (AMDDC + APD)
algo_amddc_apd = algo_radc_apd
compute_total_tardiness = compute_tardiness


ALL_BASELINES: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "SPT": algo_spt,
    "EDD": algo_edd,
    "Panneerselvam": algo_panneerselvam,
    "PSK": algo_psk,
    "MDD": algo_mdd,
    "MDDC": algo_mddc,
    "Augmented MDD": algo_augmented_mdd,
    "Augmented MDDC": algo_augmented_mddc,
    "AMDDC + APD": algo_amddc_apd,
    "AMDDC + APD (Frozen)": algo_radc_apd,
}
