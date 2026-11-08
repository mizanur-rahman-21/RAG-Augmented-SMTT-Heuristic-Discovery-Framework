"""
Exact Dynamic Programming Solver for Single-Machine Total Tardiness (1 || \\sum T_j).
====================================================================================

Implements backward dynamic programming over subsets (Held & Karp 1962; Potts & Van Wassenhove 1982):
  V(J) = min_{j in J} [ V(J \\ {j}) + max(0, C(J) - d_j) ]
where C(J) = \\sum_{k in J} p_k.

Utilizes bitmask dynamic programming accelerated with Numba JIT (when available),
with complete schedule reconstruction via backtracking.
Includes an exact brute-force permutation search solver for verification on n <= 8.
"""

from typing import Tuple, List
import itertools
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    @njit
    def _solve_dp_numba(p_arr: np.ndarray, d_arr: np.ndarray) -> Tuple[int, np.ndarray]:
        n = len(p_arr)
        size = 1 << n
        dp = np.full(size, 10**18, dtype=np.int64)
        parent = np.full(size, -1, dtype=np.int32)
        dp[0] = 0
        comp = np.zeros(size, dtype=np.int64)

        for mask in range(1, size):
            lsb = mask & -mask
            prev = mask ^ lsb
            j = 0
            temp = lsb >> 1
            while temp > 0:
                j += 1
                temp >>= 1
            comp[mask] = comp[prev] + p_arr[j]

        # Bottom-up transition
        for mask in range(1, size):
            C = comp[mask]
            best_val = 10**18
            best_j = -1
            for j in range(n):
                if (mask >> j) & 1:
                    prev_mask = mask ^ (1 << j)
                    tard = C - d_arr[j]
                    if tard < 0:
                        tard = 0
                    val = dp[prev_mask] + tard
                    if val < best_val:
                        best_val = val
                        best_j = j
            dp[mask] = best_val
            parent[mask] = best_j

        return dp[size - 1], parent


def _solve_dp_python(p_arr: np.ndarray, d_arr: np.ndarray) -> Tuple[int, np.ndarray]:
    n = len(p_arr)
    size = 1 << n
    dp = np.full(size, 10**18, dtype=np.int64)
    parent = np.full(size, -1, dtype=np.int32)
    dp[0] = 0
    comp = np.zeros(size, dtype=np.int64)

    for mask in range(1, size):
        lsb = mask & -mask
        prev = mask ^ lsb
        j = lsb.bit_length() - 1
        comp[mask] = comp[prev] + p_arr[j]

    for mask in range(1, size):
        C = comp[mask]
        best_val = 10**18
        best_j = -1
        temp = mask
        while temp > 0:
            lsb = temp & -temp
            j = lsb.bit_length() - 1
            prev_mask = mask ^ lsb
            tard = max(0, C - d_arr[j])
            val = dp[prev_mask] + tard
            if val < best_val:
                best_val = val
                best_j = j
            temp ^= lsb
        dp[mask] = best_val
        parent[mask] = best_j

    return int(dp[size - 1]), parent


def solve_smtt_exact(p: np.ndarray, d: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Computes exact optimal total tardiness and optimal sequence for an SMTT instance.
    Returns:
        (optimal_tardiness, optimal_sequence)
    """
    p_int = np.ascontiguousarray(p, dtype=np.int64)
    d_int = np.ascontiguousarray(d, dtype=np.int64)
    n = len(p_int)

    if n == 0:
        return 0, np.array([], dtype=int)
    if n == 1:
        tard = max(0, int(p_int[0] - d_int[0]))
        return tard, np.array([0], dtype=int)

    if HAS_NUMBA:
        opt_val, parent = _solve_dp_numba(p_int, d_int)
    else:
        opt_val, parent = _solve_dp_python(p_int, d_int)

    # Backtrack schedule
    seq = np.empty(n, dtype=int)
    curr_mask = (1 << n) - 1
    for pos in range(n - 1, -1, -1):
        job = parent[curr_mask]
        seq[pos] = job
        curr_mask ^= (1 << job)

    return int(opt_val), seq


def solve_smtt_brute_force(p: np.ndarray, d: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Solves SMTT to exact optimality by evaluating all n! permutations.
    For small n (n <= 8) verification.
    """
    n = len(p)
    best_tard = 10**18
    best_perm = tuple(range(n))

    for perm in itertools.permutations(range(n)):
        c = 0
        total_tard = 0
        for j in perm:
            c += p[j]
            diff = c - d[j]
            if diff > 0:
                total_tard += diff
        if total_tard < best_tard:
            best_tard = total_tard
            best_perm = perm

    return int(best_tard), np.array(best_perm, dtype=int)
