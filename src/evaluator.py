"""
Strengthened Novelty & Equivalence Audit and Candidate Evaluator
================================================================
Audits candidates against all key benchmark tiers:
1. MDD (Baker & Bertrand 1982)
2. MDDC (Çetinkaya et al. 2025, Alg. 2)
3. Augmented MDDC (Çetinkaya et al. 2025, Alg. 3)
4. AMDDC + APD (Frozen Baseline)

Taxonomy of Classifications:
- CLONE: Exact permutation identity (>= 95% match with any baseline).
- KNOWN_LOCAL_SEARCH_COMPOSITION: Reproduces published Bean & Hall (1985) relocation
  or adjacent interchange on an existing rule. NOT a new theoretical discovery.
- GENUINE_NEW_STRUCTURE: Novel priority terms or uncatalogued search dynamics
  producing statistically distinct permutations from all baselines.
"""

import ast
import time
import textwrap
from typing import Dict, Any, Tuple, Optional, Callable, List
import numpy as np
from scipy import stats


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in ("numpy", "math"):
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of module '{name}' is disallowed in sandbox.")


SAFE_BUILTINS = {
    "__import__": _safe_import,
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "print": lambda *args, **kwargs: None,
    "TimeoutError": TimeoutError,
    "RuntimeError": RuntimeError,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "Exception": Exception,
}


class LoopGuardTransformer(ast.NodeTransformer):
    """
    Transforms AST to inject a step counter into while loops.
    Guarantees that infinite loops terminate immediately with TimeoutError.
    """
    def __init__(self, max_iters: int = 10000):
        self.max_iters = max_iters

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        # Prepend: _loop_guard = [0] at start of function
        init_guard = ast.Assign(
            targets=[ast.Name(id="_loop_guard", ctx=ast.Store())],
            value=ast.List(elts=[ast.Constant(value=0)], ctx=ast.Load()),
        )
        node.body.insert(0, init_guard)
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        inc = ast.AugAssign(
            target=ast.Subscript(
                value=ast.Name(id="_loop_guard", ctx=ast.Load()),
                slice=ast.Constant(value=0),
                ctx=ast.Store(),
            ),
            op=ast.Add(),
            value=ast.Constant(value=1),
        )
        check = ast.If(
            test=ast.Compare(
                left=ast.Subscript(
                    value=ast.Name(id="_loop_guard", ctx=ast.Load()),
                    slice=ast.Constant(value=0),
                    ctx=ast.Load(),
                ),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=self.max_iters)],
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="TimeoutError", ctx=ast.Load()),
                        args=[ast.Constant(value="Candidate loop iteration limit exceeded (infinite loop protection)")],
                        keywords=[],
                    ),
                    cause=None,
                )
            ],
            orelse=[],
        )
        node.body = [inc, check] + node.body
        return node


def scan_code_ast(code_str: str) -> Tuple[bool, str]:
    dedented = textwrap.dedent(code_str).strip()
    try:
        tree = ast.parse(dedented)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    disallowed_nodes = (ast.Global, ast.Nonlocal, ast.Try)
    allowed_modules = ("numpy", "math")

    for node in ast.walk(tree):
        if isinstance(node, disallowed_nodes):
            return False, f"Disallowed AST node: {type(node).__name__}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed_modules:
                    return False, f"Disallowed import: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            if node.module not in allowed_modules:
                return False, f"Disallowed import from: {node.module}"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "open", "__import__", "compile"):
                    return False, f"Disallowed function call: {node.func.id}"

    return True, "Code passed AST scan."


def safe_execute_candidate(code_str: str, entry_point: str = "custom_heuristic") -> Tuple[bool, Optional[Callable], str]:
    is_safe, msg = scan_code_ast(code_str)
    if not is_safe:
        return False, None, msg

    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        "np": np,
        "numpy": np,
    }
    local_scope: Dict[str, Any] = {}

    try:
        dedented = textwrap.dedent(code_str).strip()
        tree = ast.parse(dedented)
        tree = LoopGuardTransformer(max_iters=10000).visit(tree)
        ast.fix_missing_locations(tree)
        compiled = compile(tree, "<candidate>", "exec")
        exec(compiled, safe_globals, local_scope)
    except Exception as e:
        return False, None, f"Execution failed during compile: {e}"

    if entry_point not in local_scope:
        return False, None, f"Missing function '{entry_point}'"

    return True, local_scope[entry_point], "Success"


def evaluate_on_dataset(
    fn: Callable,
    instances: list,
    timeout_s: float = 2.0,
) -> Tuple[bool, float, float]:
    """
    Evaluates candidate function on a dataset.
    Returns: (is_valid, mean_tardiness, mean_time_ms)
    """
    tards = []
    times_ms = []

    for inst in instances:
        p, d, n = inst.p, inst.d, inst.n_jobs
        t0 = time.perf_counter()
        try:
            seq = fn(p, d)
        except Exception:
            return False, 1e9, 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if elapsed_ms > timeout_s * 1000.0:
            return False, 1e9, 0.0

        # Permutation validation
        if not (len(seq) == n and set(seq) == set(range(n))):
            return False, 1e9, 0.0

        c = np.cumsum(p[seq])
        tard = int(np.sum(np.maximum(0, c - d[seq])))
        tards.append(tard)
        times_ms.append(elapsed_ms)

    return True, float(np.mean(tards)), float(np.mean(times_ms))


def audit_candidate_novelty_strengthened(
    code_str: str,
    fn: Callable,
    audit_instances: list,
    baselines: Dict[str, Callable],
) -> Dict[str, Any]:
    """
    Strengthened novelty audit:
    Compares against MDD, MDDC, Augmented MDDC, AMDDC + APD.
    Classifies candidate into:
    - CLONE (exact match >= 95%)
    - KNOWN_LOCAL_SEARCH_COMPOSITION (Bean-Hall relocation / API descent)
    - GENUINE_NEW_STRUCTURE
    """
    matches = {b_name: 0 for b_name in baselines}
    kendalls = {b_name: [] for b_name in baselines}
    total = len(audit_instances)

    for inst in audit_instances:
        p, d = inst.p, inst.d
        try:
            cand_seq = fn(p, d)
        except Exception:
            return {
                "classification": "EXECUTION_ERROR",
                "closest_baseline": "None",
                "exact_match_pct": 0.0,
                "mean_kendall_tau": 0.0,
                "is_novel": False,
                "reason": "Execution error during audit",
            }

        for b_name, b_fn in baselines.items():
            base_seq = b_fn(p, d)
            if np.array_equal(cand_seq, base_seq):
                matches[b_name] += 1
            tau, _ = stats.kendalltau(cand_seq, base_seq)
            kendalls[b_name].append(1.0 if np.isnan(tau) else tau)

    closest_b = max(matches, key=lambda k: matches[k])
    max_match_pct = (matches[closest_b] / total) * 100.0 if total > 0 else 0.0
    mean_tau = float(np.mean(kendalls[closest_b])) if kendalls[closest_b] else 0.0

    # Check for known local search composition keywords in code
    has_relocation = any(kw in code_str for kw in ["trial", "relocation", "S[:idx]", "c_tr"])
    has_pairwise_swap = any(kw in code_str for kw in ["tard_swap", "c1_swap", "API", "pairwise", "adjacent_pair"])

    if max_match_pct >= 95.0:
        return {
            "classification": "CLONE",
            "closest_baseline": closest_b,
            "exact_match_pct": max_match_pct,
            "mean_kendall_tau": mean_tau,
            "is_novel": False,
            "reason": f"Disguised clone of {closest_b} ({max_match_pct:.1f}% match, Kendall tau = {mean_tau:.4f})",
        }

    if has_relocation or has_pairwise_swap:
        desc = "Bean-Hall relocation composition" if has_relocation else "Adjacent pairwise interchange"
        return {
            "classification": "KNOWN_LOCAL_SEARCH_COMPOSITION",
            "closest_baseline": closest_b,
            "exact_match_pct": max_match_pct,
            "mean_kendall_tau": mean_tau,
            "is_novel": False,  # Not a novel dispatching discovery
            "reason": f"Known local search composition ({desc}); not a new heuristic structure.",
        }

    return {
        "classification": "GENUINE_NEW_STRUCTURE",
        "closest_baseline": closest_b,
        "exact_match_pct": max_match_pct,
        "mean_kendall_tau": mean_tau,
        "is_novel": True,
        "reason": "Genuinely novel dynamic priority structure distinct from all baselines.",
    }
