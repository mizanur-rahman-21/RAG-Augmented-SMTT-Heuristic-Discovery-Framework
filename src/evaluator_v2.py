"""
Scientific Candidate Evaluator & Multi-Tier Novelty Audit (v2)
=============================================================
Provides rigorous, fine-grained scientific evaluation:

Multi-Tier Status Verification:
1. Syntactically Valid: Code compiles safely under AST whitelist without dangerous constructs.
2. Successfully Evaluated: Produces strict permutations on 100% of benchmark instances within timeout.
3. Structural Novelty Classification:
   - CLONE (>= 95% permutation identity with any baseline).
   - KNOWN_LOCAL_SEARCH_COMPOSITION (incorporates adjacent pairwise swaps or Bean-Hall relocation).
   - GENUINE_NEW_PRIORITY_FUNCTION (novel dynamic mathematical index).
4. Continuous Literature Novelty:
   - Exact Kendall tau rank correlation across all 8 literature baselines.
5. Performance Tiering:
   - Outperforms AMDDC+APD (State-of-the-Art).
   - Outperforms Augmented MDDC.
   - Outperforms MDD.
   - Sub-MDD.
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


SAFE_BUILTINS_V2 = {
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
    """AST safety screening."""
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
                if node.func.id in ("eval", "exec", "open", "compile"):
                    return False, f"Disallowed function call: {node.func.id}"

    return True, "Code passed AST scan."


def safe_execute_candidate(code_str: str, entry_point: str = "custom_heuristic") -> Tuple[bool, Optional[Callable], str]:
    """Compiles candidate code safely with injected loop safeguards."""
    is_safe, msg = scan_code_ast(code_str)
    if not is_safe:
        return False, None, msg

    safe_globals = {
        "__builtins__": SAFE_BUILTINS_V2,
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


def evaluate_on_dataset_v2(
    fn: Callable,
    instances: list,
    timeout_s: float = 2.0,
) -> Tuple[bool, float, float, str]:
    """
    Evaluates candidate on an instance collection.
    Returns: (is_success, mean_tardiness, mean_time_ms, status_message)
    """
    tards = []
    times_ms = []

    for idx, inst in enumerate(instances):
        p, d, n = inst.p, inst.d, inst.n_jobs
        t0 = time.perf_counter()
        try:
            seq = fn(p, d)
        except TimeoutError as te:
            return False, 1e9, 0.0, f"Timeout/LoopGuard: {te}"
        except Exception as e:
            return False, 1e9, 0.0, f"RuntimeError on instance {idx}: {e}"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if elapsed_ms > timeout_s * 1000.0:
            return False, 1e9, 0.0, f"Per-instance timeout exceeded: {elapsed_ms:.1f}ms > {timeout_s*1000:.1f}ms"

        # Check permutation validity
        if not (len(seq) == n and set(seq) == set(range(n))):
            return False, 1e9, 0.0, f"Invalid permutation: length={len(seq)} (expected {n}), unique={len(set(seq))}"

        c = np.cumsum(p[seq])
        tard = int(np.sum(np.maximum(0, c - d[seq])))
        tards.append(tard)
        times_ms.append(elapsed_ms)

    return True, float(np.mean(tards)), float(np.mean(times_ms)), "EVALUATION_SUCCESS"


def audit_candidate_scientific_novelty(
    code_str: str,
    fn: Callable,
    audit_instances: list,
    baselines: Dict[str, Callable],
    anchor_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Comprehensive multi-tier novelty audit:
    - Compares schedule permutations across baselines.
    - Measures Kendall tau rank correlation against each baseline.
    - Inspects code structure for classical local search signatures.
    - Computes schedule fingerprint on the anchor instance.
    """
    matches = {b_name: 0 for b_name in baselines}
    kendalls = {b_name: [] for b_name in baselines}
    total = len(audit_instances)

    for inst in audit_instances:
        p, d = inst.p, inst.d
        try:
            cand_seq = fn(p, d)
        except Exception as e:
            return {
                "classification": "AUDIT_EXECUTION_ERROR",
                "closest_baseline": "None",
                "exact_match_pct": 0.0,
                "mean_kendall_tau": 0.0,
                "literature_novelty_distance": 0.0,
                "is_novel": False,
                "fingerprint": None,
                "reason": f"Audit execution failed: {e}",
            }

        for b_name, b_fn in baselines.items():
            base_seq = b_fn(p, d)
            if np.array_equal(cand_seq, base_seq):
                matches[b_name] += 1
            tau, _ = stats.kendalltau(cand_seq, base_seq)
            kendalls[b_name].append(1.0 if np.isnan(tau) else tau)

    # Closest baseline by permutation match rate
    closest_b = max(matches, key=lambda k: matches[k])
    max_match_pct = (matches[closest_b] / total) * 100.0 if total > 0 else 0.0
    mean_tau_closest = float(np.mean(kendalls[closest_b])) if kendalls[closest_b] else 0.0

    # Max correlation across all baselines
    overall_max_tau = max([float(np.mean(kendalls[b])) for b in baselines]) if baselines else 1.0
    lit_novelty_dist = max(0.0, 1.0 - overall_max_tau)

    # Check for local search patterns (APD / Bean-Hall relocation)
    has_relocation = any(kw in code_str for kw in ["trial", "relocation", "S[:idx]", "c_tr", "insert_pos"])
    has_pairwise_swap = any(kw in code_str for kw in ["tard_swap", "c1_swap", "API", "pairwise", "adjacent_pair", "swap"])

    # Anchor fingerprint for population diversity
    fingerprint = None
    if anchor_instance is not None:
        try:
            fp_seq = fn(anchor_instance.p, anchor_instance.d)
            fingerprint = [int(x) for x in fp_seq]
        except Exception:
            pass

    if max_match_pct >= 95.0:
        classification = "CLONE"
        is_novel = False
        reason = f"Disguised clone of {closest_b} ({max_match_pct:.1f}% permutation match, Kendall tau = {mean_tau_closest:.4f})"
    elif has_relocation or has_pairwise_swap:
        classification = "KNOWN_LOCAL_SEARCH_COMPOSITION"
        is_novel = False
        desc = "Bean-Hall relocation" if has_relocation else "Adjacent Pairwise Interchange"
        reason = f"Employs published local search ({desc}); not a novel dispatching priority structure."
    else:
        classification = "GENUINE_NEW_PRIORITY_FUNCTION"
        is_novel = True
        reason = f"Statistically distinct priority structure (Literature novelty distance: {lit_novelty_dist:.4f})."

    return {
        "classification": classification,
        "closest_baseline": closest_b,
        "exact_match_pct": float(max_match_pct),
        "mean_kendall_tau": float(mean_tau_closest),
        "literature_novelty_distance": float(lit_novelty_dist),
        "is_novel": is_novel,
        "fingerprint": fingerprint,
        "reason": reason,
    }
