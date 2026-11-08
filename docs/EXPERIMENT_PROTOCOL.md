# Experimental Protocol: RAG-Augmented SMTT Heuristic Discovery

## 1. Problem Formulation

The Single-Machine Total Tardiness Problem ($1||\sum T_j$) requires sequencing $n$ jobs on a single continuously available machine to minimize the sum of job tardiness values:
$$\min_{\pi \in \Pi_n} \sum_{j=1}^n \max(0, C_j(\pi) - d_j)$$
where:
- $p_j > 0$ is the processing time of job $j$;
- $d_j \ge 0$ is the due date of job $j$;
- $C_j(\pi) = \sum_{k=1}^{\pi^{-1}(j)} p_{\pi(k)}$ is the completion time of job $j$ under permutation $\pi$.

The problem is strongly NP-hard (Du and Leung, 1990).

---

## 2. Experimental Ablation Design

We evaluate whether Retrieval-Augmented Generation (RAG) accelerates or enhances the automated discovery of scheduling heuristics using LLMs.

Four conditions are evaluated under identical decoding settings, compute budget, and instance distribution:

| Condition | Prompt Inputs | Knowledge Injected |
| :--- | :--- | :--- |
| **1. No-RAG** | Task spec + Seed code | None |
| **2. Theory-RAG** | Task spec + Seed code + Theory excerpts | Domain theorems (Emmons 1969, Baker & Bertrand 1982, Potts & Van Wassenhove 1982, Bean & Hall 1985) |
| **3. Code-Memory RAG** | Task spec + Seed code + Discovered elites | Past discovered heuristics with high validation scores |
| **4. Dual-RAG** | Task spec + Seed code + Theory + Memory | Both theoretical literature and memory of discovered elites |

---

## 3. Data Split & Isolation Protocol

To strictly eliminate data leakage and memorization:
1. **Training Set (Seed 10001)**: 800 instances ($n=20$) generated via Potts & Van Wassenhove (1982). Used exclusively for candidate scoring and in-context exemplar evaluation.
2. **Validation Set (Seed 20002)**: 800 instances ($n=20$). Used strictly for model selection (identifying the single best candidate per condition/seed).
3. **Audit Set (Seed 99999)**: 40 instances ($n=20$). Used for rapid Kendall's $\tau$ correlation and permutation equivalence screening.
4. **Held-Out Test Benchmark (Seeds 30003, 31001, 31002, 31003)**: Total 3,200 instances ($n=20$). **LOCKED**. Evaluated exactly once on the final selected candidates after all discovery has concluded.

---

## 4. Candidate Validation and Sandboxing

Every candidate returned by the LLM undergoes rigorous automated screening:
1. **AST Safety Scanner**: Rejects all code containing imports (`import`, `from ... import`), file access, networking, runtime code execution (`eval`, `exec`), exception suppressing (`try`/`except`), or global variable mutations.
2. **Compilation**: Executed within an isolated dictionary environment providing only `np` and standard mathematical built-ins.
3. **Permutation Validator**: Confirms the returned sequence is a strict bijection on $\{0, 1, \dots, n-1\}$.
4. **Execution Timeout**: Any candidate exceeding the wall-clock execution threshold (2.0s for pilot, 3.0s for full) is immediately terminated and disqualified.

---

## 5. Novelty Classification Taxonomy

To prevent disguised copies of published heuristics from being reported as novel scientific discoveries, each candidate is classified against known baselines (MDD, MDDC, Augmented MDDC, AMDDC + APD):

- **CLONE**: Matches $\ge 95\%$ of permutations of any known baseline.
- **KNOWN_LOCAL_SEARCH_COMPOSITION**: Employs classical adjacent pairwise interchange (Emmons 1969; Baker 1974) or Bean & Hall (1985) relocation.
- **GENUINE_NEW_STRUCTURE**: Novel dynamic priority functions statistically distinct from existing literature.

---

## 6. Cryptographic Provenance

Every inference is hashed using SHA-256 for both prompt and response:
- Logs are written in JSONL format to `logs/candidates_provenance.jsonl`.
- Ensures zero mock, cached, or synthetic outputs are masqueraded as genuine LLM discoveries.
