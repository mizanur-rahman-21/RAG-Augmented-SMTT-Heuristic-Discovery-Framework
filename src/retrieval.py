"""
Retrieval Modules for Controlled RAG Ablation
=============================================
Provides:
1. TheoryRetriever: Indexes domain knowledge from theory_kb/ and retrieves relevant snippets.
2. CodeMemoryRetriever: Stores valid generated candidates, ranks them by validation score,
   and retrieves diverse, high-performing exemplars.
"""

import os
import glob
from typing import List, Dict, Any, Optional
from pathlib import Path


class TheoryRetriever:
    """Retrieves relevant theoretical scheduling excerpts."""

    def __init__(self, kb_dir: str):
        self.kb_dir = kb_dir
        self.documents: List[Dict[str, str]] = []
        self._load_documents()

    def _load_documents(self):
        p = Path(self.kb_dir)
        if not p.exists():
            return
        txt_files = sorted(glob.glob(os.path.join(str(p), "*.txt")))
        for fpath in txt_files:
            fname = os.path.basename(fpath)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            self.documents.append({"name": fname, "content": content})

    def retrieve(self, query: str = "", top_k: int = 2) -> List[Dict[str, str]]:
        if not self.documents:
            return []
        # Return top_k documents
        return self.documents[:top_k]

    def format_for_prompt(self, docs: List[Dict[str, str]]) -> str:
        if not docs:
            return ""
        formatted = ["### Theoretical Principles from Scheduling Literature:"]
        for d in docs:
            formatted.append(f"--- Document: {d['name']} ---")
            formatted.append(d["content"])
        return "\n".join(formatted)


class CodeMemoryRetriever:
    """
    Code-Memory Retriever:
    Maintains memory of valid candidates, metadata, and scores within each run.
    Retrieves elite candidates that achieved lowest validation tardiness.
    """

    def __init__(self):
        self.memory: List[Dict[str, Any]] = []

    def add_candidate(
        self,
        code: str,
        val_score: float,
        train_score: float,
        is_novel: bool,
        iteration: int,
        condition: str,
    ):
        self.memory.append({
            "code": code,
            "val_score": float(val_score),
            "train_score": float(train_score),
            "is_novel": bool(is_novel),
            "iteration": iteration,
            "condition": condition,
        })

    def retrieve(self, top_k: int = 2) -> List[Dict[str, Any]]:
        if not self.memory:
            return []
        # Sort by validation score (ascending: lower tardiness is better)
        sorted_mem = sorted(self.memory, key=lambda x: x["val_score"])
        return sorted_mem[:top_k]

    def format_for_prompt(self, items: List[Dict[str, Any]]) -> str:
        if not items:
            return ""
        formatted = ["### Elite Candidate Programs from Discovery Memory:"]
        for i, item in enumerate(items, 1):
            formatted.append(
                f"--- Candidate {i} (Val Tardiness: {item['val_score']:.2f}, Novel: {item['is_novel']}) ---"
            )
            formatted.append(item["code"].strip())
        return "\n".join(formatted)
