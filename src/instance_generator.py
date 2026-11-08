"""
Instance Generator for Single-Machine Total Tardiness (1 || \\sum T_j).
======================================================================
Protocol: Potts & Van Wassenhove (1982)
- Processing times: p_j ~ U[1, 100]
- Total processing time: P = \\sum_{j=1}^n p_j
- Relative Due Date Range: RDD in {0.2, 0.4, 0.6, 0.8, 1.0}
- Tardiness Factor: TF in {0.2, 0.4, 0.6, 0.8}
- Due dates: d_j ~ U[max(0, floor(P*(1 - TF - RDD/2))), ceil(P*(1 - TF + RDD/2))]
- Standard class grid: 20 classes (5 RDD x 4 TF)
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os
import numpy as np


@dataclass
class SMTTInstance:
    instance_id: str
    n_jobs: int
    p: np.ndarray
    d: np.ndarray
    rdd: float
    tf: float
    distribution: str = "uniform"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "n_jobs": int(self.n_jobs),
            "p": self.p.tolist(),
            "d": self.d.tolist(),
            "rdd": float(self.rdd),
            "tf": float(self.tf),
            "distribution": self.distribution,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SMTTInstance":
        return cls(
            instance_id=data["instance_id"],
            n_jobs=data["n_jobs"],
            p=np.array(data["p"], dtype=int),
            d=np.array(data["d"], dtype=int),
            rdd=data["rdd"],
            tf=data["tf"],
            distribution=data.get("distribution", "uniform"),
        )


def generate_single_instance(
    n_jobs: int,
    rdd: float,
    tf: float,
    rng: np.random.Generator,
    instance_id: str = "inst_0",
) -> SMTTInstance:
    p = rng.integers(1, 101, size=n_jobs)
    P = int(np.sum(p))
    low_d = int(np.floor(P * (1.0 - tf - rdd / 2.0)))
    high_d = int(np.ceil(P * (1.0 - tf + rdd / 2.0)))

    if low_d >= high_d:
        high_d = low_d + 1

    d = rng.integers(low_d, high_d + 1, size=n_jobs)
    d = np.maximum(0, d)

    return SMTTInstance(
        instance_id=instance_id,
        n_jobs=n_jobs,
        p=p,
        d=d,
        rdd=rdd,
        tf=tf,
        distribution="uniform",
    )


def generate_benchmark_suite(
    n_jobs: int = 20,
    instances_per_class: int = 40,
    seed: int = 30003,
) -> List[SMTTInstance]:
    """
    Generates standard Potts & Van Wassenhove benchmark suite.
    5 RDD x 4 TF x instances_per_class.
    """
    rng = np.random.default_rng(seed)
    rdd_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    tf_values = [0.2, 0.4, 0.6, 0.8]

    instances = []
    idx = 0
    for rdd in rdd_values:
        for tf in tf_values:
            for rep in range(instances_per_class):
                iid = f"inst_rdd{rdd:.1f}_tf{tf:.1f}_{rep:02d}_{idx:04d}"
                inst = generate_single_instance(
                    n_jobs=n_jobs,
                    rdd=rdd,
                    tf=tf,
                    rng=rng,
                    instance_id=iid,
                )
                instances.append(inst)
                idx += 1

    return instances


def save_dataset_json(instances: List[SMTTInstance], filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    data = [inst.to_dict() for inst in instances]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_dataset_json(filepath: str) -> List[SMTTInstance]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [SMTTInstance.from_dict(item) for item in data]
