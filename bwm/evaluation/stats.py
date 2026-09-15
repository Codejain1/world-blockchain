"""Paired significance testing for control comparisons.

With a handful of episodes per world, differences in mean return are easy to
over-read.  Every policy is run on the *same* episode seeds, so the comparisons
are naturally paired -- which is much more powerful than comparing two
independent means, and is the only honest way to say "A beat B" at this sample
size.

Two tests are reported:

* a **paired bootstrap** confidence interval on the mean difference, and
* an **exact paired permutation (sign-flip) test**, which makes no distributional
  assumption and is exact for small n.

Both are seeded, so the p-values are reproducible.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["paired_bootstrap", "paired_permutation_test", "compare_to_reference",
           "significance_table"]


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000,
                     alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Bootstrap CI for ``mean(a - b)`` over paired observations."""
    x = np.asarray(a, float) - np.asarray(b, float)
    n = x.size
    if n == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    return {"diff": float(x.mean()),
            "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2)),
            "n": int(n)}


def paired_permutation_test(a: Sequence[float], b: Sequence[float],
                            n_perm: int = 20_000, seed: int = 0) -> float:
    """Two-sided p-value for ``mean(a - b) == 0`` under sign-flip symmetry.

    Enumerates all ``2^n`` sign assignments when ``n <= 20`` (exact), otherwise
    samples.
    """
    x = np.asarray(a, float) - np.asarray(b, float)
    n = x.size
    if n == 0:
        return float("nan")
    obs = abs(float(x.mean()))
    if n <= 20:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        means = np.abs((signs * x).mean(axis=1))
        return float((means >= obs - 1e-12).mean())
    rng = np.random.Generator(np.random.PCG64(seed))
    signs = rng.choice([1.0, -1.0], size=(n_perm, n))
    means = np.abs((signs * x).mean(axis=1))
    return float((means >= obs - 1e-12).mean())


def _paired_series(per_policy: Dict[str, Dict[str, List[Any]]], key: str
                   ) -> Tuple[List[Tuple[str, int]], Dict[str, Dict[Tuple[str, int], float]]]:
    """Collect per-episode values keyed by (scenario, seed) for each policy."""
    values: Dict[str, Dict[Tuple[str, int], float]] = {}
    for policy, per_sc in per_policy.items():
        acc: Dict[Tuple[str, int], float] = {}
        for sc, results in per_sc.items():
            for r in results:
                acc[(sc, int(r.seed))] = float(r.summary()[key])
        values[policy] = acc
    common = sorted(set.intersection(*[set(v) for v in values.values()])) \
        if values else []
    return common, values


def compare_to_reference(per_policy: Dict[str, Dict[str, List[Any]]],
                         reference: str, key: str = "total_return",
                         seed: int = 0) -> List[Dict[str, Any]]:
    """Paired comparison of every policy against ``reference``, episode by episode."""
    common, values = _paired_series(per_policy, key)
    if reference not in values or not common:
        return []
    ref = [values[reference][k] for k in common]
    rows: List[Dict[str, Any]] = []
    for policy, v in values.items():
        if policy == reference:
            continue
        cur = [v[k] for k in common]
        bs = paired_bootstrap(cur, ref, seed=seed)
        p = paired_permutation_test(cur, ref, seed=seed)
        rows.append({
            "policy": policy, "vs": reference, "metric": key,
            "mean_diff": bs["diff"], "ci_lo": bs["lo"], "ci_hi": bs["hi"],
            "p_value": p, "n_pairs": bs["n"],
            "significant_05": bool(p < 0.05),
        })
    return sorted(rows, key=lambda r: -(r["mean_diff"] or 0))


def significance_table(per_policy: Dict[str, Dict[str, List[Any]]],
                       pairs: Sequence[Tuple[str, str]],
                       key: str = "total_return", seed: int = 0
                       ) -> List[Dict[str, Any]]:
    """Targeted head-to-head comparisons (the hypothesis's own claims)."""
    common, values = _paired_series(per_policy, key)
    rows: List[Dict[str, Any]] = []
    for a, b in pairs:
        if a not in values or b not in values or not common:
            continue
        xa = [values[a][k] for k in common]
        xb = [values[b][k] for k in common]
        bs = paired_bootstrap(xa, xb, seed=seed)
        rows.append({
            "comparison": f"{a} - {b}", "mean_diff": bs["diff"],
            "ci_lo": bs["lo"], "ci_hi": bs["hi"],
            "p_value": paired_permutation_test(xa, xb, seed=seed),
            "n_pairs": bs["n"],
        })
    return rows
