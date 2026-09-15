"""Mechanical checks against train/test contamination.

These run as tests *and* as part of every experiment report, because "we used
different seeds" is a claim that should be verified rather than asserted.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence

import numpy as np

from ..environment.runner import Trajectory

__all__ = ["audit_split_disjointness", "audit_observation_overlap",
           "trajectory_fingerprints"]


def audit_split_disjointness(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Assert every pair of splits uses disjoint episode seeds."""
    seeds = {k: set(v["seeds"]) for k, v in manifest["splits"].items()}
    collisions: Dict[str, List[int]] = {}
    names = sorted(seeds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = seeds[a] & seeds[b]
            if inter:
                collisions[f"{a}|{b}"] = sorted(inter)
    return {"ok": not collisions, "collisions": collisions,
            "n_seeds": {k: len(v) for k, v in seeds.items()}}


def trajectory_fingerprints(trajs: Sequence[Trajectory], digits: int = 6) -> List[str]:
    """Hash of each trajectory's observation matrix (rounded)."""
    out = []
    for tr in trajs:
        h = hashlib.blake2b(digest_size=12)
        h.update(np.round(tr.obs.astype(np.float64), digits).tobytes())
        out.append(h.hexdigest())
    return out


def audit_observation_overlap(a: Sequence[Trajectory], b: Sequence[Trajectory],
                              digits: int = 4) -> Dict[str, Any]:
    """Detect *exactly duplicated states* across two splits.

    Identical rows across splits would mean a model could memorise a test state
    from training.  We hash rounded observation rows and report the overlap rate.
    """
    def rows(ts: Sequence[Trajectory]) -> set:
        acc = set()
        for tr in ts:
            r = np.round(tr.obs.astype(np.float64), digits)
            acc.update(hashlib.blake2b(x.tobytes(), digest_size=10).hexdigest()
                       for x in r)
        return acc

    ra, rb = rows(a), rows(b)
    inter = ra & rb
    return {"n_a": len(ra), "n_b": len(rb), "n_shared": len(inter),
            "overlap_frac_b": len(inter) / max(len(rb), 1)}
