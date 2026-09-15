"""Trajectory dataset generation with leakage-free splits.

Split discipline
----------------
Splits are defined by **disjoint episode-seed ranges**, never by slicing inside
an episode.  Because a world's entire future is a deterministic function of its
seed, two episodes from different ranges share no state, no noise and no agent
draws.  Slicing within an episode (the usual shortcut) would leak: the same
market regime, the same balance sheets and the same price path would appear on
both sides of the split.

======================  =============================
split                   seed range
======================  =============================
``train``               [0, 100_000)
``val``                 [100_000, 200_000)
``test_iid``            [200_000, 300_000)
``ood_<scenario>``      [300_000 + 10_000*i, ...)
======================  =============================

A dataset manifest records every seed used, so any claim about contamination can
be checked mechanically (see :func:`bwm.data.audit.audit_split_disjointness`).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..environment.runner import BehaviorPolicy, EpisodeRunner, Trajectory
from ..environment.scenarios import SCENARIO_KIND, make_scenario, sample_train_config, scenario_names

__all__ = ["SPLIT_SEED_BASE", "GenSpec", "generate_split", "generate_dataset",
           "load_split", "load_manifest"]

SPLIT_SEED_BASE: Dict[str, int] = {
    "train": 0,
    "val": 100_000,
    "test_iid": 200_000,
}
_OOD_BASE = 300_000


def ood_seed_base(scenario: str) -> int:
    names = scenario_names()
    return _OOD_BASE + 10_000 * names.index(scenario)


@dataclass
class GenSpec:
    """How much data to generate and from which worlds."""

    n_train: int = 120
    n_val: int = 24
    n_test: int = 32
    n_ood: int = 16
    episode_length: int = 256
    n_agents: int = 120
    epsilon: float = 0.35            # exploration in the focal agent's behaviour policy
    scenarios: List[str] = field(default_factory=lambda: [
        s for s in scenario_names() if s != "iid"])
    record_graph: bool = True
    out_dir: str = "datasets/main"
    n_workers: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
def _episode(args: Tuple[str, int, int, int, float, bool]) -> Trajectory:
    """Generate one episode.  Top-level so it is picklable by multiprocessing."""
    scenario, seed, episode_length, n_agents, epsilon, record_graph = args
    if scenario in ("train", "val", "test_iid", "iid"):
        cfg = sample_train_config(seed, n_agents=n_agents, episode_length=episode_length)
        cfg = cfg.variant(name=scenario)
    else:
        cfg = make_scenario(scenario, seed, n_agents=n_agents,
                            episode_length=episode_length)
    runner = EpisodeRunner(cfg)
    policy = BehaviorPolicy(runner.action_space, epsilon=epsilon, seed=seed)
    tr = runner.rollout(policy, seed=seed, record_graph=record_graph)
    tr.meta["split_scenario"] = scenario
    tr.meta["kind"] = SCENARIO_KIND.get(scenario, "train")
    return tr


def generate_split(split: str, scenario: str, seeds: Sequence[int], spec: GenSpec,
                   verbose: bool = True) -> List[Trajectory]:
    """Generate every episode of one split (optionally in parallel)."""
    args = [(scenario, int(s), spec.episode_length, spec.n_agents, spec.epsilon,
             spec.record_graph) for s in seeds]
    if spec.n_workers > 1 and len(args) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(spec.n_workers) as pool:
            trajs = pool.map(_episode, args)
    else:
        trajs = [_episode(a) for a in args]
    out_dir = os.path.join(spec.out_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    for seed, tr in zip(seeds, trajs):
        tr.save(os.path.join(out_dir, f"ep_{int(seed):07d}.npz"))
    if verbose:
        ev = np.mean([t.events.mean(axis=0) for t in trajs], axis=0)
        print(f"  [{split}] {len(trajs)} episodes x {trajs[0].T} steps  "
              f"event rate {ev.sum():.3f}/step")
    return trajs


def generate_dataset(spec: GenSpec, verbose: bool = True) -> Dict[str, Any]:
    """Generate all splits and write a manifest."""
    os.makedirs(spec.out_dir, exist_ok=True)
    manifest: Dict[str, Any] = {"spec": spec.to_dict(), "splits": {}}

    plan: List[Tuple[str, str, List[int]]] = [
        ("train", "train", [SPLIT_SEED_BASE["train"] + i for i in range(spec.n_train)]),
        ("val", "val", [SPLIT_SEED_BASE["val"] + i for i in range(spec.n_val)]),
        ("test_iid", "test_iid",
         [SPLIT_SEED_BASE["test_iid"] + i for i in range(spec.n_test)]),
    ]
    for sc in spec.scenarios:
        base = ood_seed_base(sc)
        plan.append((f"ood_{sc}", sc, [base + i for i in range(spec.n_ood)]))

    for split, scenario, seeds in plan:
        if verbose:
            print(f"generating {split} ({scenario}) ...", flush=True)
        generate_split(split, scenario, seeds, spec, verbose=verbose)
        manifest["splits"][split] = {
            "scenario": scenario,
            "seeds": [int(s) for s in seeds],
            "n_episodes": len(seeds),
            "kind": SCENARIO_KIND.get(scenario, "train"),
        }
    with open(os.path.join(spec.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


# --------------------------------------------------------------------------
def load_manifest(out_dir: str) -> Dict[str, Any]:
    with open(os.path.join(out_dir, "manifest.json")) as fh:
        return json.load(fh)


def load_split(out_dir: str, split: str, limit: Optional[int] = None) -> List[Trajectory]:
    d = os.path.join(out_dir, split)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"No such split directory: {d}")
    files = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
    if limit is not None:
        files = files[:limit]
    return [Trajectory.load(os.path.join(d, f)) for f in files]
