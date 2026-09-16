#!/usr/bin/env python3
"""Fit the unified system's arbitration gate offline, then evaluate it.

Why this exists
---------------
The gate was reset at every episode boundary, so LinUCB over a 9-dimensional
context got ~53 pulls per arm before losing everything it had learned. It never
left exploration, which is why it collapsed onto a single pathway on 99.4% of
decisions. Meanwhile an oracle gate over the same components scores +0.115,
better than any other learned system in the study -- so the arbitration layer,
not the architecture, was throwing the result away.

Protocol
--------
Data is collected on **training** seeds with a uniformly random gate (so every
arm is covered), and the fitted gate is evaluated on the **held-out control**
seeds. That is the same train/test discipline every other model in the lab gets,
and it keeps the gate's advantage from being cross-episode memorisation of the
evaluation worlds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

from bwm.data.dataset import Normalizer
from bwm.environment.action_space import DiscreteActionSpace
from bwm.environment.observation import ObservationBuilder
from bwm.environment.scenarios import base_config, make_scenario, sample_train_config
from bwm.evaluation.control import run_policy_episode, summarize_control
from bwm.evaluation.stats import paired_permutation_test, paired_bootstrap
from bwm.experiments.registry import build_predictive_model
from bwm.experiments.train import _train_cfg, load_data
from bwm.models.llm.agent import LLMPolicy
from bwm.models.llm.client import LLMConfig
from bwm.models.unified.unified import LinUCBGate, UnifiedPolicy
from bwm.models.world_model.policy import WorldModelPolicy
from bwm.planning.planner import PlannerConfig
from bwm.utils.config import load_config

SCEN = ["iid", "liquidity_crisis", "unseen_agents", "novel_mechanism", "shock_storm"]


def build(cfg, results):
    ts, vs, spec, _ = load_data(cfg)
    nz = Normalizer.load(f"{results}/normalizer.npz")
    ob = ObservationBuilder(base_config(n_agents=120))
    asp = DiscreteActionSpace(base_config(n_agents=120))
    wm = build_predictive_model("wm_rssm", spec, ob, _train_cfg(cfg))
    wm.load(f"{results}/checkpoints/wm_rssm.pt")
    wm.spec, wm.normalizer = spec, nz
    return nz, ob, asp, wm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--collect-episodes", type=int, default=16)
    ap.add_argument("--collect-steps", type=int, default=160)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--eval-steps", type=int, default=160)
    ap.add_argument("--out", default="results/gate_fix")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    cfg = load_config(a.config)
    nz, ob, asp, wm = build(cfg, a.results)
    pcfg = PlannerConfig(horizon=6, n_candidates=48, n_iters=3, seed=0)
    lcfg = LLMConfig(provider="anthropic", cache_dir=None)
    F, A = ob.feature_names, asp.names

    def unified(**kw):
        return UnifiedPolicy(wm, asp, nz, LLMPolicy(F, A, lcfg, name="r"), F,
                             planner_cfg=pcfg, name=kw.pop("name", "unified"), **kw)

    # ---- 1. collect on TRAINING seeds with a uniformly random gate --------
    print(f"collecting gate data on {a.collect_episodes} training-seed episodes "
          f"(uniform-random arm) ...", flush=True)
    collector = unified(name="collector", persist_gate=True, gate_explore=1.0,
                        gate_seed=7)
    t0 = time.perf_counter()
    for i in range(a.collect_episodes):
        seed = i * 37 % 100_000                 # training seed range [0, 100k)
        sc = SCEN[i % len(SCEN)]
        env = (sample_train_config(seed, n_agents=120, episode_length=a.collect_steps + 8)
               if sc == "iid" else
               make_scenario(sc, seed, n_agents=120, episode_length=a.collect_steps + 8))
        run_policy_episode(collector, env, nz, history=8, steps=a.collect_steps,
                           seed=seed)
        if (i + 1) % 4 == 0:
            print(f"  {i+1}/{a.collect_episodes} episodes, "
                  f"{len(collector.gate_log)} decisions logged "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    X = np.stack([g[0] for g in collector.gate_log])
    arms = np.asarray([g[1] for g in collector.gate_log])
    rew = np.asarray([g[2] for g in collector.gate_log])
    print(f"\ncollected {len(X)} decisions; per-arm counts "
          f"{np.bincount(arms, minlength=3).tolist()}")
    print("mean standardised reward by arm (reason/plan/recall): " +
          ", ".join(f"{np.mean(rew[arms==k]):+.4f}" if np.any(arms == k) else "n/a"
                    for k in range(3)))

    fitted = LinUCBGate(3, UnifiedPolicy.GATE_DIM, alpha=0.4).fit(X, arms, rew)
    with open(f"{a.out}/gate.json", "w") as fh:
        json.dump(fitted.state_dict(), fh)
    print(f"fitted gate -> {a.out}/gate.json")

    # ---- 2. evaluate on the HELD-OUT control seeds ------------------------
    variants = {
        "unified_baseline": lambda: unified(name="unified_baseline"),
        "unified_warmstart": lambda: _warm(unified(name="unified_warmstart",
                                                   persist_gate=True), fitted),
        "wm_plan": lambda: WorldModelPolicy(wm, asp, nz, pcfg, name="wm_plan"),
        "llm": lambda: LLMPolicy(F, A, lcfg, name="llm"),
    }
    per: Dict[str, Dict[str, List]] = {k: {} for k in variants}
    print("\nevaluating on held-out control seeds ...", flush=True)
    for sc in SCEN:
        envs = []
        for i in range(a.eval_episodes):
            seed = 800_000 + 1000 * SCEN.index(sc) + i
            envs.append(sample_train_config(seed, n_agents=120,
                                            episode_length=a.eval_steps + 8)
                        if sc == "iid" else
                        make_scenario(sc, seed, n_agents=120,
                                      episode_length=a.eval_steps + 8))
        for name, mk in variants.items():
            pol = mk()
            eps = [run_policy_episode(pol, e, nz, history=8, steps=a.eval_steps,
                                      seed=e.seed) for e in envs]
            per[name][sc] = eps
            s = summarize_control(eps)
            print(f"  {sc:<18s} {name:<18s} ret {s['total_return']:+.4f}", flush=True)

    # ---- 3. report --------------------------------------------------------
    print("\n%-20s %12s" % ("system", "mean return"))
    means = {}
    for name in variants:
        v = [r.summary()["total_return"] for sc in SCEN for r in per[name][sc]]
        means[name] = float(np.mean(v))
        print("%-20s %+12.4f" % (name, means[name]))

    def paired(a_, b_):
        xa = [r.summary()["total_return"] for sc in SCEN for r in per[a_][sc]]
        xb = [r.summary()["total_return"] for sc in SCEN for r in per[b_][sc]]
        bs = paired_bootstrap(xa, xb, seed=0)
        return bs, paired_permutation_test(xa, xb, seed=0)

    print("\npaired comparisons (n=%d episodes):" % (len(SCEN) * a.eval_episodes))
    for x, y in (("unified_warmstart", "unified_baseline"),
                 ("unified_warmstart", "wm_plan"),
                 ("unified_warmstart", "llm")):
        bs, p = paired(x, y)
        print("  %-38s %+.4f  [%+.3f, %+.3f]  p=%.3f%s" % (
            f"{x} - {y}", bs["diff"], bs["lo"], bs["hi"], p,
            "  SIGNIFICANT" if p < 0.05 else ""))

    ws = per["unified_warmstart"][SCEN[0]][0]
    json.dump({"means": means,
               "gate_usage_example": ws.info.get("gate_usage")},
              open(f"{a.out}/summary.json", "w"), indent=2, default=str)
    print(f"\nwrote {a.out}/summary.json")
    return 0


def _warm(policy: UnifiedPolicy, fitted: LinUCBGate) -> UnifiedPolicy:
    policy.gate.load_state_dict(fitted.state_dict())
    policy._warm = True
    return policy


if __name__ == "__main__":
    sys.exit(main())
