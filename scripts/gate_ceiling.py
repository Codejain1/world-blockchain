#!/usr/bin/env python3
"""How good could the unified system be if its gate were perfect?

The differential audit shows the unified system's components are wired
correctly: forcing its gate to one pathway reproduces the corresponding
standalone system exactly. So its deficit is arbitration, not plumbing.

This measures the ceiling. At every step an *oracle gate* forks the real
simulator, tries the action each pathway proposes, keeps whichever actually
produced more reward, and executes that one. It is privileged - it sees the
consequence before committing - so it is an upper bound on what any gate could
achieve with these components, not a competitor.

If the oracle gate beats the best single pathway, combining reasoning with
simulation has real headroom here and the gate is worth fixing. If it does not,
the components are redundant and no arbitration would help.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from typing import Dict, List

import numpy as np

from bwm.data.dataset import Normalizer
from bwm.environment.action_space import DiscreteActionSpace
from bwm.environment.observation import ObservationBuilder
from bwm.environment.runner import EpisodeRunner
from bwm.environment.scenarios import base_config, make_scenario, sample_train_config
from bwm.evaluation.control import run_policy_episode
from bwm.experiments.registry import build_predictive_model
from bwm.experiments.train import _train_cfg, load_data
from bwm.models.base import DecisionContext
from bwm.models.llm.agent import LLMPolicy
from bwm.models.llm.client import LLMConfig
from bwm.models.unified.unified import UnifiedPolicy
from bwm.models.world_model.policy import WorldModelPolicy
from bwm.planning.planner import PlannerConfig
from bwm.utils.config import load_config


def oracle_gate_episode(reason_pol, plan_pol, env_cfg, nz, history=8, steps=120):
    """Run one episode choosing, at each step, whichever pathway actually did better."""
    runner = EpisodeRunner(env_cfg)
    runner.reset(env_cfg.seed)
    world, agent, asp = runner.world, runner.focal_agent, runner.action_space
    reason_pol.reset(env_cfg.seed)
    plan_pol.reset(env_cfg.seed)

    raw_win = deque(maxlen=history)
    act_win = deque(maxlen=history)
    o0 = runner.observe()
    for _ in range(history):
        raw_win.append(o0)
        act_win.append(0)

    rewards = np.zeros(steps, np.float32)
    nw = np.zeros(steps + 1, np.float32)
    nw[0] = float(world.net_worth()[agent])
    chose_plan = 0
    agree = 0

    for t in range(steps):
        raw = np.asarray(raw_win[-1], np.float32)
        raw_hist = np.stack(list(raw_win))
        ctx = DecisionContext(
            obs_hist=nz.obs(raw_hist), act_hist=np.asarray(list(act_win), np.int64),
            t=t, agent=agent, action_space=asp,
            feasible=asp.feasible_mask(world, agent), raw_obs=raw,
            info={"raw_obs_hist": raw_hist})
        a_reason = int(reason_pol.act(ctx))
        a_plan = int(plan_pol.act(ctx))
        if a_reason == a_plan:
            agree += 1
            best = a_reason
        else:
            # Privileged: fork the real world and see which action actually pays.
            scores = {}
            for a in (a_reason, a_plan):
                br = world.fork()
                act = asp.decode(a, agent, base_fee=float(br.state.gas_base_fee))
                scores[a] = float(br.step({agent: act}).rewards[agent])
            best = max(scores, key=lambda k: scores[k])
        if best == a_plan and a_plan != a_reason:
            chose_plan += 1
        res = runner.step_with_index(best)
        nxt = runner.observe()
        rewards[t] = float(res.rewards[agent])
        nw[t + 1] = float(world.net_worth()[agent])
        reason_pol.observe_outcome(ctx, best, rewards[t], nxt, res.events)
        plan_pol.observe_outcome(ctx, best, rewards[t], nxt, res.events)
        raw_win.append(nxt)
        act_win.append(best)

    init = max(abs(float(nw[0])), 1e-9)
    return {"total_return": float((nw[-1] - nw[0]) / init),
            "plan_share": chose_plan / max(steps - agree, 1),
            "agreement": agree / steps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--scenarios", default="iid,liquidity_crisis")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    ts, vs, spec, _ = load_data(cfg)
    nz = Normalizer.load(f"{a.results}/normalizer.npz")
    ob = ObservationBuilder(base_config(n_agents=120))
    asp = DiscreteActionSpace(base_config(n_agents=120))
    wm = build_predictive_model("wm_rssm", spec, ob, _train_cfg(cfg))
    wm.load(f"{a.results}/checkpoints/wm_rssm.pt")
    wm.spec, wm.normalizer = spec, nz
    pcfg = PlannerConfig(horizon=6, n_candidates=48, n_iters=3, seed=0)
    lcfg = LLMConfig(provider="anthropic", cache_dir=None)
    F, A = ob.feature_names, asp.names

    rows: List[Dict] = []
    for sc in a.scenarios.split(","):
        for i in range(a.episodes):
            seed = 800_000 + 1000 * ["iid", "liquidity_crisis", "unseen_agents",
                                     "novel_mechanism", "shock_storm"].index(sc) + i
            env = (sample_train_config(seed, n_agents=120, episode_length=a.steps + 8)
                   if sc == "iid" else
                   make_scenario(sc, seed, n_agents=120, episode_length=a.steps + 8))
            run = lambda p: run_policy_episode(p, env, nz, history=8, steps=a.steps,
                                               seed=env.seed).summary()["total_return"]
            r = {
                "scenario": sc, "seed": seed,
                "llm": run(LLMPolicy(F, A, lcfg, name="llm")),
                "wm_plan": run(WorldModelPolicy(wm, asp, nz, pcfg, name="wm_plan")),
                "unified_learned_gate": run(UnifiedPolicy(
                    wm, asp, nz, LLMPolicy(F, A, lcfg, name="r"), F,
                    planner_cfg=pcfg, name="u")),
            }
            oc = oracle_gate_episode(LLMPolicy(F, A, lcfg, name="r"),
                                     WorldModelPolicy(wm, asp, nz, pcfg, name="p"),
                                     env, nz, steps=a.steps)
            r["unified_oracle_gate"] = oc["total_return"]
            r["oracle_plan_share"] = oc["plan_share"]
            r["pathway_agreement"] = oc["agreement"]
            rows.append(r)
            print("  %-18s seed %d | llm %+.4f  wm_plan %+.4f  unified %+.4f  "
                  "ORACLE-GATE %+.4f  (agree %.0f%%, plan-share %.0f%%)" % (
                      sc, seed, r["llm"], r["wm_plan"], r["unified_learned_gate"],
                      r["unified_oracle_gate"], 100 * oc["agreement"],
                      100 * oc["plan_share"]), flush=True)

    keys = ["llm", "wm_plan", "unified_learned_gate", "unified_oracle_gate"]
    print("\n%-24s %10s" % ("system", "mean return"))
    for k in keys:
        print("%-24s %+10.4f" % (k, float(np.mean([r[k] for r in rows]))))
    print("\npathway agreement %.0f%%; when they disagree the oracle picks the "
          "planner %.0f%% of the time" % (
              100 * np.mean([r["pathway_agreement"] for r in rows]),
              100 * np.mean([r["oracle_plan_share"] for r in rows])))
    out = f"{a.results}/gate_ceiling.json"
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
