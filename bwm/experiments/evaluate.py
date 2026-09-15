"""The evaluation pipeline: prediction, counterfactual, control, ablations.

Produces one JSON results file plus the tables and plots referenced by the
report.  Nothing here selects models or thresholds -- every choice comes from the
config, and every raw metric is written out alongside the summaries.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..data.dataset import TrajectorySet
from ..data.generate import load_manifest, load_split
from ..environment.scenarios import SCENARIO_KIND, make_scenario, sample_train_config
from ..evaluation.benchmark import (composite_index, control_table,
                                    counterfactual_table, prediction_table)
from ..evaluation.control import (regret_table, run_policy_episode, summarize_control)
from ..evaluation.counterfactual import collect_probes, evaluate_counterfactual
from ..evaluation.prediction import (evaluate_prediction, reference_predictions,
                                     sample_eval_batches)
from ..models.llm.client import LLMConfig
from ..planning.oracle import OracleConfig, OraclePlannerPolicy
from ..planning.planner import PlannerConfig
from ..utils.config import Config
from .registry import GRAPH_MODELS, build_policies
from .train import TrainedBundle

__all__ = ["run_prediction_stage", "run_counterfactual_stage", "run_control_stage",
           "run_evaluation"]


def _planner_cfg(cfg: Config) -> PlannerConfig:
    p = cfg.sub("planner")
    return PlannerConfig(
        horizon=int(p.get("horizon", 6)), n_candidates=int(p.get("n_candidates", 48)),
        n_iters=int(p.get("n_iters", 3)), n_elites=int(p.get("n_elites", 8)),
        gamma=float(p.get("gamma", 0.95)), risk_lambda=float(p.get("risk_lambda", 0.0)),
        method=str(p.get("method", "cem")), seed=int(p.get("seed", 0)))


def _llm_cfg(cfg: Config) -> LLMConfig:
    l = cfg.sub("llm")
    return LLMConfig(provider=str(l.get("provider", "offline")),
                     model=str(l.get("model", "claude-opus-5")),
                     max_tokens=int(l.get("max_tokens", 400)),
                     temperature=float(l.get("temperature", 0.0)),
                     cache_dir=l.get("cache_dir"))


# --------------------------------------------------------------------------
def run_prediction_stage(bundle: TrainedBundle, cfg: Config, verbose: bool = True
                         ) -> Dict[str, Dict[str, Dict[str, Any]]]:
    data_dir = cfg.get_path("data.dir", "datasets/main")
    splits: List[str] = list(cfg.get_path("eval.splits", ["test_iid"]))
    n_samples = int(cfg.get_path("eval.n_samples", 5000))
    horizons = [int(h) for h in cfg.get_path("eval.horizons", [1, 2, 4, 6])]
    horizons = [h for h in horizons if h <= bundle.spec.horizon]
    names = bundle.obs_builder.feature_names
    seed = int(cfg.get_path("experiment.seed", 0))

    out: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in bundle.models}
    for sp in splits:
        trajs = load_split(data_dir, sp)
        ts = TrajectorySet(trajs, sp)
        batch = sample_eval_batches(ts, bundle.spec, bundle.normalizer,
                                    n_samples=n_samples, seed=seed, with_graph=True)
        # Every system on this split is scored against the *same* reference
        # predictions from the linear autoregression, so `skill_vs_ar` is
        # comparable across models and across splits.
        ref_name = "ar" if "ar" in bundle.models else (
            "persistence" if "persistence" in bundle.models else None)
        ref = (reference_predictions(bundle.models[ref_name], batch,
                                     max(horizons) if horizons else 1)
               if ref_name else None)
        if verbose:
            print(f"  prediction | {sp}: {batch['obs_hist'].shape[0]} windows "
                  f"(reference: {ref_name})", flush=True)
        for name, model in bundle.models.items():
            t0 = time.perf_counter()
            out[name][sp] = evaluate_prediction(model, batch, bundle.spec, names,
                                                horizons=horizons, reference=ref)
            out[name][sp]["eval_seconds"] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------
def run_counterfactual_stage(bundle: TrainedBundle, cfg: Config, verbose: bool = True
                             ) -> Dict[str, Dict[str, Any]]:
    c = cfg.sub("counterfactual")
    scenarios = list(c.get("scenarios", ["iid"]))
    kinds = list(c.get("kinds", ["substitute"]))
    n_eps = int(c.get("n_episodes", 3))
    n_probes = int(c.get("n_probes", 6))
    horizon = min(int(c.get("horizon", 6)), bundle.spec.horizon)
    seed_base = int(c.get("seed_base", 700_000))
    n_agents = int(cfg.get_path("data.n_agents", 120))

    probe_sets: Dict[str, List[Any]] = {}
    for sc in scenarios:
        for kind in kinds:
            probes: List[Any] = []
            for i in range(n_eps):
                seed = seed_base + 1000 * scenarios.index(sc) + 100 * kinds.index(kind) + i
                env = (sample_train_config(seed, n_agents=n_agents, episode_length=320)
                       if sc == "iid" else
                       make_scenario(sc, seed, n_agents=n_agents, episode_length=320))
                probes.extend(collect_probes(
                    env, bundle.normalizer, history=bundle.spec.history,
                    horizon=horizon, n_probes=n_probes, seed=seed,
                    burn_in=int(c.get("burn_in", 40)), stride=int(c.get("stride", 20)),
                    kind=kind))
            probe_sets[f"{sc}:{kind}"] = probes
            if verbose:
                print(f"  counterfactual | {sc}:{kind}: {len(probes)} probes", flush=True)

    out: Dict[str, Dict[str, Any]] = {}
    for name, model in bundle.models.items():
        out[name] = {}
        for key, probes in probe_sets.items():
            out[name][key] = evaluate_counterfactual(
                model, probes, needs_graph=name in GRAPH_MODELS)
    return out


# --------------------------------------------------------------------------
def run_control_stage(bundle: TrainedBundle, cfg: Config, verbose: bool = True
                      ) -> Dict[str, Any]:
    c = cfg.sub("control")
    scenarios = list(c.get("scenarios", ["iid"]))
    n_eps = int(c.get("n_episodes", 4))
    steps = int(c.get("steps", 160))
    seed_base = int(c.get("seed_base", 800_000))
    policy_names = list(c.get("policies", []))
    primary = str(c.get("primary_world_model", "wm_rssm"))
    n_agents = int(cfg.get_path("data.n_agents", 120))

    world_models = {k: v for k, v in bundle.models.items()
                    if getattr(v, "family", "") == "world_model"}
    policies = build_policies(
        policy_names, world_models=world_models,
        action_space=_action_space(bundle, n_agents),
        obs_builder=bundle.obs_builder, normalizer=bundle.normalizer,
        planner_cfg=_planner_cfg(cfg), llm_cfg=_llm_cfg(cfg),
        primary_wm=primary, seed=int(cfg.get_path("experiment.seed", 0)))

    o = cfg.sub("oracle_planner")
    if "oracle_plan" in policies:
        policies["oracle_plan"] = OraclePlannerPolicy(
            _action_space(bundle, n_agents),
            OracleConfig(horizon=int(o.get("horizon", 3)),
                         n_candidates=int(o.get("n_candidates", 6)),
                         gamma=float(o.get("gamma", 0.95)),
                         seed=int(o.get("seed", 0))))

    results: Dict[str, Dict[str, List[Any]]] = {p: {} for p in policies}
    summaries: Dict[str, Dict[str, Dict[str, Any]]] = {p: {} for p in policies}
    infos: Dict[str, Any] = {}

    for sc in scenarios:
        env_cfgs = []
        for i in range(n_eps):
            seed = seed_base + 1000 * scenarios.index(sc) + i
            env_cfgs.append(
                sample_train_config(seed, n_agents=n_agents, episode_length=steps + 8)
                if sc == "iid" else
                make_scenario(sc, seed, n_agents=n_agents, episode_length=steps + 8))
        for pname, policy in policies.items():
            t0 = time.perf_counter()
            eps = []
            for env in env_cfgs:
                eps.append(run_policy_episode(
                    policy, env, bundle.normalizer,
                    history=bundle.spec.history, steps=steps, seed=env.seed,
                    allow_simulator=(pname == "oracle_plan"),
                    needs_graph=(primary in GRAPH_MODELS
                                 and pname.startswith(("wm_", "unified")))))
            results[pname][sc] = eps
            summaries[pname][sc] = summarize_control(eps)
            summaries[pname][sc]["scenario"] = sc
            if verbose:
                s = summaries[pname][sc]
                print(f"  control | {sc:<20s} {pname:<20s} "
                      f"ret {s['total_return']:+.4f}  sharpe {s['sharpe']:+.3f}  "
                      f"{time.perf_counter()-t0:5.1f}s", flush=True)
        infos = {p: policies[p].info().__dict__ for p in policies}

    regrets = {}
    for sc in scenarios:
        regrets[sc] = regret_table({p: results[p][sc] for p in results if sc in results[p]})
    return {"summaries": summaries, "regret": regrets, "policy_info": infos,
            "scenarios": scenarios}


def _action_space(bundle: TrainedBundle, n_agents: int):
    from ..environment.action_space import DiscreteActionSpace
    from ..environment.scenarios import base_config
    return DiscreteActionSpace(base_config(n_agents=n_agents))


# --------------------------------------------------------------------------
def run_evaluation(bundle: TrainedBundle, cfg: Config, out_dir: str,
                   stages: Sequence[str] = ("prediction", "counterfactual", "control"),
                   verbose: bool = True) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    results: Dict[str, Any] = {
        "config": dict(cfg), "config_hash": cfg.hash,
        "train_logs": bundle.logs,
        "spec": bundle.spec.to_dict(),
        "dataset_manifest": load_manifest(cfg.get_path("data.dir", "datasets/main")),
    }
    t_start = time.perf_counter()

    if "prediction" in stages:
        if verbose:
            print("stage: prediction", flush=True)
        results["prediction"] = run_prediction_stage(bundle, cfg, verbose)
    if "counterfactual" in stages:
        if verbose:
            print("stage: counterfactual", flush=True)
        results["counterfactual"] = run_counterfactual_stage(bundle, cfg, verbose)
    if "control" in stages:
        if verbose:
            print("stage: control", flush=True)
        results["control"] = run_control_stage(bundle, cfg, verbose)

    results["tables"] = _build_tables(results, cfg)
    results["composite"] = _build_composites(results, cfg)
    results["data_audit"] = _data_audit(cfg, verbose)
    from ..evaluation.audit import audit_results
    results["audit"] = audit_results(results)
    results["wall_seconds"] = time.perf_counter() - t_start
    if verbose:
        sev = results["audit"]["by_severity"]
        print("audit: " + ", ".join(f"{k}={v}" for k, v in sev.items() if v), flush=True)

    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=_jsonable)
    return results


def _data_audit(cfg: Config, verbose: bool = True, n_episodes: int = 10
                ) -> Dict[str, Any]:
    """Contamination checks that run with every experiment, not on request."""
    from ..data.audit import audit_observation_overlap, audit_split_disjointness
    d = cfg.get_path("data.dir", "datasets/main")
    manifest = load_manifest(d)
    seed_audit = audit_split_disjointness(manifest)
    train = load_split(d, "train", n_episodes)
    overlaps: Dict[str, float] = {}
    for split in manifest["splits"]:
        if split == "train":
            continue
        other = load_split(d, split, n_episodes)
        overlaps[split] = audit_observation_overlap(train, other)["overlap_frac_b"]
    if verbose:
        print(f"  data audit | seeds disjoint={seed_audit['ok']} "
              f"max duplicate-state overlap={max(overlaps.values(), default=0):.5f}",
              flush=True)
    return {"seed_disjoint": seed_audit["ok"], "collisions": seed_audit["collisions"],
            "observation_overlap": overlaps}


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _build_tables(results: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    tables: Dict[str, Any] = {}
    if "prediction" in results:
        splits = list(cfg.get_path("eval.splits", []))
        tables["prediction"] = prediction_table(results["prediction"], splits)
    if "counterfactual" in results:
        tables["counterfactual"] = counterfactual_table(results["counterfactual"])
    if "control" in results:
        sc = results["control"]["scenarios"]
        tables["control"] = control_table(results["control"]["summaries"], sc)
    return tables


def _scaled_returns(control: Dict[str, Any]) -> Dict[str, float]:
    """Min-max scale mean control return between do-nothing and the best system."""
    summaries = control["summaries"]
    means: Dict[str, float] = {}
    for p, per_sc in summaries.items():
        vals = [v.get("total_return", 0.0) for v in per_sc.values()]
        means[p] = float(np.mean(vals)) if vals else 0.0
    lo = means.get("noop", min(means.values()) if means else 0.0)
    hi = max(means.values()) if means else 1.0
    span = max(hi - lo, 1e-9)
    return {p: (v - lo) / span for p, v in means.items()}


#: How a predictive system maps onto the policy that uses it, for the composite.
_MODEL_TO_POLICY = {
    "wm_rssm": "wm_plan", "wm_jepa": "wm_plan", "wm_transformer": "wm_plan",
    "wm_graph": "wm_plan", "wm_rssm_1step": "wm_plan",
}


def _build_composites(results: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    pred = results.get("prediction", {})
    cf = results.get("counterfactual", {})
    control = results.get("control")
    scaled = _scaled_returns(control) if control else {}
    primary = str(cfg.get_path("control.primary_world_model", "wm_rssm"))

    out: Dict[str, Any] = {}
    for model in pred:
        ctrl = None
        pol = _MODEL_TO_POLICY.get(model)
        # Only the world model that actually drove the planner may claim its
        # control score; the others get a partial composite with lower coverage.
        if pol and model == primary and pol in scaled:
            ctrl = {"scaled_return": scaled[pol],
                    "total_return": float(np.mean(
                        [v.get("total_return", 0.0)
                         for v in control["summaries"][pol].values()]))}
        out[model] = composite_index(pred.get(model, {}), cf.get(model), ctrl)
    # Policy-only systems (LLM, unified, oracle) get a planning-only composite.
    if control:
        for pol in control["summaries"]:
            if pol in out:
                continue
            key = f"policy:{pol}"
            ctrl = {"scaled_return": scaled.get(pol),
                    "total_return": float(np.mean(
                        [v.get("total_return", 0.0)
                         for v in control["summaries"][pol].values()]))}
            if pol.startswith("unified") or pol.startswith("wm_"):
                out[key] = composite_index(pred.get(primary, {}), cf.get(primary), ctrl)
                out[key]["note"] = (f"prediction/counterfactual components are those of "
                                    f"its world model ({primary})")
            else:
                out[key] = composite_index({}, None, ctrl)
    return out
