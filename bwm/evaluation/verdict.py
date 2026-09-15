"""Evaluate the hypothesis against the falsification criteria stated in advance.

``docs/METHODOLOGY.md`` commits, before any result is seen, to the conditions
under which the world-model hypothesis would be considered unsupported.  This
module checks those conditions mechanically against a results file so the verdict
is not a matter of narrative emphasis.

Each check returns one of:

``supported``      the evidence goes the way the hypothesis predicts
``not_supported``  the evidence goes the other way
``inconclusive``   the comparison could not be made, or the difference is not
                   distinguishable from noise at this sample size
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["evaluate_hypothesis", "format_verdict"]

WORLD_MODELS = ("wm_rssm", "wm_jepa", "wm_transformer", "wm_graph")
FLAT_CONTROL = "obsspace"        # same multi-step objective, no learned latent
SEQ_BASELINE = "transformer"


def _f(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def _cf_mean(results: Dict[str, Any], model: str) -> Optional[float]:
    per = results.get("counterfactual", {}).get(model, {})
    vals = [_f(v.get("skill_vs_zero_effect")) for v in per.values()
            if isinstance(v, dict)]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _ood_mean(results: Dict[str, Any], model: str) -> Optional[float]:
    per = results.get("prediction", {}).get(model, {})
    vals = [_f(v.get("decision_state", {}).get("skill_vs_best_naive"))
            for k, v in per.items() if k.startswith("ood_")]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _long_horizon(results: Dict[str, Any], model: str,
                  split: str = "test_iid") -> Optional[float]:
    lh = results.get("prediction", {}).get(model, {}).get(split, {}).get("long_horizon", {})
    if not lh:
        return None
    depth = str(max(int(k) for k in lh))
    return _f(lh[depth].get("decision_skill_vs_best_naive"))


def _h2h(results: Dict[str, Any], a: str, b: str) -> Optional[Dict[str, Any]]:
    for row in results.get("control", {}).get("head_to_head", []) or []:
        if row.get("comparison") == f"{a} - {b}":
            return row
    return None


def _best(vals: Dict[str, Optional[float]]) -> Tuple[Optional[str], Optional[float]]:
    ok = {k: v for k, v in vals.items() if v is not None}
    if not ok:
        return None, None
    k = max(ok, key=lambda k: ok[k])
    return k, ok[k]


def evaluate_hypothesis(results: Dict[str, Any], alpha: float = 0.05
                        ) -> Dict[str, Any]:
    """Run every pre-registered falsification check."""
    checks: List[Dict[str, Any]] = []

    def add(name: str, verdict: str, detail: str, **extra: Any) -> None:
        checks.append({"check": name, "verdict": verdict, "detail": detail, **extra})

    # --- 1. does latent state beat the same objective without it? -------
    wm_cf = {m: _cf_mean(results, m) for m in WORLD_MODELS}
    flat_cf = _cf_mean(results, FLAT_CONTROL)
    best_wm, best_cf = _best(wm_cf)
    if best_cf is None or flat_cf is None:
        add("latent_vs_same_objective", "inconclusive",
            "counterfactual scores unavailable for the comparison")
    elif best_cf > flat_cf:
        add("latent_vs_same_objective", "supported",
            f"best world model ({best_wm}, {best_cf:+.4f}) beats the "
            f"observation-space control with the identical multi-step objective "
            f"({FLAT_CONTROL}, {flat_cf:+.4f}) on counterfactual skill",
            margin=best_cf - flat_cf)
    else:
        add("latent_vs_same_objective", "not_supported",
            f"the observation-space control ({flat_cf:+.4f}) matches or beats every "
            f"world model (best {best_wm} {best_cf:+.4f}); the multi-step objective, "
            f"not latent state, explains the result",
            margin=best_cf - flat_cf)

    # --- 2. is it just sequence modelling? ------------------------------
    wm_lh = {m: _long_horizon(results, m) for m in WORLD_MODELS}
    seq_lh = _long_horizon(results, SEQ_BASELINE)
    best_lh_m, best_lh = _best(wm_lh)
    if best_lh is None or seq_lh is None:
        add("beyond_sequence_modelling", "inconclusive",
            "long-horizon scores unavailable")
    elif best_lh > seq_lh:
        add("beyond_sequence_modelling", "supported",
            f"best world model ({best_lh_m}, {best_lh:+.4f}) beats the Transformer "
            f"baseline ({seq_lh:+.4f}) at the deepest horizon",
            margin=best_lh - seq_lh)
    else:
        add("beyond_sequence_modelling", "not_supported",
            f"the Transformer baseline ({seq_lh:+.4f}) matches or beats every world "
            f"model at the deepest horizon (best {best_lh_m} {best_lh:+.4f})",
            margin=best_lh - seq_lh)

    # --- 3. does planning pay for itself? -------------------------------
    r = _h2h(results, "wm_plan", "wm_greedy")
    if r is None:
        add("planning_pays", "inconclusive", "wm_plan vs wm_greedy not available")
    elif _f(r.get("p_value")) is not None and r["p_value"] < alpha and r["mean_diff"] > 0:
        add("planning_pays", "supported",
            f"planning beats one-step greedy by {r['mean_diff']:+.4f} return "
            f"(p={r['p_value']:.3f}, n={r['n_pairs']})", **r)
    elif r["mean_diff"] <= 0:
        add("planning_pays", "not_supported",
            f"planning does not beat one-step greedy ({r['mean_diff']:+.4f} return, "
            f"p={r['p_value']:.3f})", **r)
    else:
        add("planning_pays", "inconclusive",
            f"planning leads by {r['mean_diff']:+.4f} but not distinguishably "
            f"(p={r['p_value']:.3f}, n={r['n_pairs']})", **r)

    # --- 4. the hypothesis's own ordering -------------------------------
    for name, (a, b) in {"world_model_beats_reasoner": ("wm_plan", "llm"),
                         "unified_beats_world_model": ("unified", "wm_plan"),
                         "unified_beats_reasoner": ("unified", "llm")}.items():
        r = _h2h(results, a, b)
        if r is None:
            add(name, "inconclusive", f"{a} vs {b} not available")
            continue
        p, d = _f(r.get("p_value")), _f(r.get("mean_diff"))
        if p is not None and d is not None and p < alpha and d > 0:
            add(name, "supported",
                f"{a} beats {b} by {d:+.4f} return (p={p:.3f}, n={r['n_pairs']})", **r)
        elif d is not None and d <= 0:
            add(name, "not_supported",
                f"{a} does not beat {b} ({d:+.4f} return, p={p:.3f})", **r)
        else:
            add(name, "inconclusive",
                f"{a} leads {b} by {d:+.4f} but not distinguishably (p={p:.3f}, "
                f"n={r['n_pairs']})", **r)

    # --- 5. is the environment informative at all? ----------------------
    r = _h2h(results, "oracle_plan", "wm_plan")
    summaries = results.get("control", {}).get("summaries", {})
    if "oracle_plan" in summaries and "noop" in summaries:
        def mean_ret(p: str) -> float:
            v = [x.get("total_return", 0.0) for x in summaries[p].values()]
            return float(np.mean(v)) if v else 0.0
        gap = mean_ret("oracle_plan") - mean_ret("noop")
        if gap > 0:
            add("environment_rewards_dynamics", "supported",
                f"planning with the true simulator beats do-nothing by {gap:+.4f} "
                f"return, so accurate dynamics are worth something in this world",
                margin=gap)
        else:
            add("environment_rewards_dynamics", "not_supported",
                f"true-simulator planning does not beat do-nothing ({gap:+.4f}); "
                f"the comparison is uninformative", margin=gap)
    else:
        add("environment_rewards_dynamics", "inconclusive", "oracle/noop unavailable")

    # --- 6. how far is the learned model from perfect dynamics? ----------
    if r is not None:
        add("learned_vs_perfect_dynamics", "info",
            f"the true-simulator planner beats the learned planner by "
            f"{r['mean_diff']:+.4f} return (p={r['p_value']:.3f}); this is the "
            f"headroom a better learned model could recover", **r)

    # --- 7. ablations that isolate memory and rollouts -------------------
    for name, (a, b) in {"memory_helps_world_model": ("wm_plan_memory", "wm_plan"),
                         "memory_helps_unified": ("unified", "unified_no_memory"),
                         "rollouts_help_unified": ("unified", "unified_no_rollout")}.items():
        rr = _h2h(results, a, b)
        if rr is None:
            add(name, "inconclusive", f"{a} vs {b} not available")
            continue
        p, d = _f(rr.get("p_value")), _f(rr.get("mean_diff"))
        v = ("supported" if (p is not None and p < alpha and d and d > 0)
             else "not_supported" if (d is not None and d <= 0) else "inconclusive")
        add(name, v, f"{a} - {b} = {d:+.4f} return (p={p:.3f}, n={rr['n_pairs']})", **rr)

    counts: Dict[str, int] = {}
    for c in checks:
        counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    return {"checks": checks, "counts": counts, "alpha": alpha}


def format_verdict(v: Dict[str, Any]) -> str:
    lines = ["## Pre-registered falsification checks", "",
             "These criteria were written in `docs/METHODOLOGY.md` before the "
             "results were seen.", "",
             "Counts: " + ", ".join(f"{k}={n}" for k, n in sorted(v["counts"].items())),
             ""]
    icon = {"supported": "yes", "not_supported": "NO", "inconclusive": "--",
            "info": "info"}
    lines += ["| check | verdict | detail |", "|---|---|---|"]
    for c in v["checks"]:
        lines.append(f"| `{c['check']}` | **{icon.get(c['verdict'], c['verdict'])}** "
                     f"| {c['detail']} |")
    lines.append("")
    return "\n".join(lines)
