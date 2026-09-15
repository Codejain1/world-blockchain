"""Turn a results dict into markdown tables and figures."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..evaluation.audit import format_audit
from ..evaluation.benchmark import COMPONENTS, to_csv, to_markdown

__all__ = ["write_report", "results_markdown"]


def _sorted_rows(rows: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (r.get(key) is None,
                                       -(r.get(key) if isinstance(r.get(key), (int, float))
                                         else 0)))


def results_markdown(results: Dict[str, Any]) -> str:
    md: List[str] = ["# Benchmark results", ""]
    cfgh = results.get("config_hash", "?")
    md += [f"Config hash `{cfgh}`. "
           f"Total evaluation wall time {results.get('wall_seconds', 0):.1f}s.", ""]

    # --- compute / capacity ------------------------------------------------
    tl = results.get("train_logs", {})
    if tl:
        rows = [{"model": k, "family": v.get("family"), "params": v.get("n_params"),
                 "train_s": v.get("wall_seconds"), "peak_rss_mb": v.get("peak_rss_mb")}
                for k, v in tl.items()]
        md += ["## Training cost and capacity", "",
               to_markdown(rows, ["model", "family", "params", "train_s",
                                  "peak_rss_mb"], nd=1), ""]

    tables = results.get("tables", {})
    if "prediction" in tables:
        rows = tables["prediction"]
        md += [
            "## Prediction", "",
            "Targets are grouped by mechanism, because a single pooled MSE over all "
            "112 features is dominated by two artefacts in this world:", "",
            "* `ret1_*` / `pool_move_*` are already one-step differences, so "
            "predicting their change is mostly the identity \"next return ~ its "
            "mean\" (a ridge fit scores +0.99 there while learning nothing);",
            "* `logprice_*` changes are a martingale by construction, so nobody "
            "can predict them.", "",
            "`dskill` is therefore the headline: skill on the **decision-relevant** "
            "groups (the focal agent's own balance sheet plus endogenous protocol "
            "state), scored against whichever naive reference -- persistence or a "
            "fitted linear autoregression -- does better on that split.", "",
            "`price_level_skill_iid` is a **leakage canary**: price changes are "
            "unpredictable by construction, so a clearly positive value there means "
            "a model is reading something it should not be able to read.", ""]
        cols_a = ["model"] + [c for c in rows[0] if c.startswith("dskill@")] + \
                 ["dskill_ood_mean", "dskill_ood_min", "ood_retention"]
        md += ["### Decision-state skill vs the best naive reference (headline)", "",
               to_markdown(_sorted_rows(rows, "dskill@test_iid"), cols_a), ""]
        cols_b = ["model", "agent_skill_iid", "protocol_skill_iid",
                  "population_skill_iid", "price_skill_iid", "price_level_skill_iid",
                  "next_state_skill_pers_iid"]
        md += ["### Per-group detail (in-distribution)", "",
               to_markdown(_sorted_rows(rows, "agent_skill_iid"), cols_b), ""]
        cols_c = ["model", "event_logloss_iid", "event_brier_iid", "event_ece_iid",
                  "event_auroc_iid", "reward_skill_iid"]
        md += ["### Event and reward prediction (in-distribution)", "",
               to_markdown(_sorted_rows(rows, "event_auroc_iid"), cols_c), ""]

    if "counterfactual" in tables:
        rows = tables["counterfactual"]
        cols = ["model"] + [c for c in rows[0] if c.startswith("cf_skill@")] + \
               ["cf_skill_mean", "sign_acc_material_mean"]
        md += ["## Counterfactual reasoning", "",
               "`cf_skill` is the skill score against a predictor that says the "
               "action changes nothing. Positive means genuine causal signal.", "",
               to_markdown(_sorted_rows(rows, "cf_skill_mean"), cols), ""]

    if "control" in tables:
        rows = tables["control"]
        cols = ["policy"] + [c for c in rows[0] if c.startswith("ret@")] + \
               ["return_mean", "sharpe_mean", "max_drawdown_mean"]
        md += ["## Control (total return per episode)", "",
               to_markdown(_sorted_rows(rows, "return_mean"), cols), ""]
        reg = results.get("control", {}).get("regret", {})
        if reg:
            rrows = []
            pols = sorted({p for d in reg.values() for p in d})
            for p in pols:
                row = {"policy": p}
                for sc, d in reg.items():
                    row[f"regret@{sc}"] = d.get(p)
                vals = [v for k, v in row.items() if k != "policy" and v is not None]
                row["regret_mean"] = float(np.mean(vals)) if vals else None
                rrows.append(row)
            md += ["### Episode-wise regret (lower is better)", "",
                   to_markdown(sorted(rrows, key=lambda r: r.get("regret_mean") or 9e9),
                               ["policy"] + [f"regret@{s}" for s in reg] +
                               ["regret_mean"]), ""]

    h2h = results.get("control", {}).get("head_to_head")
    if h2h:
        md += ["### Head-to-head tests (paired over identical episodes)", "",
               "Every policy ran the same episode seeds, so these are paired "
               "comparisons: a bootstrap CI on the mean difference in total "
               "return, plus an exact sign-flip permutation p-value.", "",
               to_markdown(h2h, ["comparison", "mean_diff", "ci_lo", "ci_hi",
                                 "p_value", "n_pairs"]), ""]
    sig = results.get("control", {}).get("significance_vs_reference")
    if sig:
        ref = results["control"].get("significance_reference", "noop")
        md += [f"### Paired comparison against `{ref}`", "",
               to_markdown(sig, ["policy", "mean_diff", "ci_lo", "ci_hi",
                                 "p_value", "significant_05", "n_pairs"]), ""]

    comp = results.get("composite", {})
    if comp:
        rows = []
        for name, c in comp.items():
            row = {"system": name, "composite": c.get("composite"),
                   "coverage": c.get("coverage")}
            row.update({k: v for k, v in c.get("components", {}).items()})
            rows.append(row)
        md += ["## Composite GPC index (never read without the components)", "",
               "Systems are only comparable at equal `coverage`.", "",
               to_markdown(_sorted_rows(rows, "composite"),
                           ["system", "composite", "coverage", *COMPONENTS]), ""]

    ctrl = results.get("control", {})
    info = ctrl.get("policy_info", {})
    gate = {k: v.get("extra", {}).get("gate_usage") for k, v in info.items()
            if v.get("extra", {}).get("gate_usage")}
    if gate:
        md += ["## Unified gate: which pathway did it choose?", "",
               "Pathways are `reason`, `plan`, `recall`.", ""]
        rows = []
        for k, g in gate.items():
            rows.append({"policy": k, "reason": g["share"][0], "plan": g["share"][1],
                         "recall": g["share"][2],
                         "reward_reason": g["mean_reward"][0],
                         "reward_plan": g["mean_reward"][1],
                         "reward_recall": g["mean_reward"][2]})
        md += [to_markdown(rows, ["policy", "reason", "plan", "recall",
                                  "reward_reason", "reward_plan", "reward_recall"]), ""]

    backends = {k: v.get("extra", {}).get("backend") for k, v in info.items()
                if v.get("extra", {}).get("backend")}
    if backends:
        md += ["## LLM backend actually used", ""]
        for k, v in sorted(backends.items()):
            md += [f"- `{k}`: **{v}**"]
        if any(v == "rule_based_offline" for v in backends.values()):
            md += ["", "> Rows marked `rule_based_offline` were produced by the "
                   "rule-based reasoner, not by a language model. They are a "
                   "symbolic-reasoning baseline and must not be read as evidence "
                   "about LLM capability."]
        md += [""]

    if results.get("audit"):
        md += [format_audit(results["audit"]), ""]
    da = results.get("data_audit")
    if da:
        worst = max(da.get("observation_overlap", {}).values(), default=0.0)
        md += ["## Contamination check", "",
               f"- episode seeds disjoint across splits: **{da.get('seed_disjoint')}**",
               f"- largest verbatim observation-row overlap with train: "
               f"**{worst:.5f}**", ""]
    return "\n".join(md)


def write_report(results: Dict[str, Any], out_dir: str, make_plots: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "RESULTS.md"), "w") as fh:
        fh.write(results_markdown(results))
    tables = results.get("tables", {})
    for name, rows in tables.items():
        if rows:
            to_csv(rows, os.path.join(out_dir, "tables", f"{name}.csv"))
    h2h = results.get("control", {}).get("head_to_head")
    if h2h:
        md += ["### Head-to-head tests (paired over identical episodes)", "",
               "Every policy ran the same episode seeds, so these are paired "
               "comparisons: a bootstrap CI on the mean difference in total "
               "return, plus an exact sign-flip permutation p-value.", "",
               to_markdown(h2h, ["comparison", "mean_diff", "ci_lo", "ci_hi",
                                 "p_value", "n_pairs"]), ""]
    sig = results.get("control", {}).get("significance_vs_reference")
    if sig:
        ref = results["control"].get("significance_reference", "noop")
        md += [f"### Paired comparison against `{ref}`", "",
               to_markdown(sig, ["policy", "mean_diff", "ci_lo", "ci_hi",
                                 "p_value", "significant_05", "n_pairs"]), ""]

    comp = results.get("composite", {})
    if comp:
        rows = []
        for name, c in comp.items():
            row = {"system": name, "composite": c.get("composite"),
                   "coverage": c.get("coverage")}
            row.update(c.get("components", {}))
            rows.append(row)
        to_csv(rows, os.path.join(out_dir, "tables", "composite.csv"))
    if make_plots:
        try:
            from .plots import make_all_plots
            make_all_plots(results, os.path.join(out_dir, "figures"))
        except Exception as e:                       # plotting must never fail a run
            print(f"  (plotting skipped: {type(e).__name__}: {e})")
    try:
        from .dashboard import build_dashboard
        build_dashboard(results, out_dir)
    except Exception as e:
        print(f"  (dashboard skipped: {type(e).__name__}: {e})")
