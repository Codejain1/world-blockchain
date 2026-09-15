"""Composite benchmark construction and results tables.

The composite ("GPC": Generalization + Planning + Counterfactual) exists because
the project asks for a single headline number.  It is built so that it *cannot*
be used to hide the components:

* every component is stored alongside the composite, in raw units as well as
  normalised units;
* a component that a system does not support is ``None``, never silently zero,
  and ``coverage`` records how much of the index that system actually answered;
* composites are only ever compared between systems with the same coverage.

Normalisation is deliberately boring: skill scores (already 0 = baseline,
1 = perfect) are clipped to [0, 1]; control return is min-max scaled against the
do-nothing policy and the best observed system on the same episodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["COMPONENTS", "clip01", "brier_skill", "composite_index",
           "prediction_table", "control_table", "counterfactual_table",
           "to_markdown", "to_csv"]

#: The six capabilities the hypothesis is actually about.
COMPONENTS: Tuple[str, ...] = (
    "interpolation",        # in-distribution next-state skill
    "extrapolation",        # held-out-world next-state skill
    "event_calibration",    # Brier skill on discrete events (OOD included)
    "counterfactual",       # causal-effect skill vs a zero-effect predictor
    "long_horizon",         # multi-step skill at the deepest horizon
    "planning",             # control return, scaled against do-nothing
)


def clip01(x: Optional[float]) -> Optional[float]:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return float(min(max(x, 0.0), 1.0))


def brier_skill(brier: float, base_rates: Sequence[float]) -> float:
    """Brier skill score against the climatological (base-rate) forecast."""
    br = np.asarray(list(base_rates), dtype=float)
    ref = float(np.mean(br * (1.0 - br)))
    return 1.0 - float(brier) / max(ref, 1e-12)


# --------------------------------------------------------------------------
def composite_index(prediction: Dict[str, Dict[str, Any]],
                    counterfactual: Optional[Dict[str, Any]],
                    control: Optional[Dict[str, Any]],
                    iid_split: str = "test_iid",
                    ood_prefix: str = "ood_",
                    horizon_key: Optional[str] = None) -> Dict[str, Any]:
    """Assemble one system's component scores and the composite.

    Components use ``decision_state`` -- the agent and protocol target groups
    pooled -- scored against the better of the two naive references.  The price
    group is deliberately excluded (see
    :mod:`bwm.evaluation.prediction`): most of its variance is either a
    difference identity that rewards bookkeeping, or a martingale that nobody
    can predict.  Its score is still carried in ``raw`` as a leakage canary.
    """
    comp: Dict[str, Optional[float]] = {k: None for k in COMPONENTS}
    raw: Dict[str, Any] = {}

    def _decision(v: Dict[str, Any]) -> Optional[float]:
        return v.get("decision_state", {}).get("skill_vs_best_naive")

    iid = prediction.get(iid_split)
    if iid:
        comp["interpolation"] = clip01(_decision(iid))
        raw["interpolation_decision_skill"] = _decision(iid)
        raw["interpolation_groups"] = {
            g: d.get("skill_vs_best_naive")
            for g, d in iid.get("groups", {}).items()}
        raw["price_level_canary_iid"] = iid.get("groups", {}).get(
            "price_level", {}).get("skill_vs_best_naive")

    ood = {k: v for k, v in prediction.items() if k.startswith(ood_prefix)}
    if ood:
        vals = [_decision(v) for v in ood.values()]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        if vals:
            raw["extrapolation_skill_mean"] = float(np.mean(vals))
            raw["extrapolation_skill_min"] = float(np.min(vals))
            raw["extrapolation_per_split"] = {k: _decision(v) for k, v in ood.items()}
            comp["extrapolation"] = clip01(float(np.mean(vals)))
        canary = [v.get("groups", {}).get("price_level", {}).get("skill_vs_best_naive")
                  for v in ood.values()]
        canary = [c for c in canary if c is not None and math.isfinite(c)]
        if canary:
            raw["price_level_canary_ood_max"] = float(np.max(canary))

    # Calibration pooled over in-distribution and held-out worlds: a model
    # calibrated only in-distribution has not solved this problem.
    bss: List[float] = []
    for v in prediction.values():
        ev = v.get("event")
        if not ev:
            continue
        rates = [d["base_rate"] for d in ev["per_event"].values()]
        bss.append(brier_skill(ev["brier"], rates))
    if bss:
        raw["event_brier_skill_mean"] = float(np.mean(bss))
        comp["event_calibration"] = clip01(float(np.mean(bss)))

    hs = [v.get("long_horizon", {}) for v in prediction.values()]
    depth = horizon_key
    if depth is None:
        keys = sorted({int(k) for h in hs for k in h}, reverse=True)
        depth = str(keys[0]) if keys else None
    if depth is not None:
        vals = [h[depth].get("decision_skill_vs_best_naive")
                for h in hs if depth in h]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        if vals:
            raw["long_horizon_skill_mean"] = float(np.mean(vals))
            raw["long_horizon_depth"] = int(depth)
            comp["long_horizon"] = clip01(float(np.mean(vals)))

    if counterfactual:
        vals = [v["skill_vs_zero_effect"] for v in counterfactual.values()
                if isinstance(v, dict) and "skill_vs_zero_effect" in v]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        if vals:
            raw["counterfactual_skill_mean"] = float(np.mean(vals))
            raw["counterfactual_per_setting"] = {
                k: v.get("skill_vs_zero_effect") for k, v in counterfactual.items()
                if isinstance(v, dict)}
            comp["counterfactual"] = clip01(float(np.mean(vals)))

    if control and control.get("scaled_return") is not None:
        raw["control_return"] = control.get("total_return")
        raw["control_scaled"] = control.get("scaled_return")
        comp["planning"] = clip01(control.get("scaled_return"))

    present = [v for v in comp.values() if v is not None]
    return {
        "components": comp,
        "raw": raw,
        "coverage": len(present) / len(COMPONENTS),
        "composite": float(np.mean(present)) if present else None,
        "n_components": len(present),
    }


# --------------------------------------------------------------------------
def prediction_table(results: Dict[str, Dict[str, Dict[str, Any]]],
                     splits: Sequence[str]) -> List[Dict[str, Any]]:
    """One row per model: decision-state skill per split, group detail, events."""
    rows: List[Dict[str, Any]] = []
    for model, per_split in results.items():
        row: Dict[str, Any] = {"model": model}
        for sp in splits:
            v = per_split.get(sp)
            row[f"dskill@{sp}"] = (None if not v else
                                   v.get("decision_state", {}).get("skill_vs_best_naive"))
        iid = per_split.get("test_iid")
        if iid:
            g = iid.get("groups", {})
            for name in ("agent", "protocol", "population", "price", "price_level"):
                row[f"{name}_skill_iid"] = g.get(name, {}).get("skill_vs_best_naive")
            row["next_state_skill_pers_iid"] = iid["next_state"].get(
                "skill_vs_persistence")
            row["event_logloss_iid"] = iid["event"].get("log_loss")
            row["event_brier_iid"] = iid["event"].get("brier")
            row["event_ece_iid"] = iid["event"].get("ece")
            row["event_auroc_iid"] = iid["event"].get("mean_auroc")
            row["reward_skill_iid"] = iid["reward"].get("skill_vs_mean")
        ood_vals = [v.get("decision_state", {}).get("skill_vs_best_naive")
                    for k, v in per_split.items() if k.startswith("ood_") and v]
        ood_vals = [v for v in ood_vals if v is not None and math.isfinite(v)]
        row["dskill_ood_mean"] = float(np.mean(ood_vals)) if ood_vals else None
        row["dskill_ood_min"] = float(np.min(ood_vals)) if ood_vals else None
        base = row.get("dskill@test_iid")
        # Retention is only meaningful when the in-distribution score is
        # materially positive; otherwise the ratio amplifies noise.
        row["ood_retention"] = (float(np.mean(ood_vals)) / base
                                if base is not None and base > 0.05 and ood_vals
                                else None)
        rows.append(row)
    return rows


def control_table(summaries: Dict[str, Dict[str, Dict[str, Any]]],
                  scenarios: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for policy, per_sc in summaries.items():
        row: Dict[str, Any] = {"policy": policy}
        rets, sharpes, dds = [], [], []
        for sc in scenarios:
            s = per_sc.get(sc)
            if not s:
                continue
            row[f"ret@{sc}"] = s.get("total_return")
            rets.append(s.get("total_return", 0.0))
            sharpes.append(s.get("sharpe", 0.0))
            dds.append(s.get("max_drawdown", 0.0))
        row["return_mean"] = float(np.mean(rets)) if rets else None
        row["sharpe_mean"] = float(np.mean(sharpes)) if sharpes else None
        row["max_drawdown_mean"] = float(np.mean(dds)) if dds else None
        rows.append(row)
    return rows


def counterfactual_table(results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model, per_setting in results.items():
        row: Dict[str, Any] = {"model": model}
        skills, signs = [], []
        for name, v in per_setting.items():
            if not isinstance(v, dict) or "skill_vs_zero_effect" not in v:
                continue
            row[f"cf_skill@{name}"] = v["skill_vs_zero_effect"]
            skills.append(v["skill_vs_zero_effect"])
            signs.append(v.get("sign_accuracy_material", float("nan")))
        row["cf_skill_mean"] = float(np.mean(skills)) if skills else None
        finite = [s for s in signs if s is not None and math.isfinite(s)]
        row["sign_acc_material_mean"] = float(np.mean(finite)) if finite else None
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "--"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "--"
        return f"{v:.{nd}f}"
    return str(v)


def to_markdown(rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None,
                nd: int = 4, sort_by: Optional[str] = None,
                descending: bool = True) -> str:
    if not rows:
        return "_(no rows)_"
    cols = list(columns) if columns else list(rows[0].keys())
    data = list(rows)
    if sort_by and sort_by in cols:
        def key(r):
            v = r.get(sort_by)
            return (v is None, -(v if isinstance(v, (int, float)) and descending
                                 else 0) if isinstance(v, (int, float)) else 0)
        data = sorted(data, key=lambda r: (r.get(sort_by) is None,
                                           -(r.get(sort_by) or 0) if descending
                                           else (r.get(sort_by) or 0)))
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in data:
        out.append("| " + " | ".join(_fmt(r.get(c), nd) for c in cols) + " |")
    return "\n".join(out)


def to_csv(rows: Sequence[Dict[str, Any]], path: str,
           columns: Optional[Sequence[str]] = None) -> None:
    import csv
    import os
    if not rows:
        return
    cols = list(columns) if columns else list(rows[0].keys())
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
